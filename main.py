import argparse, os, json, shutil
import pandas as pd
import numpy as np
import torch
from scipy.stats import norm
from collections import Counter
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="This DataLoader will create")
import torch.nn.functional as F

from data import load_data
from utils import ClipLoss, instantiate_from_config, multimodal_similarity_modulation, update_labels, save_results_to_csv

device_id = 0 if torch.cuda.is_available() else 'cpu'

def load_model(config, train_loader, test_loader):
    model_dict = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model_dict[k] = instantiate_from_config(v)
    return PLModel(model_dict, config, train_loader, test_loader)

class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader):
        super().__init__()
        self.config = config
        for key, value in model.items():
            setattr(self, key, value)
        self.criterion = ClipLoss()
        self.all_predicted_classes = []
        self.all_true_labels = []
        self.z_dim = self.config['z_dim']
        num_train = len(train_loader.dataset)
        self.sim_text = np.ones(num_train)
        self.sim_eeg = np.ones(num_train)
        self.match_label_text = np.ones(num_train, dtype=int)
        self.match_label_eeg = np.ones(num_train, dtype=int)
        self.alpha_text = 0.1
        self.alpha_eeg = 0.1
        self.gamma = 0.3
        self.mAP_total = 0.0
        self.match_similarities = []

    def forward(self, batch, sample_posterior=False):
        idx = batch['idx'].cpu().numpy() 
        eeg = batch['eeg']
        eeg_z_txt, eeg_z_img = self.brain(eeg, self.training)
        img_z = F.normalize(batch['img_features'], dim=-1)
        txt_z = F.normalize(batch['text_features'], dim=-1)
        logit_scale = self.brain.softplus(self.brain.logit_scale)
        eeg_loss_txt, txt_loss, logits_per_text = self.criterion(eeg_z_txt, txt_z, logit_scale)
        eeg_loss_img, img_loss, logits_per_image = self.criterion(eeg_z_img, img_z, logit_scale)
        total_loss = (eeg_loss_img.mean() + img_loss.mean() + eeg_loss_txt.mean() + txt_loss.mean()) / 4
        if self.config['data'].get('uncertainty_decouple'):
            if self.training:
                self.sim_text[idx], self.match_label_text[idx] = update_labels(
                    logits_per_text, self.sim_text, self.alpha_text, self.gamma, idx)
                self.sim_eeg[idx], self.match_label_eeg[idx] = update_labels(
                    logits_per_image, self.sim_eeg, self.alpha_eeg, self.gamma, idx)
        return eeg_z_img, eeg_z_txt, img_z, txt_z, total_loss

    def _update_predictions(self, eeg_z_img, eeg_z_txt, img_z, txt_z):
        similarity = multimodal_similarity_modulation(eeg_z_img, eeg_z_txt, img_z, txt_z)
        _, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        self.all_true_labels.append(np.arange(similarity.size(0)))
        return similarity

    def _compute_and_log_metrics(self, stage):
        preds = np.concatenate(self.all_predicted_classes, axis=0)
        labels = np.concatenate(self.all_true_labels, axis=0)
        top1_acc = (preds[:, 0] == labels).mean()
        top5_acc = (preds == labels[:, None]).any(axis=1).mean()
        self.log(f'{stage}_top1_acc', top1_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_top5_acc', top5_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.all_predicted_classes.clear()
        self.all_true_labels.clear()
        return top1_acc, top5_acc, len(labels)

    def training_step(self, batch, batch_idx):
        eeg_z_img, eeg_z_txt, img_z, txt_z, loss = self(batch, sample_posterior=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch['idx'].size(0))
        eeg_z_img = F.normalize(eeg_z_img, dim=-1)
        eeg_z_txt = F.normalize(eeg_z_txt, dim=-1)
        self._update_predictions(eeg_z_img, eeg_z_txt, img_z, txt_z)
        if batch_idx == self.trainer.num_training_batches - 1:
            self._compute_and_log_metrics('train')
            self.trainer.train_dataloader.dataset.match_label_text = self.match_label_text
            self.trainer.train_dataloader.dataset.match_label_eeg = self.match_label_eeg
        return loss

    def validation_step(self, batch, batch_idx):
        eeg_z_img, eeg_z_txt, img_z, txt_z, loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch['idx'].size(0))
        eeg_z_img = F.normalize(eeg_z_img, dim=-1)
        eeg_z_txt = F.normalize(eeg_z_txt, dim=-1)
        self._update_predictions(eeg_z_img, eeg_z_txt, img_z, txt_z)
        return loss
    
    def on_validation_epoch_end(self):
        self._compute_and_log_metrics('val')

    def test_step(self, batch, batch_idx):
        eeg_z_img, eeg_z_txt, img_z, txt_z, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch['idx'].size(0))
        eeg_z_img = F.normalize(eeg_z_img, dim=-1)
        eeg_z_txt = F.normalize(eeg_z_txt, dim=-1)
        similarity = self._update_predictions(eeg_z_img, eeg_z_txt, img_z, txt_z)
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())
        sorted_indices = torch.argsort(-similarity, dim=1)
        targets = torch.arange(similarity.size(0), device=self.device).unsqueeze(1)
        ranks = (sorted_indices == targets).nonzero(as_tuple=True)[1] + 1
        self.mAP_total += (1.0 / ranks.float()).sum().item()
        return loss
        
    def on_test_epoch_end(self):
        top1_acc, top5_acc, total_samples = self._compute_and_log_metrics('test')
        mAP = self.mAP_total / total_samples
        match_sim_mean = np.mean(self.match_similarities) if self.match_similarities else 0
        self.log('mAP', mAP, sync_dist=True)
        self.log('similarity', match_sim_mean, sync_dist=True)
        self.match_similarities.clear()
        self.mAP_total = 0
        return {
            'test_loss': self.trainer.callback_metrics['test_loss'].item(), 
            'test_top1_acc': top1_acc, 
            'test_top5_acc': top5_acc, 
            'mAP': mAP, 
            'similarity': match_sim_mean
        }
        
    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.config['train']['optimizer'])
        return [optimizer_class(self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4)]

def get_config(args, subject):
    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }
    z_dim = pretrain_map[args.vision_backbone]['z_dim']
    config = {
        'seed': args.seed,
        'name': f"{args.method}_{args.modality}_{args.exp_setting}_{args.brain_backbone}_{args.vision_backbone}",
        'exp_setting': args.exp_setting,
        'z_dim': z_dim,
        's': 6,
        'c': 0.5,
        'save_dir': 'exp',
        'train': {'epoch': args.epoch, 'optimizer': 'AdamW', 'lr': args.lr},
        'models': {
            'brain': {'target': f"model.{args.brain_backbone}", 'params': {'z_dim': z_dim}}
        },
        'data': {
            'modality': args.modality,
            'subjects': [subject],
            'model_type': args.vision_backbone,
            'train_batch_size': 1024,
            'val_batch_size': 200,
            'test_batch_size': 200,
            'train_avg': True,
            'test_avg': True
        }
    }
    if args.modality == 'eeg':
        config['timesteps'] = [0, 250]
        config['data']['timesteps'] = [0, 250]
        config['data']['data_dir'] = '../data/THINGS-EEG/Preprocessed_data_250Hz_whiten'
        config['data']['selected_ch'] = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
        config['models']['brain']['params']['c_num'] = 17
        config['models']['brain']['params']['timesteps'] = [0, 250]
    else:
        config['timesteps'] = [0, 201]
        config['data']['timesteps'] = [0, 201]
        config['data']['data_dir'] = '../data/THINGS-MEG/preprocessed_data'
        config['data']['selected_ch'] = None
        config['models']['brain']['params']['c_num'] = 271
        config['models']['brain']['params']['timesteps'] = [0, 201]
    if args.method == 'baseline':
        config['data']['uncertainty_decouple'] = False
        config['data']['blur_type'] = {'target': 'utils.NoBlur', 'params': {}}
    else:
        config['data']['uncertainty_decouple'] = True
        config['data']['blur_type'] = {
            'target': 'utils.GaussianBlur',
            'params': {'h': 224, 'w': 224, 'blur_center_size': 0.5, 'blur_kernel_size': 51, 'curve_type': 'exp', 'system_g': 3}
        }
    return config

def run_experiment(config):
    seed_everything(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)
    logger_name = config['name']
    logger_version = f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}"
    logger = TensorBoardLogger(config['save_dir'], name=logger_name, version=logger_version)
    os.makedirs(logger.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = load_data(config)
    pl_model = load_model(config, train_loader, test_loader)
    checkpoint_callback_top1 = ModelCheckpoint(save_last=False, monitor='val_top1_acc', mode='max', save_top_k=1, filename='best-top1')
    checkpoint_callback_top5 = ModelCheckpoint(save_last=False, monitor='val_top5_acc', mode='max', save_top_k=1, filename='best-top5')
    es_monitor = 'val_top1_acc' if config['exp_setting'] == 'inter-subject' else 'train_loss'
    es_mode = 'max' if config['exp_setting'] == 'inter-subject' else 'min'
    early_stop_callback = EarlyStopping(monitor=es_monitor, min_delta=0.001, patience=5, verbose=False, mode=es_mode)
    trainer = Trainer(
        log_every_n_steps=10, 
        strategy="auto",
        callbacks=[checkpoint_callback_top1, checkpoint_callback_top5, early_stop_callback],
        max_epochs=config['train']['epoch'], 
        devices=[device_id] if device_id != 'cpu' else 'auto',
        accelerator='cuda' if device_id != 'cpu' else 'cpu',
        logger=logger
    )
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results_top1 = trainer.test(ckpt_path='best', dataloaders=test_loader)
    # test_results_top1_last = trainer.test(ckpt_path='last', dataloaders=test_loader)
    test_results_top5 = trainer.test(ckpt_path=checkpoint_callback_top5.best_model_path, dataloaders=test_loader)
    # test_results_top5_last = trainer.test(ckpt_path='last', dataloaders=test_loader)
    test_results_top1[0]['test_top5_acc'] = test_results_top5[0]['test_top5_acc']
    with open(os.path.join(logger.log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results_top1, f, indent=4)
    result_filename = f"{config['name']}.csv"
    csv_path = os.path.join('results', result_filename)
    save_results_to_csv(csv_path, test_results_top1[0])

def main():
    parser = argparse.ArgumentParser(description="Multi-subject EEG/MEG representation learning")
    parser.add_argument("--method", type=str, default="decouple", choices=["decouple", "baseline"], help="Method choice (decouple or baseline)")
    parser.add_argument("--modality", type=str, default="eeg", choices=["eeg", "meg"], help="Data modality (eeg or meg)")
    parser.add_argument("--exp_setting", type=str, default="intra-subject", choices=["intra-subject", "inter-subject"], help="Experiment setting")
    parser.add_argument("--brain_backbone", type=str, default="EEGProject", help="Brain signal backbone network")
    parser.add_argument("--vision_backbone", type=str, default="RN50", help="Vision backbone network from CLIP")
    parser.add_argument("--epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args()
    num_subjects = 10 if args.modality == 'eeg' else 4
    for i in range(1, num_subjects + 1):
        subject = f"sub-{i:02d}"
        print(f"\n{'='*20} Running Subject: {subject} ({args.method} / {args.modality}) {'='*20}\n")
        config = get_config(args, subject)
        run_experiment(config)

if __name__ == "__main__":
    main()
