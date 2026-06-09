import torch, os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
from torchvision import transforms
from utils import instantiate_from_config 
import copy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_data(config):
    modality = config['data'].get('modality', 'eeg')
    exp_setting = config.get('exp_setting', 'intra-subject')
    DatasetClass = EEGDataset if modality == 'eeg' else MEGDataset
    if exp_setting == 'intra-subject':
        test_dataset = DatasetClass(config, mode='test')
        print(f'init test_dataset ({modality}) success')
        train_dataset = DatasetClass(config, mode='train')
        print(f'init train_dataset ({modality}) success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False, num_workers=25, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32, pin_memory=True)
        return train_loader, test_loader, test_loader
    
    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = DatasetClass(config, mode='test')
        print(f'init test_dataset ({modality}) success')
        total_sub = 10 if modality == 'eeg' else 4
        all_subjects = [f'sub-{i:02}' for i in range(1, total_sub + 1)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = copy.deepcopy(config)
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = DatasetClass(leave_one_subjects_config, mode='test')
        print(f'init val_dataset ({modality}) success')
        train_dataset = DatasetClass(leave_one_subjects_config, mode='train')
        print(f'init train_dataset ({modality}) success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False, num_workers=25)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False, drop_last=False, num_workers=32)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32)
        return train_loader, val_loader, test_loader 

def simplify_path(full_path):
    try:
        parts = full_path.split('/')
        images_folder_idx = -1
        for i, part in enumerate(parts):
            if part in ['test_images', 'training_images']:
                images_folder_idx = i
                break
        if images_folder_idx == -1:
            return full_path
        class_folder = parts[images_folder_idx + 1]
        image_name = parts[images_folder_idx + 2]
        class_name = '_'.join(class_folder.split('_')[1:])
        class_text = class_name.replace('_', ' ')
        return f"test_images/{class_name}/{image_name}", class_text
    except (ValueError, IndexError):
        return full_path

class BaseBrainDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.subjects = config['data']['subjects']
        print(f'subjects: {self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.avg = config['data'][f"{mode}_avg"]
        self.blur_type = config['data']['blur_type']
        self.timesteps = config['data']['timesteps']
        self.s = config.get('s', 6)
        self.c = config.get('c', 0.5)
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                         'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                         'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                         'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                         'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                         'O1', 'Oz', 'O2']
        if self.selected_ch == "None" or self.selected_ch is None:
            if self.config['data'].get('modality', 'eeg') == 'eeg':
                self.selected_ch = self.channels
            else:
                self.selected_ch = None
        self.n_cls = 1654 if self.mode == 'train' else 200
        if self.config['data']['uncertainty_decouple']:
            self.blur_transform = {}
            for cdis, time in zip([self.c, 0, -self.c], ['max', 'mid', 'min']):
                self.blur_transform[time] = {}
                for shift, tag in zip([-self.s, 0, self.s], ['low', 'medium', 'high']):
                    blur_param = copy.deepcopy(config['data']['blur_type'])
                    blur_param['params']['blur_kernel_size'] = blur_param['params']['blur_kernel_size'] + shift
                    blur_param['params']['blur_center_size'] = blur_param['params']['blur_center_size'] + cdis
                    self.blur_transform[time][tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])
        process_term = [transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
        self.process_transform = transforms.Compose(process_term)

    def load_common_features(self, features_filename, pretrain_map):
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
        else:
            device_id = 0 if torch.cuda.is_available() else 'cpu'
            self.vlmodel, self.preprocess, _ = open_clip.create_model_and_transforms(self.model_type, device=f"cuda:{device_id}", pretrained=pretrain_map[self.model_type]['pretrained'])
            for param in self.vlmodel.parameters():
                param.requires_grad = False
            self.vlmodel.eval()
            if self.config['data']['uncertainty_decouple']:
                self.img_features = {}
                for time in ['max', 'mid', 'min']:
                    self.img_features[time] = {}
                    for tag in ['low', 'medium', 'high']:
                        self.img_features[time][tag] = self.ImageEncoder(self.loaded_data[0]['img'], self.blur_transform[time][tag])
                all_features = []
                for time in ['max', 'mid', 'min']:
                    for tag in ['low', 'medium', 'high']:
                        all_features.append(self.img_features[time][tag])
                self.img_features['avg'] = {
                    k: sum(feat[k] for feat in all_features) / len(all_features)
                    for k in self.img_features['max']['medium']
                }
            else:
                self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
            self.text_features = self.Textencoder(self.loaded_data[0]['text'])
            torch.save({
                'text_features': self.text_features,
                'img_features': self.img_features,
            }, features_filename)
            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform=None):
        if blur_transform is None:
            blur_transform = self.blur_transform
        self.vlmodel.eval()
        set_images = list(set(images))
        set_images.sort()
        batch_size = 128
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size), desc="Encoding Images"):
            batch_images = set_images[i:i + batch_size]
            device = next(self.vlmodel.parameters()).device
            ele = [self.process_transform(blur_transform(Image.open(os.path.join(self.data_dir, '../Image_set_Resize', img)).convert("RGB"))) for img in batch_images]
            image_inputs = torch.stack(ele).to(device)
            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        return {set_images[i]: image_features[i].float().cpu() for i in range(len(set_images))}
    
    @torch.no_grad()
    def Textencoder(self, text):   
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"A photo of a {t}.") for t in set_text])
        device = next(self.vlmodel.parameters()).device
        text_inputs = text_inputs.to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return {set_text[i]: text_features[i].float().cpu() for i in range(len(set_text))}
    
    def __len__(self):
        return self.trial_all_subjects

class EEGDataset(BaseBrainDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.per_trials = 4 if self.mode == 'train' else 80
        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)
        self.match_label_text = np.ones(self.trial_all_subjects, dtype=int)
        self.match_label_eeg = np.ones(self.trial_all_subjects, dtype=int)
        data_dir = os.path.join(self.data_dir, '../Image_feature', f"{config['data']['blur_type']['target'].rsplit('.', 1)[-1]}")
        os.makedirs(data_dir, exist_ok=True)
        features_filename = os.path.join(data_dir, f"{self.name}_{mode}.pt")
        pretrain_map = {
            'RN50': {'pretrained': 'openai', 'resize': (224, 224)},
            'RN101': {'pretrained': 'openai', 'resize': (224, 224)},
            'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224)},
            'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224)},
            'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224)},
            'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224)},
            'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224)},
            'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224)},
        }
        self.load_common_features(features_filename, pretrain_map)

    def load_data(self, data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ', 1)[-1]}----")
        loaded_data = torch.load(data_path)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]
        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:, 0]
            avg_data['img'] = loaded_data['img'][:, 0]
            avg_data['text'] = loaded_data['text'][:, 0]
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data
        for k, v in loaded_data.items():
            if k in ['eeg', 'label', 'img', 'text', 'session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data    

    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject
        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        eeg_mean = eeg if self.avg else self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()
        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]
        match_label_text = self.match_label_text[index]
        match_label_eeg = self.match_label_eeg[index]
        if self.config['data']['uncertainty_decouple']:
            if self.mode == 'train':
                time = 'min' if match_label_text == 0 else ('max' if match_label_text == 2 else 'mid')
                tag = 'low' if match_label_eeg == 0 else ('high' if match_label_eeg == 2 else 'medium')
                img_features = self.img_features[time][tag][img_path]
                text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
            else:
                img_features = torch.cat([self.img_features[t]['medium'][img_path] for t in ['max', 'mid', 'min']], dim=0)
                text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
                text_features = torch.cat([text_features, text_features, text_features], dim=0)
        else:
            img_features = self.img_features[img_path]
            text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        text = f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        return {
            'idx': index,
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'label': label,
            'match_label_text': match_label_text,
            'match_label_eeg': match_label_eeg,
            'img_path': img_path,
            'img': 'None',
            'img_features': img_features,
            'text': text,
            'text_features': text_features,
            'session': self.loaded_data[subject]['session'][trial_index],
            'subject': subject,
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }

class MEGDataset(BaseBrainDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.per_trials = 1 if self.mode == 'train' else 12
        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]
        for i in range(len(self.loaded_data)):
            self.loaded_data[i]['img'] = np.array([simplify_path(path)[0] for path in self.loaded_data[i]['img']], dtype=self.loaded_data[i]['img'].dtype)
            self.loaded_data[i]['text'] = np.array([simplify_path(path)[1] for path in self.loaded_data[i]['img']], dtype=self.loaded_data[i]['img'].dtype)
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)
        self.match_label_text = np.ones(self.trial_all_subjects, dtype=int)
        self.match_label_eeg = np.ones(self.trial_all_subjects, dtype=int)
        data_dir = os.path.join(self.data_dir, '../Image_feature', f"{config['data']['blur_type']['target'].rsplit('.', 1)[-1]}")
        os.makedirs(data_dir, exist_ok=True)
        features_filename = os.path.join(data_dir, f"{self.name}_{mode}.pt")
        pretrain_map = {
            'RN50': {'pretrained': 'openai', 'resize': (224, 224)},
            'RN101': {'pretrained': 'openai', 'resize': (224, 224)},
            'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224)},
            'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224)},
            'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224)},
            'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224)},
            'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224)},
            'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224)},
        }
        self.load_common_features(features_filename, pretrain_map)

    def load_data(self, data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ', 1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]
        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['img'] = np.array(loaded_data['img'])
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['img'] = loaded_data['img'].reshape(-1)
            loaded_data = _data
        return loaded_data    

    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject
        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        eeg_mean = eeg if self.avg else self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()
        img_path = self.loaded_data[subject]['img'][trial_index]
        match_label_text = self.match_label_text[index]
        match_label_eeg = self.match_label_eeg[index]
        if self.config['data']['uncertainty_decouple']:
            if self.mode == 'train':
                time = 'min' if match_label_text == 0 else ('max' if match_label_text == 2 else 'mid')
                tag = 'low' if match_label_eeg == 0 else ('high' if match_label_eeg == 2 else 'medium')
                img_features = self.img_features[time][tag][img_path]
                text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
            else:
                img_features = torch.cat([self.img_features[t]['medium'][img_path] for t in ['max', 'mid', 'min']], dim=0)
                text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
                text_features = torch.cat([text_features, text_features, text_features], dim=0)
        else:
            img_features = self.img_features[img_path]
            text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        text = f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        return {
            'idx': index,
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'match_label_text': match_label_text,
            'match_label_eeg': match_label_eeg,
            'img_path': img_path,
            'img': 'None',
            'img_features': img_features,
            'text': text,
            'text_features': text_features,
            'subject': subject,
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }