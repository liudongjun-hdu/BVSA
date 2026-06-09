import random
import numpy as np
import torch
import importlib
import cv2
from PIL import Image
from torch import nn
from torch.nn import functional as F
from scipy.stats import norm
from scipy.optimize import fsolve, newton
import os
import pandas as pd

def multimodal_similarity_modulation(eeg_z_img, eeg_z_txt, img_z, txt_z, confidence_level=0.95, return_threshold=False):
    eeg_visual_sim = eeg_z_img @ img_z.T
    eeg_semantic_sim = eeg_z_txt @ txt_z.T
    diagonal_semantic = torch.diagonal(eeg_semantic_sim).cpu().detach().numpy()
    batch_sim = diagonal_semantic
    mean_sim = np.mean(batch_sim)
    std_sim = np.std(batch_sim, ddof=1)
    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - alpha / 2)
    threshold = mean_sim - z_score * std_sim / np.sqrt(len(batch_sim))
    sample_mask = (torch.tensor(batch_sim) > threshold).float().to(eeg_semantic_sim.device)
    enhancement_mask = sample_mask.unsqueeze(1).expand_as(eeg_semantic_sim)
    enhancement = eeg_visual_sim * eeg_semantic_sim * enhancement_mask
    similarity = eeg_visual_sim + enhancement
    if return_threshold:
        return similarity, threshold, batch_sim
    return similarity

def update_labels(logits, sim_array, alpha, gamma, idx):
    diag_elements = torch.diagonal(logits).cpu().detach().numpy()
    batch_sim = gamma * diag_elements + (1 - gamma) * sim_array[idx]
    mean_sim = np.mean(batch_sim)
    std_sim = np.std(batch_sim, ddof=1)
    match_label = np.ones_like(batch_sim)
    z_alpha_2 = norm.ppf(1 - alpha / 2)
    lower = mean_sim - z_alpha_2 * std_sim
    upper = mean_sim + z_alpha_2 * std_sim
    match_label[diag_elements > upper] = 0
    match_label[diag_elements < lower] = 2
    return batch_sim, match_label

def save_results_to_csv(file_path, results_dict):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    columns = ['test_loss', 'test_top1_acc', 'test_top5_acc', 'mAP', 'similarity']
    if not os.path.exists(file_path):
        df = pd.DataFrame(index=columns)
        df['Run_1'] = [results_dict[col] for col in columns]
    else:
        df = pd.read_csv(file_path, index_col=0)
        next_run_num = len(df.columns) + 1
        for col in columns:
            df.loc[col, f'Run_{next_run_num}'] = results_dict[col]
    df.to_csv(file_path)

def instantiate_from_config(config, reload=False):
    if config in ('__is_first_stage__', '__is_unconditional__'):
        return None
    if not isinstance(config, dict) or "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module_path, cls_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    if reload:
        importlib.reload(module)
    obj_cls = getattr(module, cls_name)
    params = config.get("params") or {}
    return obj_cls(**params)

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
       
    def compute_ranking_weights(self, loss_list):
        ranks = torch.argsort(torch.argsort(loss_list))
        weights = 1.0 / (ranks.float() + 1.0)
        return weights
    
    def forward(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)
        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')
        return image_loss, text_loss, logits_per_image

class NoBlur:
    def __call__(self, x):
        return x

class GaussianBlur:
    def __init__(self, h, w, blur_center_size, blur_kernel_size, curve_type='exp', **kwargs):
        self.blur_kernel_size = blur_kernel_size
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt((h - center_y - 1)**2 + (w - center_x - 1)**2)
        x0 = np.clip(np.sqrt((Y - center_y)**2 + (X - center_x)**2) / max_dist, 0.0, 1.0)
        if curve_type == 'linear':
            y0 = 1.0 - x0
        elif curve_type == 'exp':
            y0 = np.exp(-kwargs.get('system_g', 4) * x0)
        elif curve_type == 'quadratic':
            y0 = 1.0 - x0**2
        elif curve_type == 'log':
            b = 1.0 / (np.e - 1)
            y0 = np.log(b) + 1.0 - np.log(x0 + b)
        elif curve_type == 'brachistochrone':
            def eq(vars): return [vars[1]*(vars[0]-np.sin(vars[0]))-1, -vars[1]*(1-np.cos(vars[0]))+1]
            _, r = fsolve(eq, [1.0, 1.0])
            t0 = newton(lambda t: t - np.sin(t) - x0/r, np.ones_like(x0), fprime=lambda t: 1 - np.cos(t))
            y0 = -r * (1.0 - np.cos(t0)) + 1.0
        else:
            y0 = np.zeros_like(x0)
        self.mask = ((1.0 - blur_center_size) * y0).astype(np.float32)

    def __call__(self, img, blur_kernel_size=None): 
        k_size = blur_kernel_size or self.blur_kernel_size
        img_np = np.array(img)
        is_rgb = img_np.ndim == 3 and img_np.shape[2] == 3
        if is_rgb:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(img_np, (k_size, k_size), 0)
        alpha = (1.0 - self.mask)[..., np.newaxis] if img_np.ndim == 3 else (1.0 - self.mask)
        blended = cv2.convertScaleAbs(img_np * (1.0 - alpha) + blurred * alpha)
        if is_rgb:
            blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)