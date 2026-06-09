import torch
import torch.nn as nn
import numpy as np

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)

def _build_proj_block(in_dim, out_dim, drop_rate):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        ResidualAdd(nn.Sequential(
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(drop_rate),
        )),
        nn.LayerNorm(out_dim)
    )

class EEGProject(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super().__init__()
        self.input_dim = c_num * (timesteps[1] - timesteps[0])
        self.model_txt = _build_proj_block(self.input_dim, z_dim, drop_proj)
        self.model_img = _build_proj_block(self.input_dim, z_dim, drop_proj)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x, training):
        x = x.view(x.shape[0], -1) 
        x_txt, x_img = self.model_txt(x), self.model_img(x)
        if training:
            return x_txt, x_img
        return x_txt.repeat(1, 3), x_img.repeat(1, 3)

class BaseModel(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, embedding_dim=1440):
        super().__init__()
        self.backbone = None
        self.project = nn.Sequential(
            nn.Flatten(),
            _build_proj_block(embedding_dim, z_dim, drop_rate=0.5)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.project(self.backbone(x.unsqueeze(1)))

class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (c_num, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
    
class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1400)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10)),
            nn.Conv2d(25, 25, (c_num, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(25, 50, (1, 10)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(50, 100, (1, 10)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(100, 200, (1, 10)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )

class EEGnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1248)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (c_num, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.Dropout2d(0.5)
        )

class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )