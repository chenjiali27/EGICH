import torch
from torch import nn
from torchvision import models
import scipy.misc
import scipy.io
from ops import *
import torch.nn.functional as F
from time import time
import numpy as np

def init_parameters_recursively(layer):
    if isinstance(layer, nn.Sequential):  
        for sub_layer in layer:
            init_parameters_recursively(sub_layer)
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear): 
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, std=0.01)
    else:
        return

class ImageNetV0(nn.Module):
    def __init__(self, cfg):
        super(ImageNetV0, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.numClass = cfg.numClass
        self.dimImg = cfg.dimImg

        self.feature = nn.Sequential(
            nn.Linear(self.dimImg, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, self.SEMANTIC_EMBED),
            nn.BatchNorm1d(self.SEMANTIC_EMBED),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.hash = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.bit),
            nn.Tanh()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)

    def forward(self, inputs):
        fea_I = self.feature(inputs)
        hsh_I = self.hash(fea_I)

        norm = torch.norm(hsh_I, p=2, dim=1, keepdim=True)  
        hsh_I = hsh_I / norm

        return torch.squeeze(fea_I), torch.squeeze(hsh_I)
    
    def get_hash(self, feature):
        hsh_I = self.hash(feature)

        norm = torch.norm(hsh_I, p=2, dim=1, keepdim=True)  
        hsh_I = hsh_I / norm

        return torch.squeeze(hsh_I)


class TextNetV0(nn.Module):
    def __init__(self, cfg):
        super(TextNetV0, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.numClass = cfg.numClass
        self.dimText = cfg.dimText
        
        self.feature = nn.Sequential(
            nn.Linear(self.dimText, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, self.SEMANTIC_EMBED),
            nn.BatchNorm1d(self.SEMANTIC_EMBED),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.hash = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.bit),
            nn.Tanh()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)

    def forward(self, y):
        fea_T = self.feature(y)
        hsh_T = self.hash(fea_T)

        norm = torch.norm(hsh_T, p=2, dim=1, keepdim=True) 
        hsh_T = hsh_T / norm 

        return torch.squeeze(fea_T), torch.squeeze(hsh_T)
    
    def get_hash(self, feature):
        hsh_T = self.hash(feature) 

        norm = torch.norm(hsh_T, p=2, dim=1, keepdim=True) 
        hsh_T = hsh_T / norm 

        return torch.squeeze(hsh_T)
