import torch
import torch.nn as nn
import pytorch_lightning as ptl


class SpeakerClassification(nn.Module):
    def __init__(self, config):
        super().__init__() 
        
        self.layers = nn.ModuleList()
        
    def forward(self, x):

        return