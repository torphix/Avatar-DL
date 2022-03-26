import torch
import torch.nn as nn


class FilterModel(nn.Module):
    '''Used for auto cropping videos'''
    def __init__(self, config):
        super().__init__() 
        self.layers = nn.ModuleList()
        for n in range(len(config['kernel_size'])):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(config['in_d'] if n == 0 else config['hid_d'],
                              config['hid_d'],
                              config['kernel_size'][n]),
                    nn.BatchNorm2d(config['hid_d']),
                    nn.LeakyReLU()))
            
        self.layers.append(
                nn.Sequential(
                    nn.Conv2d(config['hid_d'],
                              config['out_d'],
                              config['kernel_size'][n]),
                    nn.Sigmoid()))
        
    def forward(self, x):
        return self.layers(x)
    
    
