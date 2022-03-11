import torch
import torch.nn as nn
from utils import calculate_padding


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__() 
        
        padding = calculate_padding(config['kernel_size'], 
                                    config['stride'])
        self.layer_1 = nn.Sequential(
            nn.Conv2d(**config['layer_1'],
                      stride=config['stride'],
                      kernel_size=config['kernel_size'],
                      padding=padding),
            nn.BatchNorm2d(config['layer_1']['out_channels']),
            nn.ReLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(**config['layer_2'],
                      stride=config['stride'],
                      kernel_size=config['kernel_size'],
                      padding=padding),
            nn.BatchNorm2d(config['layer_2']['out_channels']),
            nn.ReLU(),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(**config['layer_3'],
                      stride=config['stride'],
                      kernel_size=config['kernel_size'],
                      padding=padding),
            nn.BatchNorm2d(config['layer_3']['out_channels']),
            nn.ReLU(),
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(**config['layer_4'],
                      stride=config['stride'],
                      kernel_size=config['kernel_size'],
                      padding=padding),
            nn.BatchNorm2d(config['layer_4']['out_channels']),
            nn.ReLU(),
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(**config['layer_5'],
                      stride=config['stride'],
                      kernel_size=config['kernel_size'],
                      padding=padding),
            nn.BatchNorm2d(config['layer_5']['out_channels']),
            nn.ReLU(),
        )
        self.layer_6 = nn.Sequential(
            nn.Conv2d(**config['layer_6'],
                      stride=config['stride'],
                      kernel_size=config['kernel_size'],
                      padding=padding),
            nn.BatchNorm2d(config['layer_6']['out_channels']),
            nn.ReLU()
        )
        
    def forward(self, img):
        out_1 = self.layer_1(img)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        out_4 = self.layer_4(out_3)
        out_5 = self.layer_5(out_4)
        out_6 = self.layer_6(out_5)
        return [out_1, out_2, out_3, out_4, out_5, out_6]
    
    
class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.layers = nn.Sequential(
            # Layer 1
            nn.Conv1d(**config['layer_1']),
            nn.BatchNorm1d(config['layer_1']['out_channels']),
            nn.ReLU(),
            # Layer 2
            nn.Conv1d(**config['layer_2']),
            nn.BatchNorm1d(config['layer_2']['out_channels']),
            nn.ReLU(),
            # Layer 3
            nn.Conv1d(**config['layer_3']),
            nn.BatchNorm1d(config['layer_3']['out_channels']),
            nn.ReLU(),
            # Layer 4
            nn.Conv1d(**config['layer_4']),
            nn.BatchNorm1d(config['layer_4']['out_channels']),
            nn.ReLU(),
            # Layer 5
            nn.Conv1d(**config['layer_5']),
            nn.BatchNorm1d(config['layer_5']['out_channels']),
            nn.ReLU(),
        )
        self.encoder = nn.GRU(**config['gru'], batch_first=True)
        
    def forward(self, audio):
        '''
        audio: [BS, N, L]
        z: [BS, L, N]
        out: [BS, N, L]
        '''
        z = self.layers(audio)
        z, h_0 = self.encoder(z.transpose(1,2))
        return z.transpose(1,2)


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.image_encoder = ImageEncoder(config['image_encoder'])
        self.audio_encoder = AudioEncoder(config['audio_encoder'])
        self.noise_encoder = nn.GRU(**config['noise_generator'])
        
    def forward(self, img, audio):
        BS, L, N = audio.size()
        noise = torch.normal(mean=0., 
                             std=torch.tensor(0.6),
                             size=(BS, 1, 10))
        noise_z, h_0 = self.noise_encoder(noise)
        audio_z = self.audio_encoder(audio)
        img_zs = self.image_encoder(img)
        print(img_zs[-1].size(), audio_z.size(), noise_z.size())
        gen_out = torch.cat((img_zs[-1], audio_z, noise_z))
        return img_zs
    
import yaml

with open('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/avatar/realistic/configs/models.yaml', 'r') as f:
    config=  yaml.load(f.read(), Loader=yaml.FullLoader)
    
gen = Generator(config['generator'])

img = torch.randn((3, 3, 96,128))
audio = torch.randn((1, 80, 500))

img_zs = gen(img, audio)

print(img_zs[-1].size())