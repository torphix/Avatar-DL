import math
import torch
import torch.nn as nn
from utils import calculate_output_length, calculate_padding, prime_factors


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__() 
        
        padding = calculate_padding(config['kernel_size'], 
                                    config['stride'])
        self.layer_1 = self._make_layer(config['layer_1']['in_d'],
                                        config['layer_1']['out_d'],
                                        config['stride'],
                                        config['kernel_size'],
                                        padding)
        self.layer_2 = self._make_layer(config['layer_1']['in_d'],
                                        config['layer_1']['out_d'],
                                        config['stride'],
                                        config['kernel_size'],
                                        padding)
        self.layer_3 = self._make_layer(config['layer_1']['in_d'],
                                        config['layer_1']['out_d'],
                                        config['stride'],
                                        config['kernel_size'],
                                        padding)
        self.layer_4 = self._make_layer(config['layer_1']['in_d'],
                                        config['layer_1']['out_d'],
                                        config['stride'],
                                        config['kernel_size'],
                                        padding)
        self.layer_5 = self._make_layer(config['layer_1']['in_d'],
                                        config['layer_1']['out_d'],
                                        config['stride'],
                                        config['kernel_size'],
                                        padding)
        self.layer_6 = self._make_layer(config['layer_1']['in_d'],
                                        config['layer_1']['out_d'],
                                        config['stride'],
                                        config['kernel_size'],
                                        padding)
        
    def _make_layer(in_d, out_d, stride, kernel_size, pad):
            return nn.Sequential(
                nn.Conv2d(in_d, out_d,
                        stride=stride,
                        padding=pad,
                        kernel_size=kernel_size),
                nn.BatchNorm2d(out_d),
                nn.ReLU())
            
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

        hid_d=config['hid_d']
        out_d=config['out_d']
        features = config['audio_length'] * config['sample_rate']
        strides = prime_factors(features)
        kernels = [2 * strides[i] for i in range(len(strides))]
        paddings = [calculate_padding(kernels[i], strides[i], features)
                    for i in range(len(strides))]

        self.layers = nn.ModuleList()
        for i in range(len(strides)):
            features = calculate_output_length(features, 
                                                kernels[i],
                                                stride=strides[i],
                                                padding=paddings[i])
            # Layer 1
            if i == 0:
                self.layers.append(nn.Sequential(
                    nn.Conv1d(1, hid_d, kernels[i], strides[i], paddings[i]),
                    nn.BatchNorm1d(hid_d),
                    nn.ReLU(),    
                ))
            # Intermediate layers
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv1d(hid_d, hid_d*2, kernels[i], strides[i], paddings[i]),
                    nn.BatchNorm1d(hid_d*2),
                    nn.ReLU(),    
                ))
                hid_d = hid_d*2
            # Output layer
        self.layers.append(nn.Sequential(
                    nn.Conv1d(hid_d, out_d, features),
                    nn.BatchNorm1d(out_d),
                    nn.Tanh(),    
                ))
        
        self.layers = nn.Sequential(
            *self.layers
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


class Encoder(nn.Module):
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
        out = torch.cat((img_zs[-1].squeeze(-1), audio_z, noise_z.transpose(1,2)), dim=-1)
        return out


class FrameDecoder(nn.Module):
    def __init__(self, config):
        super().__init__() 
        
        padding = calculate_padding(config['kernel_size'], 
                                    config['stride'])
        self.layers = nn.ModuleList()
        
    def _make_layer(in_d, out_d, stride, kernel_size, pad):
            return nn.Sequential(
                nn.Conv2d(in_d, out_d,
                        stride=stride,
                        padding=pad,
                        kernel_size=kernel_size),
                nn.BatchNorm2d(out_d),
                nn.ReLU())
            
    def forward(self, latent, img_hids):
        for i in range(self.layers):
            latent = self.layers[i](latent) + img_hids[i]
        return latent  
    
    
import yaml

with open('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/avatar/realistic/configs/models.yaml', 'r') as f:
    config=  yaml.load(f.read(), Loader=yaml.FullLoader)
    
gen = Generator(config['generator'])

img = torch.randn((3, 3, 96,128))
audio_frames = torch.randn((3, 1, 3200))

img_zs = gen(img, audio_frames)

print(img_zs.size())