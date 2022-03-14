import torch
import torch.nn as nn

from utils import calculate_padding
import torch.nn.functional as F


class FrameDiscriminator(nn.Module):
    '''
    To get high quality frame image
    Expects generated / true image (256,256)
    Starting frame for identity (256,256)
    '''
    def __init__(self, config):
        super().__init__() 
        
        self.layers = nn.ModuleList()
        for i in range(len(config['feature_sizes'])-1):
            padding = calculate_padding(config['kernel_size'][i], 
                                        config['stride'])
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(6 if i == 0 else config['feature_sizes'][i],
                              config['feature_sizes'][i+1],
                              config['kernel_size'][i],
                              stride=config['stride'],
                              padding=padding),
                    nn.BatchNorm2d(config['feature_sizes'][i+1]),
                    nn.LeakyReLU(0.2, True)))
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(config['feature_sizes'][-1], 1, 12),
                nn.Sigmoid()))

    def forward(self, x, starting_frame):
        '''
        x: generated / ground truth image
        starting_frame: identity frame
        '''
        x = torch.cat((x, starting_frame), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x.view(-1)


class VideoDiscriminator(nn.Module):
    '''To get high quality fluid video generation'''
    def __init__(self, config, img_size):
        super().__init__() 
        
        self.max_n_frames = config['max_n_frames']
        self.layers = nn.ModuleList()
        self.prelayer = nn.Sequential(
                            nn.Conv3d(3, config['feature_sizes'][0],
                                     kernel_size=(config['max_n_frames'], 10, 10)),
                            nn.BatchNorm3d(config['feature_sizes'][0]),
                            nn.LeakyReLU(0.2, True))
        
        for i in range(len(config['feature_sizes'])-1):
            self.layers.append(nn.Sequential(
                                nn.Conv2d(config['feature_sizes'][i],
                                          config['feature_sizes'][i+1],
                                          config['kernel_sizes'][i],
                                          stride=config['stride']),
                                nn.BatchNorm2d(config['feature_sizes'][i+1]),
                                nn.LeakyReLU(0.2, True)))
        
        self.linear = nn.Sequential(
                            nn.Linear(512*26*26, 1),
                            nn.Sigmoid())

    def forward(self, x):
        '''
        x: frames (synthetic or real): [1, C, N Frames, H, W]
        '''
        _, C, N, H, W = x.size()
        x = F.pad(x, pad=(0, 0, 0, 0, int(self.max_n_frames-N), 0, 0, 0), value=0)
        x = self.prelayer(x)
        for layer in self.layers:
            x = layer(x.squeeze(2))
        x = torch.flatten(x)
        x = self.linear(x)  
        return x
  
  
class SyncDiscriminator(nn.Module):
    '''Syncronization of video and frame'''
    def __init__(self, config, img_size, audio_length):
        super().__init__() 
        # Video encoder
        self.video_encoder_layers = nn.ModuleList()
        self.video_encoder_layers.append(
            nn.Sequential(
                nn.Conv3d(3, config['video_feature_sizes'][0],
                          kernel_size=(5, 4, 4),
                          stride=(1,2,2)),
                nn.BatchNorm3d(config['video_feature_sizes'][0]),
                nn.LeakyReLU(0.2, inplace=True)))        
        


        for i in range(len(config['video_feature_sizes'])-1):
            self.video_encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(config['video_feature_sizes'][i],
                              config['video_feature_sizes'][i+1],
                              kernel_size=config['video_kernel_sizes'][i],
                              stride=config['video_stride']),
                    nn.BatchNorm2d(config['video_feature_sizes'][i+1]),
                    nn.LeakyReLU(0.2, inplace=True)))
            
        # Only bottom half of image is used
        height, width = img_size[0] // 2, img_size[1]    
        for i in range(len(config['video_feature_sizes'])):
            # output_size = [(W-K + 2P) / S] + 1
            height = int((height-config['video_kernel_sizes'][i] + 2*0) / config['video_stride']) + 1
            width = int((width-config['video_kernel_sizes'][i] + 2*0) / config['video_stride']) + 1
        linear_input = int(height*width*config['video_feature_sizes'][-1])
        self.video_linear = nn.Linear(linear_input, 256)
        
        # Audio encoder
        width = audio_length 
        self.audio_encoder_layers = nn.ModuleList()
        for i in range(len(config['audio_feature_sizes'])-1):
            # output_size = [(W-K + 2P) / S] + 1
            width = int((width-config['audio_kernel_sizes'][i] + 2*0) / config['audio_stride'][i]) + 1
            self.audio_encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(1 if i == 0 else config['audio_feature_sizes'][i],
                              config['audio_feature_sizes'][i+1],
                              kernel_size=config['audio_kernel_sizes'][i],
                              stride=config['audio_stride'][i]),
                    nn.BatchNorm1d(config['audio_feature_sizes'][i+1]),
                    nn.LeakyReLU(0.2, inplace=True)))

        linear_input = int(width*config['audio_feature_sizes'][-1])
        self.audio_linear = nn.Linear(linear_input, 256)
        
        self.discriminator = nn.Sequential(
                                    nn.Linear(256, 1),
                                    nn.Sigmoid())
        
    def forward(self, frames, audio):
        # Video
        for i, layer in enumerate(self.video_encoder_layers):
            frames = layer(frames)
            if i == 0: frames = frames.squeeze(2)
        frames = torch.flatten(frames)
        frame_emb = self.video_linear(frames)
        # Audio
        for i, layer in enumerate(self.audio_encoder_layers):
            audio = layer(audio)
        audio = torch.flatten(audio)
        audio_emb = self.audio_linear(audio)
        
        sim_score = (frame_emb - audio_emb)**2
        x = self.discriminator(sim_score)
        return sim_score, x
    
    
    
