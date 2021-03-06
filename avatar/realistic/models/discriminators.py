import torch
import torch.nn as nn

from .utils import calculate_padding
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
                nn.Conv2d(config['feature_sizes'][-1], 1, (8,7)),
                nn.Sigmoid()))
        # (10, 12)

    def forward(self, x, starting_frame):
        '''
        x: [BS, C, H, W] generated / ground truth image
        starting_frame: [1, C, H, W] identity frame
        '''
        starting_frame = starting_frame.expand(x.size(0), -1,-1,-1)
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
                            nn.Linear(15360, 1),
                            nn.Sigmoid())

    def forward(self, x):
        '''
        x: frames (synthetic or real): 
        [BS (2 real/fake), C, N Frames, H, W]
        '''
        _, N, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.pad(x, pad=(0, 0, 0, 0, int(self.max_n_frames-N), 0, 0, 0), value=0)
        x = self.prelayer(x)
        x = x.squeeze(2)
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x)
        x = self.linear(x)  
        return x
  
  
class SyncDiscriminator(nn.Module):
    '''Syncronization of video and frame'''
    def __init__(self, config, img_size):
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
        # TODO: Fix hardcoded input values
        self.video_linear = nn.Linear(6144, 256)
        
        # Audio encoder
        width = config['audio_length'] 
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
        # TODO: Fix hardcoded input values
        self.audio_linear = nn.Linear(1280, 256)
        
        self.discriminator = nn.Sequential(
                                    nn.Linear(256, 1),
                                    nn.Sigmoid())
        
        self.param = nn.Parameter(torch.empty(0))
        
    @property    
    def device(self):
        return self.param.device  
    
    def pad(self, frames, audio):
        pad_amount = abs(audio.shape[0] - frames.shape[0])
        if audio.shape[0] > frames.shape[0]:
            pad_frame = torch.zeros((pad_amount, frames.shape[1]), 
                                     device=self.device)
            frames = torch.cat((frames, pad_frame),dim=0)
        elif audio.shape[0] < frames.shape[0]:
            pad_frame = torch.zeros((pad_amount, audio.shape[1]), 
                                     device=self.device)
            audio = torch.cat((audio, pad_frame),dim=0)
        return frames, audio
        
    def forward(self, frames, audio):
        '''
        Frames: [Num_Chunks, ChunkLen(5), C, H//2, W]  
        Audio: [Chunks, L (0.2s) * Features, 1]
        Output: [Chunk, 1] Prediction for each frame
        '''
        # 3d conv input: [N, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        for i, layer in enumerate(self.video_encoder_layers):
            frames = layer(frames)
            if i == 0: frames = frames.squeeze(2)
        frames = frames.view(frames.size(0), -1)
        frame_emb = self.video_linear(frames)
        # Audio
        audio = audio.squeeze(0).float()
        
        for i, layer in enumerate(self.audio_encoder_layers):
            audio = layer(audio)
        audio = audio.view(audio.size(0), -1)
        audio_emb = self.audio_linear(audio)
        # Pad vid frames to match audio frames
        frame_emb, audio_emb = self.pad(frame_emb, audio_emb)
        # Compute score
        sim_score = (frame_emb - audio_emb)**2
        x = self.discriminator(sim_score)
        return x
    
    
    # Sync discriminator,