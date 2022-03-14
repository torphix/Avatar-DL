import os
import torch
import random
from scipy.io import wavfile
import skvideo.io
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import numpy as np
from utils import cut_audio_sequence, read_video
import cv2


class GANDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        root = 'avatar/realistic/data/datasets'
        if config['name'] == 'crema': 
            self.audio_path = os.path.abspath(f'{root}/processed/AudioWAV')
            self.video_path = os.path.abspath(f'{root}/processed/VideoFlash')
        
        # Video settings
        self.fps = config['fps']
        self.img_size = config['img_size']
        self.img_frames_per_audio_clip = config['img_frames_per_audio_clip'] # 5
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1])),
            # transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # Dataset settings
        self.max_n_video_frames = config['max_n_video_frames']
        self.real_frame_sample_size = config['real_frame_sample_size'] 
        # Audio
        self.audio_frame_feature_len = config['audio_frame_feature_len']
        self.audio_frame_overlap = config['audio_frame_overlap']
        # Files
        self.wavs = [f'{self.audio_path}/{wav}' 
                     for wav in os.listdir(self.audio_path)]
        self.videos = [f'{self.video_path}/{vid}'
                       for vid in os.listdir(self.video_path)]
            
    def sample_frames(self, video):
        sample_idxs = random.sample(range(0, video.shape[0]), self.real_frame_sample_size)
        real_frames = video[np.array(sample_idxs), :, :, :]
        return real_frames
    
    def cut_video_sequence(self, video):
        # Split video into list of blocks of 5 frames
        # Make video even length
        def _round(x, base=5):
            return base * round(x/base)
        
        crop_to = _round(video.shape[0], base=self.img_frames_per_audio_clip)
        if crop_to > video.shape[0]: crop_to -= self.img_frames_per_audio_clip
        video = video[:crop_to, :,:,:]
        video_blocks = torch.split(video, self.img_frames_per_audio_clip, dim=0)
        return torch.stack(video_blocks, dim=0)
    
    def __len__(self):
        return len(os.listdir(self.audio_path))
    
    def __getitem__(self, idx):
        '''
        Audio clip : video frames -> 1:5 ratio , 0.2s : 5x 
        '''
        # Cut video
        sample_rate, audio = wavfile.read(f'{self.wavs[idx]}')
        video = read_video(self.videos[idx])
        video = self.img_transform(video)
        video_blocks = self.cut_video_sequence(video)
        # Cut audio 1 clip per frame
        cutting_stride = int(sample_rate / self.fps)
        audio_padding = self.audio_frame_feature_len - cutting_stride
        audio_frames = cut_audio_sequence(
            torch.tensor(audio).view(-1, 1),
            cutting_stride,
            audio_padding,
            self.audio_frame_feature_len)
        
        datapoint = {
            # Discriminator inputs
            'real_frames_all': video,
            'real_frames_subset': self.sample_frames(video),
            'real_video_blocks': video_blocks,
            # Generator outputs
            'fake_video_insync': [],
            'fake_video_all': [],
            'fake_video_subset': [],
            'first_frame': video[0],
            'sliced_audio': audio_frames
        }
        return datapoint



def test_sample_dataset():
    import yaml
    with open('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/avatar/realistic/configs/data.yaml', 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    dataset = GANDataset(data_config)
    datapoint = dataset.__getitem__(1)
    print(datapoint['real_frames_all'].size())
    print(datapoint['real_frames_subset'].size())
    print(datapoint['real_video_blocks'].size())
    print(datapoint['first_frame'].size())
    print(datapoint['sliced_audio'].size())

