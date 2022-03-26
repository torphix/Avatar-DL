import os
from re import S
import torch
import random
from scipy.io import wavfile
import skvideo.io
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from . import face_alignment_mean_points
from .utils import (cut_audio_sequence, 
                    read_video,
                    resample_audio,
                    split_audio,
                    cut_video_sequence,
                    sample_frames)


class GANDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        root = 'avatar/realistic/data/datasets'
        assert config["name"] in ['lex', 'crema'], \
            "Dataset names 'lex' or 'crema' currently supported"
        self.audio_path = os.path.abspath(f'{root}/processed/{config["name"]}/AudioWAV')
        self.video_path = os.path.abspath(f'{root}/processed/{config["name"]}/VideoFlash')
        
        # Video settings
        self.fps = config['fps']
        self.img_size = config['img_size']
        self.img_frames_per_audio_clip = config['img_frames_per_audio_clip'] # 5 (0.2s)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1])),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        # Dataset settings
        self.max_n_video_frames = config['max_n_video_frames']
        self.real_frame_sample_size = config['real_frame_sample_size'] 
        # Audio
        self.audio_frame_size = config['audio_frame_size']
        self.audio_sample_rate = config['audio_sample_rate']
        # Files
        self.videos = [f'{self.video_path}/{vid}'
                       for vid in os.listdir(self.video_path)]
        # Build wav file from video filename
        self.wavs = [f"{self.audio_path}/{vid.split('.')[0]}.wav"
                     for vid in os.listdir(self.video_path)]

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        '''
        Audio clip : video frames -> 1:5 ratio , 0.2s : 5x 
        '''
        assert (self.videos[idx].split(".")[0].split("/")[-1] ==
                self.wavs[idx].split(".")[0].split("/")[-1]), \
            f'''Video {self.videos[idx]} 
            and audio file {self.wavs[idx]} are not the same!'''
        # Cut video
        sample_rate, audio = wavfile.read(f'{self.wavs[idx]}')
        video = read_video(self.videos[idx])
        video = torch.stack([align_image(img, 
                                         self.img_size,
                                         face_alignment_mean_points)
                             for img in video])
        save_image(video, 'real.png')
        video = self.img_transform(video)
        video_blocks = cut_video_sequence(video, self.img_frames_per_audio_clip)
        video_frame_subset = sample_frames(video, self.real_frame_sample_size)
        # Audio Processing
        audio = torch.tensor(audio)
        # For stero audio ie: 2 channels
        if audio.shape[-1] == 2:
            audio = torch.sum(audio, dim=-1) / 2
        if self.audio_sample_rate != sample_rate:
            audio = resample_audio(audio, sample_rate, self.audio_sample_rate)
        # Normalize audio
        audio = audio / torch.max(audio)
        cutting_stride = int(self.audio_sample_rate / self.fps)
        audio_frame_feat_len = int(self.audio_sample_rate * self.audio_frame_size)
        audio_padding = audio_frame_feat_len - cutting_stride
        # Cut audio 1 clip per frame
        audio_generator_input = cut_audio_sequence(
            audio.view(-1, 1),
            cutting_stride,
            audio_padding,
            audio_frame_feat_len)
        # Split audio into 0.2s chunks for sync_discriminator
        audio_chunks = split_audio(
            audio.view(-1, 1),
            self.audio_sample_rate, 
            self.audio_frame_size) 
        datapoint = {
            # Discriminator inputs
            'real_video_all': video,
            'real_frames_subset': video_frame_subset,
            'real_video_blocks': video_blocks,
            'audio_chunks': audio_chunks,
            # Generator outputs
            'fake_video_all': [],
            'fake_frames_subset': [],
            # Generator inputs
            'identity_frame': video[0],
            'audio_generator_input': audio_generator_input,
        }
        return datapoint


def test_sample_dataset():
    import yaml
    with open('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/avatar/realistic/configs/data.yaml', 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    dataset = GANDataset(data_config)
    datapoint = dataset.__getitem__(100)
    print(datapoint['real_frames_all'].size())
    print(datapoint['real_frames_subset'].size())
    print(datapoint['real_video_blocks'].size())
    print(datapoint['first_frame'].size())
    print(datapoint['audio_generator_input'].size())
    print(datapoint['audio_chunks'].size())
    
def remove_large_datapoint(min_v, max_v, video_path, audio_path):
    '''Expects min_v & max_v to be in frames'''
    audios = [a for a in os.listdir(audio_path)]
    for i, f in enumerate(tqdm(os.listdir(video_path))):
        video = read_video(f'{video_path}/{f}')
        if video.shape[0] > max_v:
            os.remove(f'{video_path}/{f}')
            os.remove(f'{audio_path}/{audios[i]}')
        elif video.shape[0] < min_v:
            os.remove(f'{video_path}/{f}')
            os.remove(f'{audio_path}/{audios[i]}')
    
def ensure_equal(video_path, audio_path):
    audios = [a.split('.')[0] for a in os.listdir(audio_path)]
    videos = [v.split('.')[0] for v in os.listdir(video_path)]
    excess_audios = set(audios) - set(videos)
    excess_videos = set(videos) - set(audios)
    print(excess_audios)
    print(excess_videos)
    [os.remove(f'{audio_path}/{f}.wav') for f in excess_audios]        
    [os.remove(f'{video_path}/{f}.flv') for f in excess_videos]        
    assert len([a for a in os.listdir(audio_path)]) == len([a for a in os.listdir(video_path)]), \
        'Unequal lengths error in code!!!'
        
    
def rm_uneven(video_path, audio_path, sr, fps, frame_size):
    audios = [a.split('.')[0] for a in os.listdir(audio_path)]
    videos = [v.split('.')[0] for v in os.listdir(video_path)]
    for i, video in enumerate(tqdm(videos)):
        sr, audio = wavfile.read(f'{audio_path}/{audios[i]}.wav')
        video = read_video(f'{video_path}/{video}.flv')
        cutting_stride = int(sr / fps)
        audio_frame_feat_len = int(sr * frame_size)
        audio_padding = audio_frame_feat_len - cutting_stride
        audio_generator_input = cut_audio_sequence(
            torch.tensor(audio).unsqueeze(1),
            cutting_stride,
            audio_padding,
            audio_frame_feat_len)
        print(video.size())
        print(audio_generator_input.size())
        
    
        
def get_max_min_frames_per_video():
    vs = []
    root = '/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/avatar/realistic/data/datasets/processed/VideoFlash'
    for vid in tqdm(os.listdir(f'{root}')):
        video = read_video(f'{root}/{vid}')
        vs.append(video.shape[0])
    return min(vs), max(vs), vs
    
