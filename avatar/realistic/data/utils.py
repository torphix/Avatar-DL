import os
import cv2
import torch
import shutil
import math
import torch.nn.functional as F
from tqdm import tqdm
from scipy.io import wavfile


def get_dataset(dataset_name):
    if dataset_name == 'crema':
        root = 'avatar/realistic/data/datasets/'
        os.makedirs(f'{root}/raw', exist_ok=True)
        os.chdir(f'{root}/raw')
        os.system('git lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git ')
        os.makedirs(f'{root}/processed/CREMA-D', exist_ok=True)
        shutil.move(f'{root}/raw/CREMA-D/AudioWAV/', f'{root}/processed/CREMA-D/AudioWAV/')
        shutil.move(f'{root}/raw/CREMA-D/VideoFlash/', f'{root}/processed/CREMA-D/VideoFlash/')
        shutil.rmtree(f'{root}/raw/CREMA-D')
    elif dataset_name == 'insertHere':
        pass
    else:
        raise Exception('Error only datasets crema supported ATM')

# Audio
def cut_audio_sequence(seq, cutting_stride, pad_samples, audio_frame_feature_len):
    '''Splits audio into 1:1 frame audio clip mapping (generator input)'''
    pad_left = torch.zeros(pad_samples // 2, 1)
    pad_right = torch.zeros(pad_samples - pad_samples // 2, 1)

    seq = torch.cat((pad_left, seq), 0)
    seq = torch.cat((seq, pad_right), 0)

    stacked = seq.narrow(0, 0, audio_frame_feature_len).unsqueeze(0)
    iterations = (seq.size()[0] - audio_frame_feature_len) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(0, i * cutting_stride, audio_frame_feature_len).unsqueeze(0)))
    return stacked

def split_audio(audio, sample_rate, split_size):
    '''
    split_size in seconds: 
    divides audio into chunks {split_size} seconds long
    (discriminator input) 
    '''
    audio = audio.squeeze(1)
    chunk_length = sample_rate * split_size
    chunks = audio.shape[0] / chunk_length
    pad_to = (math.ceil(chunks) * chunk_length) - audio.shape[0]
    audio = F.pad(audio, (0, int(pad_to)), value=0)
    audio = audio.view(-1, int(chunk_length))
    return audio.unsqueeze(1)

def get_audio_max_n_frames(audio_dir, length, overlap):
    print('Getting max number of frames')
    frames = []
    for f in tqdm(os.listdir(audio_dir)):
        samplerate, data = wavfile.read(f'{audio_dir}/{f}')
        stacked = cut_audio_sequence(torch.tensor(data), length, overlap, samplerate)
        frames.append(stacked.shape[0])
    return max(frames)

def get_frame_rate(metadata):
    numerator, denominator  = metadata['video']['@avg_frame_rate'].split('/')
    return int(numerator / denominator)

def shuffle_audio(audio_blocks):
    shuffle_idx = torch.randperm(audio_blocks.size(1))
    audio_blocks = audio_blocks[:, shuffle_idx, :, :]
    return audio_blocks

# Video
import skvideo.io

def read_video(vid_path):
    vid_data = skvideo.io.vread(vid_path)
    return torch.tensor(vid_data).permute(0, 3, 1, 2)

def cut_video_sequence(video, frames_per_block):
    # Split video into list of blocks of 5 frames
    # Make video even length
    def _round(x, base=5):
        return base * round(x/base)
    
    crop_to = _round(video.shape[0], base=frames_per_block)
    if crop_to > video.shape[0]: crop_to -= frames_per_block
    video = video[:crop_to, :,:,:]
    video_blocks = torch.split(video, frames_per_block, dim=0)
    return torch.stack(video_blocks, dim=0)

def sample_frames(video, sample_size):
    selection_idx = torch.randperm(sample_size)
    real_frames = video[selection_idx, :, :, :]
    return real_frames

