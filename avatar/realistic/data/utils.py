import os
import cv2
import torch
import shutil
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


def cut_audio_sequence(seq, cutting_stride, pad_samples, audio_frame_feature_len):
    pad_left = torch.zeros(pad_samples // 2, 1)
    pad_right = torch.zeros(pad_samples - pad_samples // 2, 1)

    seq = torch.cat((pad_left, seq), 0)
    seq = torch.cat((seq, pad_right), 0)

    stacked = seq.narrow(0, 0, audio_frame_feature_len).unsqueeze(0)
    iterations = (seq.size()[0] - audio_frame_feature_len) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, 
                                seq.narrow(0, i * cutting_stride, audio_frame_feature_len).unsqueeze(0)))
    return stacked


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


def read_video(vid_path):
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    images = [torch.tensor(image)]
    while success:
        success, image = vidcap.read()
        if image is not None:
            images.append(torch.tensor(image))
    return torch.stack(images).permute(0, 3, 1, 2).type(torch.FloatTensor)