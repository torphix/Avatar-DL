import os
import json
import torch
import librosa
import torch.nn as nn
import pytorch_lightning as ptl
from torch.utils.data import DataLoader, Dataset

class SpeakerClassification(nn.Module):
    def __init__(self, config):
        super().__init__() 
        
        self.max_wav_len = config['max_wav_len']
        self.layers = nn.ModuleList()
        for i in range(len(config['k_sizes'])-1):
            self.layers.append(
                nn.Conv1d(config['hid_channels'][i],
                          config['hid_channels'][i+1],
                          config['k_size'][i]),
                nn.ReLU(),
                nn.BatchNorm1d(config['hid_channels'][i+1]))
        
        self.layers.append(
            nn.Linear(256, config['n_classes']),
            nn.Softmax())
        self.layers = nn.Sequential(**self.layers)        
        
    def forward(self, wavs):
        wavs = wavs[:, :self.max_wav_len]
        x = self.layers(wavs)
        return x
    
    
class SpeakerClassificationDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        '''
        Expects path to wav folder 
        & Text doc with wav ID + speaker ID
        text|speaker_id
        '''
        self.wav_paths = [f'{config["wav_dir"]}/{wav}'
                          for wav in os.listdir(config['wav_dir'])]
        self.sample_rate = config['sample_rate']
        self.max_wav_len = config['max_wav_len']
        
        # Get speaker IDS
        with open(config['metadata_path'], 'r') as f:
            lines = f.readlines()
        speaker_ids = [line.split("|")[1] for line in lines]

        print(f'Number of speakers found: {len(set(speaker_ids))}')
        # Assign each speaker to a key
        self.speaker_keys = {speaker: i for i, speaker in enumerate(set(speaker_ids))}
        assert config['n_classes'] == len(self.speaker_keys.values()), \
            f"Differing desired number of classes {config['n_classes']} & actual {len(self.speaker_keys)}"
        with open(config['speaker_id_path'], 'w') as f:
            f.write(json.dumps(self.speaker_keys))
        
    def __len__(self):
        return len(self.wav_paths)
    
    def __getitem__(self, index):
        wav, sr = librosa.load(self.wav_paths[index],
                               sr=self.sample_rate)
        wav = wav[:self.max_wav_len]
        # Compute speaker  ids
        return 
        