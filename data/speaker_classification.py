import os
import yaml
import json
import torch
import librosa
import torch.nn as nn
import pytorch_lightning as ptl
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset, random_split


class SpeechClassifier(nn.Module):
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
        file_id|text|speaker_id
        '''
        self.wav_dir = config['wav_dir']
        self.sample_rate = config['sample_rate']
        self.max_wav_len = config['max_wav_len']
        self.n_classes = config['n_classes']
        
        # Get speaker IDS
        with open(config['metadata_path'], 'r') as f:
            lines = f.readlines()
        self.speaker_ids = [line.split("|")[-1] for line in lines]
        self.file_ids = [line.split("|")[0] for line in lines]
        
        print(f'Number of speakers found: {len(set(self.speaker_ids))}')
        # Assign each speaker to a key
        self.speaker_keys = {speaker: i for i, speaker in enumerate(set(self.speaker_ids))}
        assert config['n_classes'] == len(self.speaker_keys.values()), \
            f"Differing desired number of classes {config['n_classes']} & actual {len(self.speaker_keys)}"
        with open(config['speaker_id_path'], 'w') as f:
            f.write(json.dumps(self.speaker_keys))
        
    def __len__(self):
        return len(os.listdir(self.wav_dir))
    
    def __getitem__(self, idx):
        wav, sr = librosa.load(self.file_ids[idx],
                               sr=self.sample_rate)
        wav = wav[:self.max_wav_len]
        speaker_id = torch.zeros((self.n_classes))
        speaker_id[self.speaker_ids[idx]] = 1
        return wav, speaker_id
        
    def collate_fn(self, batch):
        wavs, speaker_ids = [], []
        for datapoint in batch:
            wav, speaker_id = datapoint
            wav = pad(wav, (0, self.max_wav_len-wav.shape[0]), value=0)
            wavs.append(wav), speaker_ids.append(speaker_id)
        return torch.stack(wavs), torch.stack(speaker_ids)
    
    
    
class SpeechClassifierModule(ptl.LightningModule):
    def __init__(self, module_config, model_config, data_config):
        super().__init__() 
        with open(module_config, 'r') as f:
            self.module_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(model_config, 'r') as f:
            model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(data_config, 'r') as f:
            self.data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        self.model = SpeechClassifier(model_config)
        self.dataset = SpeakerClassificationDataset(self.data_config)
        self.train_ds, self.val_ds = random_split(self.dataset, 
                                                  self.data_config['dataset_split'])

    def train_dataloader(self):
        return DataLoader(
                self.train_ds, 
                **self.data_config['dataloader'])

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            **self.data_config['dataloader'])

    def forward(self, batch):
        outputs = self.model(batch)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        
        return {'loss': loss, 'logs':logs}

    def validation_step(self, batch, batch_idx):
        return {'loss': loss, 'logs':logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.module_config['learning_rate'])
        return {'optimizer':optimizer,
                'scheduler': ReduceLROnPlateau(optimizer)}