import os
import yaml
import torch.nn as nn
from scipy.io import wavfile
import pytorch_lightning as ptl
from librosa.feature import melspectrogram
from torch.utils.data import Dataset, DataLoader

class AudioIDClassifier(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.layers  = nn.ModuleList()
        for i in range(len(config['kernels'])):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(config['in_channels'][i],
                              config['out_channels'][i],
                              kernel_size=config['kernels'][i]),
                    nn.BatchNorm1d(config['out_channels'][i]),
                    nn.ReLU(),
                ))
        self.layers.append(
            nn.Sequential(
                nn.Linear(256, 2),
                nn.Sigmoid(),
            ))

    def forward(self, mel):
        x = mel
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class AudioIDDataset(Dataset):
    def __init__(self, text_file, wav_path):
        '''
        text_file: one sample per line 
            - format: text | speaker id
        '''
        with open(text_file, 'r') as f:
            data = f.readlines()
        with open('tts/data/config/data_config.yaml', 'r') as f:
            self.data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            
        self.text = [line.strip('\n') for line in data]
        self.wav_root = wav_path
        self.wavs = [wav for wav in os.listdir(wav_path)][:len(self.text)]
        
    def collate_fn(self, batch):
        for samples in batch:
            pass
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        datapoint = self.data[index]
        text = datapoint.split('|')[0]
        speaker = datapoint.split('|')[1]
        wav = self.wavs[index]
        assert wav.split('_')[0] != index, \
            f'Incorrect wav index {wav.split("_")[0]}, text'
        sr, wav = wavfile.read(f'{self.wav_root}/{wav}')
        mel = melspectrogram(wav, sr, 
                             self.data_config['n_fft'],
                             self.data_config['hop_length'],
                             self.data_config['win_length'])
        
        return text.strip(" "), speaker.strip(" "), mel
    
    
class AudioIDClassifierModule(ptl.LightningModule):
    def __init__(self):
        super().__init__() 
        with open('tts/data/config/id_model.yaml', 'r') as f:
            model_config = yaml.load(f.load(), Loader=yaml.FullLoader)
            
        self.model = AudioIDClassifier(model_config)

    def train_dataloader(self):
        dataset = AudioIDDataset('tts/data/dataset/raw/text/jocko.txt',
                                 'tts/data/dataset/raw/audio/')
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        return dataloader

    def forward(self,):
        return

    def training_step(self, batch, batch_idx):
        return {'loss': loss, 'logs':logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return {'optimizer':optimizer,
                'scheduler': ReduceLROnPlateau(optimizer)}