from .utils import *
import torch.nn as nn
from .vocoder.vocoder import get_vocoder


class TTS(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        configs = get_configs()
        self.max_wav_value = configs[0]['preprocessing']['audio']['max_wav_value']
        self.model = get_model(model_path, configs, device)
        self.vocoder = get_vocoder(device)
    
    def forward(self, text):
        inputs = prepare_sample(text)
        mel = self.model(*inputs[2:], d_control=1.0)
        wavs = self.vocoder(mel).squeeze(1)
        wavs = (wavs.cpu().numpy() * self.max_wav_value).astype("int16")
        return wavs