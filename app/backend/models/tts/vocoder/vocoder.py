import json
import torch
from models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_vocoder(device):
    with open("config.json", "r") as f:
            config = json.load(f)
    config = AttrDict(config)
    vocoder = Generator(config)
    ckpt = torch.load("generator_universal.pth.tar", map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder

