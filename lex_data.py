from turtle import width
import torch
import numpy as np
from skvideo.io import vread
from torchvision.transforms.functional import crop
from torchvision.transforms import Lambda, Compose, ToTensor
from torchvision.utils import save_image


video = vread('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/avatar/realistic/data/datasets/processed/lex/VideoFlash/output.mp4')
video = torch.tensor(video) / 255
N, H, W, C  = video.size()
width_crop = W // 4
video = video[:, :H//2, width_crop:-width_crop, :] 
video = video.permute(0, 3, 1, 2)
print(video.size())
save_image(video[0], 'cropped.png')



# 1. Crop image sides so its just the face and suit (as too big to fit in model atm) (maybe resize proportianlly?)
# 2. Align face 
# 3. Train on aligned faces
# 4. Add suit to aligned faces