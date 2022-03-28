import os
import numpy as np
from tqdm import tqdm
from retinaface import RetinaFace
from moviepy.editor import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


'''
Crops face from images
VideoFlash
    - sample_set 1
    - sample_set 2
Aligned
    - sample_set 1
'''


def align_dataset(input_dirs):
    print('Processing images...')
    for folder in tqdm(os.listdir(input_dirs)):
        for video in tqdm(os.listdir(f'{input_dirs}/{folder}')):
            frames, fps = align_faces_retina(f'{input_dirs}/{folder}/{video}')
            new_clip = ImageSequenceClip(frames, fps=fps)
            new_clip.write_videofile(f"{input_dirs}/{folder}/{video}", fps=fps)
            

def align_faces_retina(input_file):
    clip = VideoFileClip(input_file)    
    frames = clip.iter_frames()
    frames = np.stack([frame for frame in frames])
    new_frames = []
    for i, frame in enumerate(frames):
        if i == 0:
            res = RetinaFace.detect_faces(frame)['face_1']
            face_bb = res['facial_area']
            x, y, w, h = face_bb[0] - 35, face_bb[1] - 50, face_bb[2] + 40, face_bb[3] + 30
        new_frames.append(frame[y: h, x: w])
    return new_frames, clip.fps

        
align_dataset('/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/avatar/realistic/data/datasets/processed/lex/VideoFlash')