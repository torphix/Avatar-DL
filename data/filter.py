import os
import json
import shutil
import numpy as np
from tqdm import tqdm
# from data.AudioClassifier.classifier import batch_inference

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class SeperateSpeakers():
    def __init__(self,
                 in_dir,
                 out_dir,
                 p_threshold,
                 batch_size=8,
                 model_name='e13_classifer_14-04-22.pth',
                 device='cpu'):
        '''
        param: p_threshold: Any probabilities higher than this will be added to folder
        '''
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.p_threshold = float(p_threshold)
        self.batch_size = int(batch_size)
        self.model_name = model_name
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")

    def speaker_id(self):
        import librosa
        import torch
        from natsort import natsorted
        for file in natsorted(os.listdir(self.in_dir)):
            speech, _ = librosa.load(f'{self.in_dir}/{file}', sr=16000, mono=True)
            inputs = self.feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt")
            outputs = self.model(**inputs).logits
            outputs = torch.argmax(outputs, dim=-1)
            print(file,outputs)


    def seperate_speakers(self):    
        os.makedirs(f'{self.out_dir}/lex', exist_ok=True)
        os.makedirs(f'{self.out_dir}/others', exist_ok=True)
        os.makedirs(f'{self.out_dir}/confused', exist_ok=True)
        for subfolder in os.listdir(self.in_dir):
            for output in batch_inference(f'{self.in_dir}/{subfolder}', 
                                        self.batch_size,
                                        model_name=self.model_name):
                wav_paths, outputs = output
                for idx, output in enumerate(outputs):
                    if output[0] > self.p_threshold:
                        wav_path = wav_paths[idx].split("/")[-1]
                        shutil.copyfile(f'{self.in_dir}/{subfolder}/{wav_path}',
                                        f'{self.out_dir}/lex/{wav_path}')
                    elif output[1] > self.p_threshold:
                        wav_path = wav_paths[idx].split("/")[-1]     
                        shutil.copyfile(f'{self.in_dir}/{subfolder}/{wav_path}',
                                        f'{self.out_dir}/others/{wav_path}')
                    else:
                        wav_path = wav_paths[idx].split("/")[-1]     
                        shutil.copyfile(f'{self.in_dir}/{subfolder}/{wav_path}', 
                                        f'{self.out_dir}/confused/{output[0]}-{output[1]}-{wav_path}')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

