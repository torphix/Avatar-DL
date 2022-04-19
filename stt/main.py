import os
from .asr import ASR
from tqdm import tqdm
from .finetune import ASRFinetune


def transcribe_subfolders(input_dir, output_dir):
    asr = ASR()
    for folder in tqdm(os.listdir(input_dir)):
        for file in tqdm(os.listdir(f'{input_dir}/{folder}')):
            asr.file_inference(f'{input_dir}/{folder}/{file}', 
                               f'{output_dir}/{folder}/{file.split(".")[0]}.txt')

def transcribe_folder(input_dir, output_dir):
    asr = ASR()
    for file in tqdm(os.listdir(input_dir)):
        asr.file_inference(f'{input_dir}/{file}', 
                           f'{output_dir}/{file.split(".")[0]}.txt')
    

def finetune(input_dir, output_dir, device):
    asr = ASRFinetune(input_dir, output_dir, device)
    asr.finetune()