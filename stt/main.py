import os
from .asr import ASR
from tqdm import tqdm


def transcribe_subfolders(input_dir, output_dir):
    for folder in tqdm(os.listdir(input_dir)):
        transcribe_folder(f'{input_dir}/{folder}', f'{output_dir}/{folder}')

def transcribe_folder(input_dir, output_dir):
    asr = ASR()
    for file in tqdm(os.listdir(input_dir)):
        asr.file_inference(f'{input_dir}/{file}', 
                           f'{output_dir}/{file.split(".")[0]}.txt')
    

    