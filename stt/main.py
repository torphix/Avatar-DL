import os
from .asr import ASR
from tqdm import tqdm
from .utils import SpellingCorrector


def transcribe_subfolders(input_dir, output_dir):
    for folder in os.listdir(input_dir):
        transcribe_folder(f'{input_dir}/{folder}', f'{output_dir}/{folder}')

def transcribe_folder(input_dir, output_dir):
    asr = ASR()
    for file in os.listdir(input_dir):
        asr.file_inference(f'{input_dir}/{file}', 
                           f'{output_dir}/{file.split(".")[0]}.txt')
    
def correct_spelling(input_dir, additional_vocab=''):
    corrector = SpellingCorrector(additional_vocab)
    for folder in tqdm(os.listdir(input_dir)):
        corrector.correct_folder(f'{input_dir}/{folder}')
    