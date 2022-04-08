import os
import sys
import shutil
import argparse
from tqdm import tqdm
from natsort import natsorted

def process_lex(audio_dir, text_dir, output_dir):
    '''
    Create folder full of wavs
    Metadata.csv in LJSpeech format
    '''
    audio_folders = natsorted([f'{audio_dir}/{folder}' for folder in os.listdir(audio_dir)])
    text_folders = natsorted([f'{text_dir}/{folder}' for folder in os.listdir(text_dir)])
    audio_files = natsorted([
        f'{folder}/{file}'
        for folder in audio_folders
        for file in os.listdir(folder)
    ])
    text_files = natsorted([
        f'{folder}/{file}'
        for folder in text_folders
        for file in os.listdir(folder)
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/wavs', exist_ok=True)
    for i in tqdm(range(len(audio_files))):
        audio_f, text_f = audio_files[i], text_files[i]
        f_name = audio_f.split("/")[-1].split(".")[0]
        assert audio_f.split("/")[-1].split(".")[0] == text_f.split("/")[-1].split(".")[0], \
            'Audio and text file are not the same, check code'
        shutil.copyfile(audio_f, f'{output_dir}/wavs/{f_name}.wav')
        with open(text_f, 'r') as f:
            text = f.read().strip("\n")
        with open(f'{output_dir}/metadata.csv', 'a') as f:
            f.write(f'{f_name}|{text}|{text}\n')
    print('Processing complete')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--audio_dir', required=True,
                        help='Audio input path') 
    parser.add_argument('-t', '--text_dir', required=True,
                        help='Text input path') 
    parser.add_argument('-o', '--output_dir', required=True,
                        help='output path') 
    args, leftover_args = parser.parse_known_args()
    
    process_lex(args.audio_dir, args.text_dir, args.output_dir)
    