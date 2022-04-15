import os
import sys
import shutil
import argparse
from tqdm import tqdm
from natsort import natsorted

def process(audio_dir, text_dir, output_dir):
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
    

def merge_lexicons(primary_lexicon_path, secondary_lexicon_path):
    
    with open(primary_lexicon_path, 'r') as f:
        primary_lexicon = f.readlines()
    with open(secondary_lexicon_path, 'r') as f:
        secondary_lexicon = f.readlines()
    

    primary_lines = {line.split(' ')[0].upper(): " ".join(line.split(' ')[1:])
                        for line in primary_lexicon}
    secondary_lines = {line.split(' ')[0].upper(): " ".join(line.split(' ')[1:])
                        for line in secondary_lexicon}
    
    primary_keys = list(primary_lines.keys())
    for k,v in secondary_lines.items():
        if k not in primary_keys:
            primary_lines[k] = v
    
    with open(primary_lexicon_path, 'w') as f:
        for k,v in primary_lines.items():
            f.write(f'{k.upper().strip(" ")} {v}')
    print('Merge complete')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    command = sys.argv[1]    

    if command == 'process':
        parser.add_argument('-a', '--audio_dir', required=True,
                            help='Audio input path') 
        parser.add_argument('-t', '--text_dir', required=True,
                            help='Text input path') 
        parser.add_argument('-o', '--output_dir', required=True,
                            help='output path') 
        args, leftover_args = parser.parse_known_args()
        process(args.audio_dir, args.text_dir, args.output_dir)
    elif command == 'merge_lexicons':
        parser.add_argument('-pl', '--primary_lexicon', required=True) 
        parser.add_argument('-sl', '--secondary_lexicon', required=True) 
        args, leftover_args = parser.parse_known_args()
        merge_lexicons(args.primary_lexicon, args.secondary_lexicon)
    
    else:
        print('''
              Command not found please try:
                - process
                - merge_lexicons
              ''')
        