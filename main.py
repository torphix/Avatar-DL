import os
import sys
import argparse
from natsort import natsorted
from tqdm import tqdm
from data.create import create_dataset
from data.filter import SeperateSpeakers
from stt.main import transcribe_subfolders
from data.datasets.lex_fridman.lex import get_dataset_length
from avatar.realistic.train import train as realistic_avatar_train

if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    # Dataset Commands
    if command == 'create_dataset':
        parser.add_argument('-d', '--dataset', required=True,
                            help='Name of dataset, eg: "lex", "crema" ') 
        args, leftover_args = parser.parse_known_args()
        create_dataset(args.dataset)
        
    elif command == 'seperate_speakers':
        parser.add_argument('-id', '--in_dir', required=True,
                            help='Embed and clusters the audio') 
        parser.add_argument('-od', '--out_dir', required=True,
                            help='Folder where seperated speakers should be placed') 
        parser.add_argument('-pt', '--p_threshold', default=0.6,
                            help='Only confidence scores above this value will be saved') 
        parser.add_argument('-bs', '--batch_size', default=8,
                            help='Processing batch size for embedding') 
        parser.add_argument('-mn', '--model_name', default='e13_classifer_14-04-22.pth',
                            help='Model name') 
        parser.add_argument('-d', '--device', default='cpu',
                            help='Model name') 
        args, leftover_args = parser.parse_known_args()
        processor = SeperateSpeakers(args.in_dir, 
                                     args.out_dir,
                                     args.p_threshold,
                                     args.batch_size,
                                     args.model_name,
                                     args.device)
        processor.seperate_speakers()
        
    
    elif command == 'clear_download_log':
        parser.add_argument('-d', '--dataset', required=True,
                            help='Name of dataset, eg: "lex", "crema" ') 
        args, leftover_args = parser.parse_known_args()
        with open(f'/{args.dataset}/download_log.txt', 'w') as f:
            f.write()
    
    elif command == 'train_realistic_avatar':
        realistic_avatar_train()
        
    # ASR commands
    elif command == 'transcribe_folder':
        parser.add_argument('-i', '--input_dir', required=True,
                            help='Input dir') 
        parser.add_argument('-o', '--output_dir', required=True,
                            help='Output dir will mimic inputs dirs file structure')
        parser.add_argument('-bs', '--batch_size', default=8)
        args, leftover_args = parser.parse_known_args()
        transcribe_subfolders(args.input_dir, args.output_dir)        
        
    elif command == 'format_audio_text_dirs_for_tts':
        parser.add_argument('-a', '--audio_dir', required=True,
                            help='Input dir') 
        parser.add_argument('-t', '--text_dir', required=True,
                            help='Output dir will mimic inputs dirs file structure')
        parser.add_argument('-o', '--output_file', required=True)
        args, leftover_args = parser.parse_known_args()
        audio_files = [f'{file}' for file in natsorted(os.listdir(args.audio_dir))]
        text_files = [f'{args.text_dir}/{file}' for file in natsorted(os.listdir(args.text_dir))]
        with open(args.output_file, 'w') as output_f:
            for i in tqdm(range(len(text_files))):
                text, audio = text_files[i], audio_files[i]
                with open(text, 'r') as f:
                    text = f.read().strip("\n")
                output_f.write(f'{audio.split(".")[0]}|{text}\n')
                
        
    elif command == 'get_dataset_length':
        parser.add_argument('-i', '--input_dir', required=True,
                            help='Input dir') 
        args, leftover_args = parser.parse_known_args()
        get_dataset_length(args.input_dir)
        
    elif command == 'correct_text_files':
        parser.add_argument('-i', '--input_dir', required=True,
                            help='Input dir') 
        parser.add_argument('-av', '--additional_vocab', default='',
                            help='Extra words to check') 
        args, leftover_args = parser.parse_known_args()
        correct_spelling(args.input_dir, args.additional_vocab)
    
    # TTS commands
    # elif command == 'generate_tts_dataset':
    #     parser.add_argument('-cp', '--config_path', required=True,
    #                         help='Creates a dataset of text audio pairs') 
    #     args, leftover_args = parser.parse_known_args()
    #     with open(args.config_path, 'r') as f:
    #         data = yaml.load(f.read(), Loader=yaml.FullLoader)
    #         get_data(data['links'],
    #                  data['output_path'],
    #                  data['min_silence_len'], 
    #                  data['seperate_speakers'])
                
                
    else:
        print('''
              Available Commands Are:
                - create_dataset 
                - train_realisitic_avatar
              ''') 