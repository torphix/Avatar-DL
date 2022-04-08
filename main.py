import sys
import argparse
from data.create import create_dataset
from stt.main import transcribe_folder
from data.datasets.lex_fridman.lex import get_dataset_length
from avatar.realistic.train import train as realistic_avatar_train

if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    # Realisitc Avatar Commands
    if command == 'create_dataset':
        parser.add_argument('-d', '--dataset', required=True,
                            help='Name of dataset, eg: "lex", "crema" ') 
        args, leftover_args = parser.parse_known_args()
        create_dataset(args.dataset)
        
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
        transcribe_folder(args.input_dir, args.output_dir, int(args.batch_size))        
        
    elif command == 'get_dataset_length':
        parser.add_argument('-i', '--input_dir', required=True,
                            help='Input dir') 
        args, leftover_args = parser.parse_known_args()
        get_dataset_length(args.input_dir)
        
    else:
        print('''
              Available Commands Are:
                - create_dataset 
                - train_realisitic_avatar
              ''') 