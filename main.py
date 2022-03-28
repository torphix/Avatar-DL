import sys
import shutil
import argparse
from avatar.realistic.train import train
from avatar.realistic.data.create import create_dataset

if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    if command == 'create_realistic_avatar_dataset':
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
        train()
        
    else:
        print('''
              Available Commands Are:
                - create_realistic_avatar_dataset 
                - train_realisitic_avatar
              ''') 