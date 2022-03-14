import sys
import argparse
from avatar.realistic.data import data

if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()
    
    if command == 'get_avatar_dataset':
        parser.add_argument('--dataset', required=True) 
        args, leftover_args = parser.parse_known_args()
        data.get_dataset(args.dataset)