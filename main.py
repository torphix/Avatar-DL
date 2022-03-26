import sys
import argparse
from avatar.realistic.data import data
from avatar.realistic.train import train
import avatar.realistic.data.face_alignment.face_alignment as fa

if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    if command == 'get_avatar_dataset':
        parser.add_argument('--dataset', required=True) 
        args, leftover_args = parser.parse_known_args()
        data.get_dataset(args.dataset)
        
    elif command == 'train_realistic_avatar':
        train()
        
    elif command == 'align_faces':
        parser.add_argument("--input_dir", required=True,
                        help="path to root directories of input images")
        parser.add_argument("--output_dir", required=True,
                        help="path to destination directories of output images")
        args, leftover_args = parser.parse_known_args()
        fa.align_faces(args.input_dir, args.output_dir)