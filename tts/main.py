import sys
import argparse
from inference import tts_cmd_line
from modules.ptl_module import train


if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()
    
    if command == 'train':
        parser.add_argument('-ckpt', '--load_ckpt', 
                            required=False,
                            help='Path to checkpoint') 
        args, leftover_args = parser.parse_known_args()
        train(args)
    
    elif command == 'synthesize':
        parser.add_argument('-t', '--text', 
                            required=True,
                            help='Text you want converted to speech') 
        parser.add_argument('-c', '--ckpt', 
                            required=True,
                            help='Path to model') 
        args, leftover_args = parser.parse_known_args()
        tts_cmd_line(args)