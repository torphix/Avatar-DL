import os
import sys
import yaml
import shutil
import argparse
import subprocess
from tqdm import tqdm
from natsort import natsorted
from data.create import create_dataset
from data.filter import SeperateSpeakers
from tts.main import preprocess as tts_preprocess
from stt.main import asr_finetune, asr_inference, find_oov
from data.datasets.lex_fridman.lex import get_dataset_length
from avatar.realistic.train import train as realistic_avatar_train


def update_config_with_args(config_path, args):
    with open(config_path, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    for key, value in vars(args).items():
        if value is None: continue
        try:
            config[key] = value 
        except:
            raise Exception(f'Arg value:{key} not found in config') 
    return config


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
    elif command == 'asr_transcribe':
        parser.add_argument('-d', '--device')
        parser.add_argument('-mpn', '--model_path_or_name',
                            help='Disk location of saved model or name of hugging face repo')
        parser.add_argument('-i', '--input_dir',
                            help='Path to dataset, will recusivly search folders for wav files')
        parser.add_argument('-o', '--output_dir',
                            help='Output path, mimics input path structure')
        parser.add_argument('-cp', '--config_path', default='stt/config/inference.yaml',
                            help='Output path, mimics input path structure')
        parser.add_argument('-lm', '--use_lm', default=True,
                            help='Use language model with transcription, (longer & more accurate)')
        args, leftover_args = parser.parse_known_args()  
        config_path = args.config_path 
        del args.config_path
        config = update_config_with_args(config_path, args)
        if args.use_lm == False: del config['lm_dir']
        asr_inference(config)
            
    elif command == 'asr_finetune':
        parser.add_argument('-o', '--output_dir',
                            help='Location where finetuned model is saved to')
        parser.add_argument('-mpn', '--model_path_or_name',
                            help='Disk location of saved model or name of hugging face repo')
        parser.add_argument('-cp', '--config_path', default='stt/config/finetune.yaml',
                            help='Output path, mimics input path structure')
        args, leftover_args = parser.parse_known_args()  
        config_path = args.config_path
        del args.config_path
        config = update_config_with_args(config_path, args)
        asr_finetune(config)
        
    elif command == 'find_oov':
        parser.add_argument('-i', '--input_dir', required=True,
                            help='Input directory')
        parser.add_argument('-o', '--output_file', required=True,
                            help="Where the OOVs are written to")
        parser.add_argument('-l', '--lexicon', required=True,
                            help='Path to lexicon, should be in format WORD \t PHONEMES')
        args, leftover_args = parser.parse_known_args()  
        find_oov(args)
        
    elif command == 'add_oovs_to_lexicon':
        parser.add_argument('-ovf', '--oov_file', required=True,
                            help="Where the OOVs are written to")
        parser.add_argument('-lexicon', '--lex_file', required=True,
                            help="Lexicon file")
        args, leftover_args = parser.parse_known_args()  
        correct_oovs(args)
    
    elif command == 'generate_lexicon':
        parser.add_argument('-i', '--input_dir', required=True,
                            help='Path to .wav & .lab files')
        parser.add_argument('-o', '--output_path', required=True,
                            help='Path to output lexicon')
        parser.add_argument('-lm', '--language_model', default='english_g2p')
        parser.add_argument('-ol', '--update_old_lexicon',
                            help='Path to old lexicon')
        args, leftover_args = parser.parse_known_args()
        print('Aligning Corpus, may take a while please be patient')
        subprocess.run(f'''
                       conda run -n aligner mfa g2p {args.language_model} {args.input_dir} {args.output_path}''',
                        shell=True, capture_output=True, text=True)

        # Clean lexicon
        with open(args.output_path, 'r') as f:
            lines = set(f.readlines())
        with open(args.output_path, 'w') as f:
            f.writelines(lines)
        print('Alignment complete please check output file')
        
    elif command == 'update_old_lexicon':
        parser.add_argument('-nl', '--new_lexicon')
        parser.add_argument('-ol', '--old_lexicon')
        parser.add_argument('-o', '--output_path')
        args, leftover_args = parser.parse_known_args()
        with open(args.old_lexicon, 'r') as f:
            old_lexicon = set(f.readlines())
        with open(args.new_lexicon, 'r') as f:
            new_lexicon = set(f.readlines())
        lexicon = old_lexicon | new_lexicon
        with open(args.output_path, 'w') as f:
            f.writelines(lexicon)
    
    # TTS commands

        
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