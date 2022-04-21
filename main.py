import os
import sys
import yaml
import argparse
from tqdm import tqdm
from natsort import natsorted
from data.create import create_dataset
from data.filter import SeperateSpeakers
from data.datasets.lex_fridman.lex import get_dataset_length
from avatar.realistic.train import train as realistic_avatar_train
from stt.main import asr_finetune, asr_inference, correct_oovs, find_oov


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
    elif command == 'asr_inference':
        parser.add_argument('-d', '--device')
        parser.add_argument('-mpn', '--model_path_or_name',
                            help='Disk location of saved model or name of hugging face repo')
        parser.add_argument('-i', '--input_dir',
                            help='Path to dataset, will recusivly search folders for wav files')
        parser.add_argument('-o', '--output_dir',
                            help='Output path, mimics input path structure')
        parser.add_argument('-cp', '--config_path', default='stt/config/inference.yaml',
                            help='Output path, mimics input path structure')
        args, leftover_args = parser.parse_known_args()  
        config_path = args.config_path
        del args.config_path
        config = update_config_with_args(config_path, args)
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
        
    elif command == 'correct_oov':
        parser.add_argument('-f', '--oov_file', required=True,
                            help="Where the OOVs are written to")
        args, leftover_args = parser.parse_known_args()  
        correct_oovs(args)
        
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