import os
import json

def search_for_oov(input_dir, output_file, lexicon_file):
    '''
    Searchs through all text files looking for oov words 
    writing them to an output file at the end
    '''
    oov = []
    with open(lexicon_file, 'r') as f:
        lexicon = f.readlines()
        lexicon = [word.split('\t')[0] for word in lexicon]
        
    for subfolder in os.listdir(input_dir):
        for file in os.listdir(f'{input_dir}/{subfolder}'):
            with open(f'{input_dir}/{subfolder}/{file}', 'r') as f:
                text = f.read().strip("\n").upper().split(" ")
            for word in text:
                if word in lexicon:
                    continue
                else:
                    oov.append(word) 
                
            with open(output_file, 'a') as f:
                f.writelines(oov)
            
                