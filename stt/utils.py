import os
import json
from tqdm import tqdm
from spellchecker import SpellChecker


def search_for_oov(input_dir, output_file, lexicon_file):
    '''
    Searchs through all text files looking for oov words 
    writing them to an output file at the end
    '''
    oov = set()
    with open(lexicon_file, 'r') as f:
        lexicon = f.readlines()
        lexicon = [line.split(None, maxsplit=1)[0].strip(" ") for line in lexicon]
    for subfolder in tqdm(os.listdir(input_dir)):
        for file in tqdm(os.listdir(f'{input_dir}/{subfolder}')):
            with open(f'{input_dir}/{subfolder}/{file}', 'r') as f:
                text = f.read().split(" ")
            for word in text:
                word = word.strip("\n").strip(" ").upper()
                if word in lexicon:
                    continue
                else:
                    oov.add(f'{word}\t{file}\n') 
                
            with open(output_file, 'w') as f:
                f.writelines(set(oov))
    with open(output_file, 'r') as f:
        total_oov = len(f.readlines())
    print(f'OOV search complete. Found: {total_oov}')
                
                
def auto_correct_oov(oov_file):
    checker = SpellChecker()
    with open(oov_file, 'r') as f:
        oovs = f.readlines()
    for i, oov in enumerate(tqdm(oovs)):
        word = oov.split("\t")[0]
        correction = checker.correction(word)
        correct = input(f"Correct? {word} -> {correction} [y/n]:")
        if correct == 'y': 
            oovs[i] = '{}\t{}\n'.format(oovs[i].strip("\n"), correct.upper())
        elif correct == 'n':
            update = input(f"Enter custom correction (leave blank to maintain current):")
            if update == '':
                continue           
            else:            
                oovs[i] = '{}\t{}\n'.format(oovs[i].strip("\n"), update.upper())
    with open(oov_file, 'w') as f:
        f.writelines(oovs)
        