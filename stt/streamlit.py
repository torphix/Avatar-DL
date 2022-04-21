import re
import os
import sys
import argparse
import streamlit as st
from natsort import natsorted
from spellchecker import SpellChecker

'''Used to Fix incorrect ASR transcriptions'''

def parse_args(args):
    parser = argparse.ArgumentParser('ASR corrector')
    parser.add_argument('--oov_file', help='File containing OOVs formatted word\tfilename.txt')
    parser.add_argument('--text_dir', help='Dir with subdirs containing text files in subdirs')
    parser.add_argument('--audio_dir', help='Corresponding Audio directory')
    return parser.parse_args(args)

args = parse_args(sys.argv[1:])
OOV_FILE = args.oov_file
TEXT_DIR = args.text_dir
AUDIO_DIR = args.audio_dir
if 'index' not in st.session_state:
    st.session_state['index'] = 0
    
# Edit Methods
def update_txt_file():
    with open(f'{TEXT_DIR}/{folder_name}/{f_name}.txt', 'w') as f:
        f.write(st.session_state.text_input)
    st.session_state['index'] += 1
        
def post_auto_correct():
    with open(f'{TEXT_DIR}/{folder_name}/{f_name}.txt', 'w') as f:
        f.write(auto_correct_text)
    st.session_state['index'] += 1

def load_next():
    update_txt_file()
def load_before():
    st.session_state['index'] -= 1
    
def skip_to_func():
    print(st.session_state.skip_to)
    st.session_state['index'] = int(st.session_state.skip_to)
    
def delete():
    os.remove(f"{AUDIO_DIR}/{folder_name}/{f_name}.wav")
    os.remove(f"{TEXT_DIR}/{folder_name}/{f_name}.txt")


text_files = natsorted([file for file in os.listdir(TEXT_DIR)])
wav_files = natsorted([file for file in os.listdir(AUDIO_DIR)])

with open(OOV_FILE, 'r') as f:
    oovs =[line.strip("\n") for line in f.readlines()]

f_names = natsorted(list(set([line.split("\t")[1].split(".")[0] for line in oovs])))
f_name = f_names[st.session_state['index']]
folder_name = "_".join(f_name.split("_")[1:])

with open(f'{TEXT_DIR}/{folder_name}/{f_name}.txt', 'r') as f:
    TEXT = f.read()
    
st.text(TEXT)
st.audio(f'{AUDIO_DIR}/{folder_name}/{f_name}.wav')

# UI
st.text_input('New text', value=TEXT, key='text_input')
st.button(label='Post Edits', key='edit',on_click=update_txt_file)
# Auto correct
spellchecker = SpellChecker()
auto_correct_text = " ".join([spellchecker.correction(word) for word in TEXT.split(" ")])
st.text(auto_correct_text)
# st.button(label='Post Auto Correct', key='auto_correct',on_click=post_auto_correct)
st.button(label='Next', key='next', on_click=load_next)
st.button(label='Back', key='back', on_click=load_before)
st.text_input('Skip to', value=st.session_state['index'], key='skip_to')
st.button(label='Skip to', on_click=skip_to_func)
st.button(label='DELETE', key='delete', on_click=delete)
st.text(f_name)
st.text(folder_name)
st.text(f'{st.session_state["index"]}/{len(f_names)}')


# Why is file name different upon clicking post button