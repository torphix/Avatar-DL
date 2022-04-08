import os
import streamlit as st
from natsort import natsorted

INPUT_DIR = "/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/tts/PortaSpeech/raw_data/output_path/lex/LJSpeech"
if 'index' not in st.session_state:
    st.session_state['index'] = 0
TEXT = ''

def update_txt_file():
    with open(f'{INPUT_DIR}/{text_files[st.session_state["index"]]}', 'w') as f:
        f.write(st.session_state.text_input)
    st.session_state['index'] += 1
        
def load_next():
    update_txt_file()
def load_before():
    st.session_state['index'] -= 1
    
def skip_to_func():
    print(st.session_state.skip_to)
    st.session_state['index'] = int(st.session_state.skip_to)
    
def delete():
    os.remove(f"{INPUT_DIR}/{audio_files[st.session_state['index']]}")
    os.remove(f"{INPUT_DIR}/{text_files[st.session_state['index']]}")

text_files, audio_files = [], []
for file in os.listdir(INPUT_DIR):
    if file.endswith('.lab'):
        text_files.append(file)
    elif file.endswith('.wav'):
        audio_files.append(file)
        
text_files = natsorted(text_files)    
audio_files = natsorted(audio_files)

with open(f'{INPUT_DIR}/{text_files[st.session_state["index"]]}', 'r') as f:
    TEXT = f.read()
st.text(TEXT)

audio_f = audio_files[st.session_state['index']]
st.audio(f'{INPUT_DIR}/{audio_f}')

st.text_input('New text', value=TEXT, key='text_input')
st.button(label='Post Edits', key='edit',on_click=update_txt_file)
st.button(label='Next', key='next', on_click=load_next)
st.button(label='Back', key='back', on_click=load_before)
st.text_input('Skip to', value=st.session_state['index'], key='skip_to')
st.button(label='Skip to', on_click=skip_to_func)
st.text(audio_f)
st.button(label='DELETE', key='delete', on_click=delete)