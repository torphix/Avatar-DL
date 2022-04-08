import os
from .asr import ASR
from tqdm import tqdm

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)

def transcribe_folder(input_dir, output_dir, batch_size):
    print('Transcribing folder & subfolders...')
    asr = ASR()
    if len(list(folders_in(input_dir))) > 0:
        for folder in os.listdir(input_dir):
            asr.folder_inference(f'{input_dir}/{folder}', f'{output_dir}/{folder}', batch_size)
    else:
        asr.folder_inference(f'{input_dir}/{folder}', output_dir, batch_size)
    print('Done')
        
        
