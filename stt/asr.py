import os
from tqdm import tqdm 
from huggingsound import SpeechRecognitionModel


class ASRInference():
    def __init__(self, config):
        self.model = SpeechRecognitionModel(config['model_path_or_name'], config['device'])
        self.input_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.batch_size = config['batch_size']
        # TODO add language model support
        # TODO add levenshtein auto correct
        self.wav_paths = []
        self.inputs = []
        
    def get_files(self, path):
        '''
        Recursivly searchs dirs for wav files
        returns tuple for each file (root, path filename)
        '''
        print('Loading Files...')
        for entity in tqdm(os.listdir(path)):
            if entity.endswith('wav'):
                sub_path = path.replace(f'{self.input_dir}/', '').replace(entity, '')
                self.wav_paths.append((self.input_dir, sub_path, entity))
                self.inputs.append(os.path.join(self.input_dir, sub_path, entity))
            elif os.path.isdir(f'{path}/{entity}'):
                self.get_files(f'{path}/{entity}')
                
    def save_outputs(self, outputs, wav_paths):
        print('Saving files...')
        for i, wav_path in enumerate(tqdm(wav_paths)):
            path = os.path.join(self.output_dir, wav_path[1])
            os.makedirs(path, exist_ok=True)
            with open(f'{path}/{wav_path[2].split(".")[0]}.txt', 'w') as f:
                f.write(outputs[i])
        
    def transcribe(self):
        self.get_files(self.input_dir)
        print('Transcribing..')
        outputs = self.model.transcribe(self.inputs, self.batch_size)
        transcriptions = [output['transcription'] for output in outputs]
        self.save_outputs(transcriptions, self.wav_paths)        
    
        
