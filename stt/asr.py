import re
import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from random import shuffle
import scipy.signal as sps
import torch.optim as optim
from natsort import natsorted
import torch.nn.functional as F
from transformers import (Wav2Vec2ForCTC,
                          Wav2Vec2Processor,
                          Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor)

from stt.data import DataCollatorCTCWithPadding


class ASR:
    def __init__(self,
                 model_dict=None,
                 processor="facebook/wav2vec2-base-960h",
                 root_path="/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/stt",
                 device=None):
        super().__init__()
        
        self.root_path = root_path
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        if model_dict is not None:
            self.model.load_state_dict(torch.load(model_dict))
            
    def load_audio(self, path, target_sr):
        audio, current_sr = librosa.load(path)
        number_of_samples = round(audio.shape[0] * float(target_sr) / current_sr)
        audio_input = sps.resample(audio, number_of_samples)        
        return audio_input

    def build_batch(self, batch_list):
        max_len = max([v.shape[0] for v in batch_list])
        batch_list = [F.pad(v, (0, int(max_len-v.shape[0]))) for v in batch_list]
        return torch.stack(batch_list)
            
    def file_inference(self, input_path, output_path=None, sr=16000):     
        audio = self.load_audio(input_path, sr)   
        input_values = self.processor(audio, 
                                      sampling_rate=sr,
                                      return_tensors="pt").input_values
        input_values = input_values.to(self.device)
        # retrieve logits & take argmax
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        # transcribe
        transcription = self.processor.decode(predicted_ids[0])
        output_dir = "/".join(output_path.split("/")[:-1])
        os.makedirs(output_dir, exist_ok=True)
        if output_path is not None:
            with open(output_path, 'w+') as f:
                f.write(transcription)
        return transcription

    def folder_inference(self, input_dir, output_dir=None, batch_size=8):
        files = natsorted([f'{input_dir}/{file}' for file in os.listdir(input_dir)])
        inputs, outputs = [], []
        for i in tqdm(range(len(files))):
            inputs.append(self.load_audio(files[i], 16000))
            if i % batch_size == 0 or i == len(files)-1:
                inputs = self.processor(inputs, 
                                        sampling_rate=16000,
                                        return_tensors="pt",
                                        padding=True).input_values.to(self.device)
                output = self.model(inputs).logits
                predicted_ids = torch.argmax(output, dim=-1)
                transcriptions = [self.processor.decode(ids) for ids in predicted_ids]
                outputs.append(transcriptions)
                inputs = []    
                print(outputs)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            for i, t in enumerate(transcriptions):
                f_name = files[i].split("/")[-1].split(".")[0]
                with open(f'{output_dir}/{f_name}.txt', 'w+') as f:
                    f.write(f'{t}\n') 
        return outputs
    
    def text_cleaner(self, text):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        return re.sub(chars_to_ignore_regex, '', text).lower()
    
    def get_vocab(self, input_dir):
        print('Building Vocabulary...')
        vocab = set()
        for file in tqdm(os.listdir(input_dir)):
            if '.txt' in file or '.lab' in file:
                with open(f'{input_dir}/{file}') as f:
                    text = f.read().strip("\n")
                vocab = vocab | set([char for char in text])
            else: 
                continue
        vocab = {v:k for v, k in enumerate(vocab)}
        vocab["|"] = vocab[" "]
        del vocab[" "]
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)
        print(f'Found vocab: {vocab}')
        with open(f'{input_dir}/vocab.json', 'w') as vocab_f:
            json.dump(vocab, vocab_f)
        return vocab
    
    def finetune(self, 
                 input_dir,
                 batch_size,
                 target_sample_rate,
                 save_path,
                 epochs):
        '''
        Input dir: *.wav & *.lab audio, text files
        '''
        vocab = self.get_vocab(input_dir)
        tokenizer = Wav2Vec2CTCTokenizer(f"{input_dir}/vocab.json", 
                                         unk_token="[UNK]",
                                         pad_token="[PAD]",
                                         word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=16000,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer)
        collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        self.model =model = Wav2Vec2ForCTC.from_pretrained(
                                            "facebook/wav2vec2-base", 
                                            ctc_loss_reduction="mean", 
                                            pad_token_id=self.processor.tokenizer.pad_token_id)
        optimizer = optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9)
        
        # Load batch
        
        shuffle(data)
        inputs, targets = [], []
        self.model.freeze_feature_extractor()
        for e in tqdm(range(epochs)):
            for i, datapoint in enumerate(data):
                # Load & Resample audio
                audio = self.load_audio(f"{self.root_path}/{datapoint['file_path']}", 
                                        target_sample_rate)
                inputs.append(audio)
                targets.append(datapoint['words'])
                if i % batch_size == 0:
                    inputs = self.processor(inputs, 
                                            sampling_rate=target_sample_rate,
                                            return_tensors="pt",
                                            padding=True).input_values
                    inputs = inputs.to(self.device)
                    with self.processor.as_target_processor():
                        targets = self.processor(targets,
                                                 return_tensors='pt',
                                                 padding=True).input_ids
                        targets = targets.to(self.device)
                        print(targets)
                    # Train loop
                    optimizer.zero_grad()        
                    loss = self.model(inputs, labels=targets).loss
                    loss.backward()
                    optimizer.step()
                    targets, inputs = [], []
                    
            print(f'Epoch Loss is: {loss}')
        torch.save(self.model.state_dict(), save_path)
        
    def compute_metrics(self, pred):
        wer_metric = load_metric("wer")
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
