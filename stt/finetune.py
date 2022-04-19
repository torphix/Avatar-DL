import re
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from natsort import natsorted
from datasets import load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor)
from stt.data import ASRDataset, DataCollatorCTCWithPadding


class ASRFinetune:
    def __init__(self,
                 input_dir,
                 output_dir,
                 device=None,
                 model_dict=None):
        super().__init__()
        self.device = device
        self.output_dir = output_dir
        self.vocab = self.get_vocab(input_dir)
        self.tokenizer = Wav2Vec2CTCTokenizer(f"{input_dir}/vocab.json", 
                                                unk_token="[UNK]",
                                                pad_token="[PAD]",
                                                word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=16000,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor,
                                           tokenizer=self.tokenizer)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", 
                                                    ctc_loss_reduction="mean", 
                                                    pad_token_id=self.processor.tokenizer.pad_token_id)
        self.model.freeze_feature_extractor()
        self.dataset = ASRDataset(input_dir, self.processor)
        
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        if model_dict is not None:
            self.model.load_state_dict(torch.load(model_dict))
            
    def text_cleaner(self, text):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        return re.sub(chars_to_ignore_regex, '', text).lower()
    
    def get_vocab(self, input_dir):
        print('Building Vocabulary...')
        vocab = set()
        for file in tqdm(os.listdir(input_dir)):
            if '.txt' in file or '.lab' in file:
                with open(f'{input_dir}/{file}') as f:
                    text = f.read().strip("\n").lower()
                vocab = vocab | set([char for char in text])
            else: 
                continue
        print(vocab)
        vocab = {k:v for v, k in enumerate(vocab)}
        vocab["|"] = vocab[" "]
        del vocab[" "]
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)
        print(f'Found vocab: {vocab}')
        with open(f'{input_dir}/vocab.json', 'w') as vocab_f:
            json.dump(vocab, vocab_f)
        return vocab
    
    def finetune(self):
        '''
        Input dir: *.wav & *.lab audio, text files
        '''
        self.model.freeze_feature_extractor()
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            group_by_length=True,
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=30,
            fp16=True,
            gradient_checkpointing=True, 
            save_steps=500,
            eval_steps=500,
            logging_steps=500,
            learning_rate=1e-4,
            weight_decay=0.005,
            warmup_steps=1000,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            tokenizer=self.processor.feature_extractor)
                
        trainer.train()
        torch.save(self.model.state_dict(), f'{self.output_dir}/finetuned_model.pth')
        
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
    
