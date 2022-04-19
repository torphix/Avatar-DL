import os
import torch
import librosa
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from transformers import Wav2Vec2Processor
from typing import Any, Dict, List, Optional, Union
    
class ASRDataset(Dataset):
    def __init__(self, input_dir, processor):
        super().__init__()
        self.processor = processor
        self.texts, self.wavs = [], []
        for file in os.listdir(input_dir):
            if file.endswith('lab') or file.endswith('txt'):
                self.texts.append(f'{input_dir}/{file}')
            elif file.endswith('wav') or file.endswith('mp3'):
                self.wavs.append(f'{input_dir}/{file}')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):   
        wav, txt = self.wavs[index], self.texts[index]
        wav, sr = librosa.load(wav, 16000, mono=True)
        wav = self.processor(wav, sampling_rate=sr).input_values[0]
        with open(txt, 'r') as f:
            txt = f.read().strip('\n')
        with self.processor.as_target_processor():
            txt = self.processor(txt).input_ids
        feature = {
            'input_values': wav,
            'labels':txt,
            }
        return feature
    
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch