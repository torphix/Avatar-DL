import torch
import numpy as np
from text import text_to_sequence
from torch.utils.data import Dataset
from utils import pad_1D, pad_2D, pad_3D



class TrainDataset(Dataset):
    def __init__(self, preprocess_config):
        
        self.data_path = f'{preprocess_config["path"]["preprocessed_path"]}'
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        with open(f'{self.data_path}/data.txt', 'r') as f:
            self.data = f.readlines()
        
    def calc_split(self, split_size):
        train_size = int(self.__len__() * split_size[0])
        val_size = int(self.__len__() - train_size)
        return train_size, val_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phonemes = torch.tensor(text_to_sequence(self.data[idx].split("|")[2],
                                                 self.cleaners))
        raw_text = self.data[idx].split("|")[-1]
        speaker = self.data[idx].split("|")[1]
        file_id = self.data[idx].split("|")[0]
        mel = np.load(f'{self.data_path}/mel/{speaker}-mel-{file_id}.npy')
        duration = np.load(f'{self.data_path}/duration/{speaker}-duration-{file_id}.npy')
        phones_per_word = np.load(f'{self.data_path}/phones_per_word/{speaker}-phones_per_word-{file_id}.npy')
        attn_prior = np.load(f'{self.data_path}/attn_prior/{speaker}-attn_prior-{file_id}.npy')
       
        sample = {
            "file_id": file_id,
            "phonemes": phonemes,
            "raw_text": raw_text,
            "mel": mel,
            "duration": duration,
            "word_boundary": phones_per_word,
            "attn_prior": attn_prior,
        }
        return sample
    
    def collate_fn(self, batch):
        file_ids = [datapoint['file_id'] for datapoint in batch]
        raw_text = [datapoint['raw_text'] for datapoint in batch]
        word_lens = torch.tensor([datapoint['word_boundary'].shape[0] for datapoint in batch])
        mel_lens = torch.tensor([len(datapoint['mel']) for datapoint in batch])
        phoneme_lens = torch.tensor([datapoint['phonemes'].shape[0] for datapoint in batch])
        phonemes = pad_1D([datapoint['phonemes'] for datapoint in batch])
        mel = pad_2D([datapoint['mel'] for datapoint in batch])
        durations = pad_1D([datapoint['duration'] for datapoint in batch])
        word_boundaries = pad_1D([datapoint['word_boundary'] for datapoint in batch])
        attn_priors = pad_3D([datapoint['attn_prior'] for datapoint in batch],
                             len(batch), max(phoneme_lens), max(mel_lens))
        
        return {
            'file_ids':file_ids,
            'raw_text':raw_text,
            'word_lens': word_lens,
            'max_word_len':max(word_lens),
            'phoneme_lens':phoneme_lens,
            'max_phoneme_len':max(phoneme_lens),
            'phonemes':phonemes,
            'mels':mel,
            'mel_lens':mel_lens,
            'max_mel_len':max(mel_lens),
            'durations':durations,
            'word_boundaries':word_boundaries,
            'attn_priors':attn_priors,
        }
            


class InferenceDataset(Dataset):
    pass