import re
import os
import yaml
import json
import torch
import argparse
import numpy as np
from g2p_en import G2p
from scipy.io import wavfile
from string import punctuation
from text import text_to_sequence
from vocoder.api import get_vocoder, vocoder_infer
from model.PortaSpeech import PortaSpeech
from utils import (get_configs,
                   plot_mels,
                   synth_samples,
                   word_level_subdivision)


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    word_boundaries = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phone_list = lexicon[w.lower()]
        else:
            phone_list = list(filter(lambda p: p != " ", g2p(w)))
        if phone_list:
            phones += phone_list
            word_boundaries.append(len(phone_list))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    if preprocess_config["preprocessing"]["text"]["sub_divide_word"]:
        word_boundaries = word_level_subdivision(
            word_boundaries, preprocess_config["preprocessing"]["text"]["max_phoneme_num"])

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(
        phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    ))

    return np.array(sequence), np.array(word_boundaries)


def tts_cmd_line(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    preprocess_config, model_config, train_config = get_configs()
    phonemes, word_boundaries = preprocess_english(
        args.text, preprocess_config)
    phonemes, word_boundaries = np.array(
        [phonemes]), np.array([word_boundaries])
    phoneme_lens = np.array([len(phonemes[0])])
    word_lens = np.array([len(word_boundaries[0])])
    batchs = [{
        'phonemes':torch.tensor(phonemes, device=device),
        'phoneme_lens':torch.tensor(phoneme_lens, device=device),
        'max_phoneme_len':torch.tensor(max(phoneme_lens), device=device),
        'word_lens':torch.tensor(word_lens, device=device),
        'max_word_len':torch.tensor(max(word_lens), device=device),
        'word_boundaries':torch.tensor(word_boundaries, 
                                       device=device),
    }]
    model = PortaSpeech(preprocess_config, 
                        model_config,
                        train_config).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    with torch.no_grad():
        for batch in batchs:
            output = model(**batch)

    vocoder = get_vocoder(device)
    wavs = vocoder_infer(output['output'].to(device), vocoder)
    for wav in wavs:
        wavfile.write('otuput.wav', 22050, wav.squeeze(0))
    plot_mels(output['output'].cpu().numpy(), ['Output'])