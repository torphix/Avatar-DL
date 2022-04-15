import os
import re
import yaml
import torch
import numpy as np
from g2p_en import G2p
from string import punctuation
from text import text_to_sequence
from model.PortaSpeech import PortaSpeech

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

def word_level_subdivision(phones_per_word, max_phoneme_num):
    res = []
    for l in phones_per_word:
        if l <= max_phoneme_num:
            res.append(l)
        else:
            s, r = l//max_phoneme_num, l % max_phoneme_num
            res += [max_phoneme_num]*s + ([r] if r else [])
    return res


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

def prepare_sample(text):
        ids = raw_texts = [text]
        root_path = os.path.abspath()
        with open(f'{root_path}/configs/model.yaml', 'r') as f:
            model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(f'{root_path}/configs/preprocess.yaml', 'r') as f:
            preprocess_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        # Speaker Info
        speakers = np.array([0])  # single speaker is allocated 0
        spker_embed = None
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts, word_boundaries = preprocess_english(
                text, preprocess_config)
            texts, word_boundaries = np.array(
                [texts]), np.array([word_boundaries])
        else:
            raise NotImplementedError
        text_lens = np.array([len(texts[0])])
        text_w_lens = np.array([len(word_boundaries[0])])
        batchs = [(ids, raw_texts, speakers, texts,
                   text_lens, max(text_lens), word_boundaries, text_w_lens, max(text_w_lens), spker_embed)]
        return batchs
    
def get_model(model_path, configs, device):
    (preprocess_config, model_config, train_config) = configs
    model = PortaSpeech(preprocess_config, model_config, train_config).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.requires_grad_ = False
    return model

def get_configs():
    root_path = os.path.abspath('.')
    preprocess_config = yaml.load(open(
            f"{root_path}/configs/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
            f"{root_path}/configs/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
            f"{root_path}/configs/train.yaml", "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config