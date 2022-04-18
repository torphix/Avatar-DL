import torch
import torch.nn.functional as F
import numpy as np

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return ~mask


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :np.shape(input_)[0], :np.shape(input_)[1]] = input_
    return inputs_padded


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def word_level_pooling(src_seq, src_len, wb, src_w_len, reduce="sum"):
    """
    src_seq -- [batch_size, max_time, dim]
    src_len -- [batch_size,]
    wb -- [batch_size, max_time]
    src_w_len -- [batch_size,]
    """
    batch, device = [], src_seq.device
    for s, sl, w, wl in zip(src_seq, src_len, wb, src_w_len):
        m, split_size = s[:sl, :], list(w[:wl].int())
        m = nn.utils.rnn.pad_sequence(torch.split(m, split_size, dim=0))
        if reduce == "sum":
            m = torch.sum(m, dim=0)  # [src_w_len, hidden]
        elif reduce == "mean":
            m = torch.div(torch.sum(m, dim=0), torch.tensor(
                split_size, device=device).unsqueeze(-1))  # [src_w_len, hidden]
        else:
            raise ValueError()
        batch.append(m)
    return pad(batch).to(device)


def word_level_subdivision(phones_per_word, max_phoneme_num):
    res = []
    for l in phones_per_word:
        if l <= max_phoneme_num:
            res.append(l)
        else:
            s, r = l//max_phoneme_num, l % max_phoneme_num
            res += [max_phoneme_num]*s + ([r] if r else [])
    return res


def reparameterize(mu, logvar):
    """
    mu -- [batch_size, max_time, dim]
    logvar -- [batch_size, max_time, dim]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu