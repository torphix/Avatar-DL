import os
import yaml
import torch
import matplotlib
import numpy as np
import torch.nn as nn
from scipy.io import wavfile
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

matplotlib.use("Agg")

def get_configs():
    abs_path = os.path.abspath('.')
    preprocess_config = yaml.load(open(f"{abs_path}/configs/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(f"{abs_path}/configs/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(f"{abs_path}/configs/train.yaml", "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/kl_loss", losses[2], step)
        logger.add_scalar("Loss/pn_loss", losses[3], step)
        logger.add_scalar("Loss/dur_loss", losses[4], step)
        logger.add_scalar("Loss/helper_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return ~mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(model, targets, predictions, vocoder, model_config, preprocess_config, num_gpus):

    if num_gpus > 1:
        model = model.module

    basename = targets[0][0]
    src_len = predictions[6][0].item()
    mel_len = predictions[7][0].item()
    alignment = predictions[8][0][:, 0, :mel_len,
                               :src_len].float().detach().transpose(-2, -1)
    mel_target = targets[11][0, :mel_len].float().detach().transpose(0, 1)
    mel_mask = predictions[5][0, :mel_len].unsqueeze(0).detach()
    mel_reconst_vg = predictions[0][0,
                                    :mel_len].float().detach()
    # Variational Generator Reconstruction
    residual = predictions[11][0, :mel_len].unsqueeze(0).detach()
    out_residual = model.variational_generator.residual_layer(
        mel_reconst_vg.unsqueeze(0))
    mel_reconst_vg = mel_reconst_vg.transpose(0, 1)

    # PostNet Inference on the reconstruction
    mel_reconst_pn = model.postnet.inference(
        mel_mask.unsqueeze(1),
        g=(out_residual + residual).transpose(1, 2),
    )[0].float().detach()

    # Variational Generator Inference
    mel_prediction_vg, out_residual, _ = model.variational_generator.inference(
        predictions[7][0].unsqueeze(0), mel_mask, residual)
    mel_prediction_vg = mel_prediction_vg[0].float().detach().transpose(0, 1)

    # PostNet Inference on the inference
    mel_prediction_pn = model.postnet.inference(
        mel_mask.unsqueeze(1),
        g=(out_residual + residual).transpose(1, 2),
    )[0].float().detach()

    fig = plot_mel(
        [
            mel_prediction_pn.cpu().numpy(),
            mel_prediction_vg.cpu().numpy(),
            mel_reconst_pn.cpu().numpy(),
            mel_reconst_vg.cpu().numpy(),
            mel_target.cpu().numpy(),
        ],
        ["Synthetized Spectrogram", "VG-Synthetized Spectrogram", "PN-Reconstructed Spectrogram",
            "VG-Reconstructed Spectrogram", "Ground-Truth Spectrogram"],
    )
    attn_fig = plot_multi_attn(
        [
            alignment.cpu().numpy(),
        ],
        # ["Word-to-Phoneme Attention Alignment"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction_pn.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, attn_fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, output_path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        mel_len = predictions[7][i].item()
        mel_prediction = predictions[0][i, :, :mel_len].detach()

        fig = plot_mel(
            [mel_prediction.cpu().numpy()],
            ["Synthetized Spectrogram"])
        plt.savefig(f"{output_path}/{basename}png")
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[0]
    lengths = predictions[7] * \
        preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    print(output_path)
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(
            f'{output_path}/{basename}.wav',
            sampling_rate, wav)


def plot_mel(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False, figsize=(20, 12))
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small",
                               left=False, labelleft=False)
        axes[i][0].set_anchor("W")
    plt.tight_layout()

    return fig


def plot_multi_attn(data, titles=None, save_dir=None):
    figs = list()
    for i, attn in enumerate(data):
        fig = plt.figure()
        num_head = attn.shape[0]
        for j, head_ali in enumerate(attn):
            ax = fig.add_subplot(2, num_head // 2, j + 1)
            ax.set_xlabel(
                'Audio timestep') if j % 2 == 1 else None
            ax.set_ylabel('Text timestep') if j >= num_head-2 else None
            im = ax.imshow(head_ali, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        figs.append(fig)
        if save_dir is not None:
            plt.savefig(save_dir[i])
        plt.close()

    return figs


def plot_embedding(out_dir, embedding, embedding_speaker_id, gender_dict, filename='embedding.png'):
    colors = 'r', 'b'
    labels = 'Female', 'Male'

    data_x = embedding
    data_y = np.array(
        [gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10, 10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data == i, 0],
                    tsne_all_data[tsne_all_y_data == i, 1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        if type(x).__module__ == np.__name__:
            x = torch.tensor(x)
        x_padded = F.pad(
            x, (0, length - x.shape[0]), mode="constant", value=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        x = torch.tensor(x)
        PAD = 0
        if x.shape[0] > max_len:
            raise ValueError("not max_len")
        s = x.shape[1]
        x_padded = F.pad(
            x.transpose(0,1), (0, max_len - x.shape[0]), mode="constant", value=PAD)
        return x_padded.transpose(0,1)[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.shape[0] for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :input_.shape[0], :input_.shape[1]] = input_
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


def reparameterize(mu, logvar):
    """
    mu -- [batch_size, max_time, dim]
    logvar -- [batch_size, max_time, dim]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


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


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def plot_mels(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False, figsize=(20, 12))
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small",
                               left=False, labelleft=False)
        axes[i][0].set_anchor("W")
    plt.tight_layout()
    plt.savefig('otuput.png')
    return fig