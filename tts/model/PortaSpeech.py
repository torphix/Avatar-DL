import torch.nn as nn
from .postnet import FlowPostNet
from utils import get_mask_from_lengths
from .linguistic_encoder import LinguisticEncoder
from .variational_generator import VariationalGenerator


class PortaSpeech(nn.Module):
    """ PortaSpeech """

    def __init__(self, preprocess_config, model_config, train_config):
        super(PortaSpeech, self).__init__()
        self.model_config = model_config

        self.linguistic_encoder = LinguisticEncoder(model_config, train_config)
        self.variational_generator = VariationalGenerator(
            preprocess_config, model_config)
        self.postnet = FlowPostNet(preprocess_config, model_config)

    def forward(
        self,
        phonemes,
        phoneme_lens,
        max_phoneme_len,
        word_boundaries,
        word_lens,
        max_word_len,
        attn_priors=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        durations=None,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(phoneme_lens, max_phoneme_len)
        src_w_masks = get_mask_from_lengths(word_lens, max_word_len)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len)
                     if mel_lens is not None else None)
    
        (output,
        log_d_predictions,
        d_rounded,
        mel_lens,
        mel_masks,
        alignments,
        alignment_logprobs)= self.linguistic_encoder(
                                             phonemes,
                                             phoneme_lens,
                                             word_boundaries,
                                             src_masks,
                                             word_lens,
                                             src_w_masks,
                                             mel_masks,
                                             max_mel_len,
                                             attn_priors,
                                             durations,
                                             d_control)

        residual = output
        if mels is not None: # Train
            output, out_residual, dist_info = self.variational_generator(
                mels, mel_lens, mel_masks, output)
            postnet_output = self.postnet(
                mels.transpose(1, 2),
                mel_masks.unsqueeze(1),
                g=(out_residual + residual).transpose(1, 2),
            )
        else: # Inference
            _, out_residual, dist_info = self.variational_generator.inference(
                mel_lens, mel_masks, output)
            output = self.postnet.inference(
                mel_masks.unsqueeze(1),
                g=(out_residual + residual).transpose(1, 2),
            )
            postnet_output = None

        return {
            'output':output,
            'postnet_output':postnet_output,
            'log_d_predictions':log_d_predictions,
            'd_rounded':d_rounded,
            'src_masks':src_masks,
            'mel_masks':mel_masks,
            'phoneme_lens':phoneme_lens,
            'mel_lens':mel_lens,
            'alignments':alignments,
            'dist_info':dist_info,
            'src_w_masks':src_w_masks,
            'residual':residual,
            'alignment_logprobs':alignment_logprobs}
