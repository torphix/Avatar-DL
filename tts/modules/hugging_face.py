from transformers import PreTrainedModel, PretrainedConfig

from tts_hf.utils import get_configs_of
from model.PortaSpeech import PortaSpeech


hf_config = PretrainedConfig(name_or_path='torphix/portaspeech-tts-keon-lee')

class PortaSpeechModel(PreTrainedModel):
    def __init__(self,
                 preprocessing_config, 
                 model_config,
                 train_config):
        super().__init__()
         
        self.model = PortaSpeech(preprocessing_config,
                                 model_config,
                                 train_config)
        
    def forward(self, 
                phonemes,
                src_lens,
                max_src_len,
                word_boundaries,
                src_w_lens,
                max_src_w_len):
        return self.model.forward(
                    phonemes,
                    src_lens,
                    max_src_len,
                    word_boundaries,
                    src_w_lens,
                    max_src_w_len)
        
        
preprocessing_config, model_config, train_config = get_configs_of()
model = PortaSpeechModel(preprocessing_config, 
                         model_config,
                         train_config)
