from stt.utils import auto_correct_oov, search_for_oov
from .asr import ASRInference
from .finetune import ASRFinetune

def asr_inference(config):
    asr = ASRInference(config)
    asr.transcribe()
    
def asr_finetune(config):
    asr = ASRFinetune(config)
    asr.finetune()

def find_oov(args):
    search_for_oov(args.input_dir, 
                   args.output_file,
                   args.lexicon)
    
def correct_oovs(args):
    auto_correct_oov(args.oov_file)