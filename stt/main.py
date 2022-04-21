from .asr import ASRInference
from .finetune import ASRFinetune
from stt.utils import auto_correct_oov, search_for_oov

def asr_inference(config):
    '''
    Check stt/config/inference.yaml for options
    '''
    asr = ASRInference(config)
    asr.transcribe()
    
def asr_finetune(config):
    '''
    Check stt/config/finetune.yaml for options
    '''
    asr = ASRFinetune(config)
    asr.finetune()

def find_oov(args):
    '''
    Iterates over transcriptions finding any
    words that do not appear in the lexicon
    returns words and corresponding wav files 
    to output file
    '''
    search_for_oov(args.input_dir, 
                   args.output_file,
                   args.lexicon)
    
def correct_oovs(args):
    '''
    Uses Peter norvigs pyspellchecker to 
    auto correct transcription 
    '''
    auto_correct_oov(args.oov_file)
    
