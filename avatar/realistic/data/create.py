import yaml
from .lex import create_lex_dataset


def create_dataset(dataset_name):
    if dataset_name == 'lex':
        processing_path = 'avatar/realistic/configs/processing/lex.yaml'
        with open(processing_path, 'r') as f:
            processing_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            create_lex_dataset(
                        processing_config['links'],
                        processing_config['silence_length_crop'],
                        processing_config['video_fps'],
                        processing_config['max_frames'],
                        processing_config['clip_min_len'],
                        processing_config['clip_max_len'])
    else:
        raise ValueError('Only dataset lex currently supported')
        
    