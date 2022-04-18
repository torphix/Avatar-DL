from sklearn import preprocessing
import yaml
from data.datasets.lex_fridman.lex import create_lex_dataset


def create_dataset(dataset_name):
    if dataset_name == 'lex':
        processing_path = 'data/datasets/lex_fridman/config.yaml'
        with open(processing_path, 'r') as f:
            processing_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            create_lex_dataset(processing_config)
    else:
        raise ValueError('Only dataset lex currently supported')
        
    