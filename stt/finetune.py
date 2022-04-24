import json
from huggingsound import (TrainingArguments,
                          SpeechRecognitionModel,
                          TokenSet)


class ASRFinetune():
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.output_dir = config['output_dir']
        self.model = SpeechRecognitionModel(config['model_path_or_name'],
                                            self.device)        
        self.tokens = TokenSet(config['tokens'])
        self.training_args = TrainingArguments(**config['training_args'])
        self.data_path = config['datapath']
        self.data_split = config['data_split']
        
    def load_datasets(self, dataset_path):
        with open(dataset_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(point.strip('\n')) for point in data]
        train_len = int(len(data)*self.data_split[0])
        eval_len = -int(len(data) - train_len)
        train_data, eval_data = data[:train_len], data[eval_len:]
        return train_data, eval_data

    def finetune(self):
        train_data, eval_data = self.load_datasets(self.data_path)
        self.model.finetune(
            self.output_dir, 
            training_args=self.training_args,
            train_data=train_data, 
            eval_data=eval_data, # the eval_data is optional
            token_set=self.tokens)
        
            