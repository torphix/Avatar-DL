import torch
import numpy as np
from utils import plot_mels
from utils import get_configs
from datetime import datetime
import pytorch_lightning as ptl
from data.dataset import TrainDataset
from torch.utils.data import DataLoader
from modules.loss import PortaSpeechLoss
from torch.utils.data import random_split
from model.PortaSpeech import PortaSpeech
from vocoder.api import get_vocoder, vocoder_infer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger


class TTSModule(ptl.LightningModule):
    def __init__(self, 
                preprocessing_config,
                model_config,
                train_config,
                device=None):
        super().__init__() 
        self.preprocess_config = preprocessing_config
        self.train_config = train_config
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.portaspeech = PortaSpeech(
            preprocessing_config,
            model_config,
            self.train_config)
        self.vocoder = get_vocoder(device)
        self.loss = PortaSpeechLoss(self.train_config)
        self.dataset = TrainDataset(preprocessing_config)
        self.train_ds, self.val_ds = random_split(self.dataset, 
                                                  self.dataset.calc_split(self.train_config['split_size']))
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          **self.train_config['dataloader'],
                          collate_fn=self.dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                          **self.train_config['dataloader'],
                          collate_fn=self.dataset.collate_fn)

    def forward(self, batch):
        inputs = batch.copy()
        inputs.pop('file_ids')
        inputs.pop('raw_text')
        outputs = self.portaspeech(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        losses = self.loss(batch['mels'], outputs)
        logs = {
            'train_total_loss':losses['total_loss'].detach(),
            'mel_loss':losses['mel_loss'].detach(),
            'kl_loss':losses['kl_loss'].detach(),
            'postnet_loss':losses['postnet_loss'].detach(),
            'duration_loss':losses['duration_loss'].detach(),
            'helper_loss':losses['helper_loss'].detach(),
        }
        self.log_dict(logs)
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        
        losses = self.loss(batch['mels'], outputs)
        wavs, fig = self.synthesize(outputs)
        if batch_idx != 0:
            self.log_media(batch['file_ids'][0], fig, wavs)
        logs = {
            'val_total_loss':losses['total_loss'].detach(),
        }
        self.log_dict(logs)
        return losses['total_loss']
    
    def synthesize(self, outputs):
        # Log outputs
        mel_len = outputs['mel_lens'][0].item()
        mel_mask = outputs['mel_masks'][0, :mel_len].unsqueeze(0).detach()
        # Variational Generator Reconstruction
        mel_reconst_vg = outputs['output'][0,:mel_len].float().detach()
        residual = outputs['residual'][0, :mel_len].unsqueeze(0).detach()
        out_residual = self.portaspeech.variational_generator.residual_layer(
            mel_reconst_vg.unsqueeze(0))
        mel_prediction_pn = self.portaspeech.postnet.inference(
            mel_mask.unsqueeze(1),
            g=(out_residual + residual).transpose(1, 2),
        )[0].float().detach()
        wavs = vocoder_infer(mel_prediction_pn.unsqueeze(0), self.vocoder)
        from scipy.io import wavfile
        for wav in wavs:
            wavfile.write('end.wav', 22050, wav.squeeze(0))
        fig = plot_mels([mel_prediction_pn.detach().cpu().numpy()], ['postnet'])
        return wavs, fig
    
    def log_media(self, ids, mels, wavs):
        # plot_mel(mels[0][i], 'Reconstructed')
        for i in range(len(wavs)):
            tensorboard = self.logger.experiment
            tensorboard.add_audio(ids[i], wavs[i].squeeze(0),
                                  sample_rate=self.preprocess_config['preprocessing']['audio']['sampling_rate'])
            tensorboard.add_figure(ids[i], mels)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.portaspeech.parameters(), 
                                     **self.train_config['optimizer'])
        return {'optimizer':optimizer,
                'scheduler': ReduceLROnPlateau(optimizer)}
        

def train(args):
    preprocessing_config, model_config, train_config = get_configs()
    module = TTSModule(preprocessing_config, 
                       model_config,
                       train_config)
    
    if train_config['trainer']['checkpoint_path'] is not None:
        print('Loading Trainer From checkpoint')
        module.load_from_checkpoint(train_config['trainer']['checkpoint_path'])
    if args.load_ckpt:
        print('Loading Portaspeech base model from checkpoint')
        # module.portaspeech.load_state_dict(torch.load(args.load_ckpt)['model'])
        module.portaspeech.load_state_dict(torch.load(args.load_ckpt))
        
    logger = TensorBoardLogger("tb_logs", name="portaspeech")
    trainer_config = train_config['trainer']
    trainer_config.pop('checkpoint_path')
    trainer = ptl.Trainer(logger=logger,
                          **trainer_config)
    trainer.fit(module)
    torch.save(
        {'model': module.portaspeech.state_dict()},
        f'trained_models/e{train_config["trainer"]["min_epochs"]}-{datetime.now().strftime("%d-%m-%y")}.pth')