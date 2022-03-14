from operator import mod
import torch
import yaml
import random
import pytorch_lightning as ptl
from models.generator import Generator
from models.discriminators import VideoDiscriminator, SyncDiscriminator, FrameDiscriminator


class TrainModule(ptl.LightningModule):
    def __init__(self, model_config, data_config):
        super().__init__() 
        with open(model_config, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(data_config, 'r') as f:
            data_config = yaml.load(f, Loader=yaml.FullLoader)
            
        self.generator = Generator(model_config['generator'],
                                   model_config['img_size'])
        self.video_discriminator = VideoDiscriminator(model_config['video_discriminator'],
                                                      model_config['img_size'])
        self.sync_discriminator = SyncDiscriminator(model_config['sync_discriminator'],)
        self.frame_discriminator = FrameDiscriminator(model_config['frame_discriminator'])

    def train_dataloader(self):
        return

    def val_dataloader(self):
        return

    def training_step(self, batch, batch_idx):
        # Generate fake samples
        starting_frame, audio_frames = batch['first_frame'], batch['sliced_audio']
        starting_frame = starting_frame.expand(audio_frames.size(0), 
                                               *starting_frame.size())
        
        fake_samples = self.generator(starting_frame, audio_frames)
        batch['fake_video_all'] = fake_samples
        batch['fake_video_subset'] = random.sample(0, fake_samples.shape[0],
                                                   batch['real_frames_subset'].size(0))
        
        
        return {'loss': loss, 'logs':logs}

    def validation_step(self, batch, batch_idx):
        return {'loss': loss, 'logs':logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return {'optimizer':optimizer,
                'scheduler': ReduceLROnPlateau(optimizer)}
        
        
        