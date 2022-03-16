from email.mime import audio
import os
import torch
import yaml
import random
import pytorch_lightning as ptl
from .data.utils import cut_video_sequence, sample_frames, shuffle_audio
from .models.generator import Generator
from .models.discriminators import VideoDiscriminator, SyncDiscriminator, FrameDiscriminator
from .data.data import GANDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


class TrainModule(ptl.LightningModule):
    def __init__(self, model_config, data_config):
        '''Fake: 1, Real 0'''
        super().__init__() 
        with open(model_config, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(data_config, 'r') as f:
            data_config = yaml.load(f, Loader=yaml.FullLoader)
        self.dataset = GANDataset(data_config)
        
        self.generator = Generator(model_config['generator'],
                                   model_config['img_size'])
        self.video_discriminator = VideoDiscriminator(model_config['video_discriminator'],
                                                      model_config['img_size'])
        self.sync_discriminator = SyncDiscriminator(model_config['sync_discriminator'],
                                                    model_config['img_size'])
        self.frame_discriminator = FrameDiscriminator(model_config['frame_discriminator'])
        # Loss discounts
        self.sync_loss_gamma = model_config['sync_loss_gamma']
        self.frame_loss_gamma = model_config['frame_loss_gamma']
        self.video_loss_gamma = model_config['video_loss_gamma']
        self.reconstruction_loss_gamma = model_config['reconstruction_loss_gamma']
        
    def squeeze_batch(self, batch):
        return {k:v.squeeze(0) if len(v) != 0 else []
                for k,v in batch.items()}

    def frame_loss(self, fake, real):
        loss = torch.log(real) + torch.log(1 - fake)
        return loss * self.frame_loss_gamma
    
    def sync_loss(self, sync_real, unsync_real, sync_fake):
        loss = torch.log(sync_real) + ((torch.log(1-unsync_real)) / 2) + ((torch.log(1-sync_fake)) / 2)
        return loss * self.sync_loss_gamma
    
    def video_loss(self, real, fake):
        loss = torch.log(real) + (torch.log(1- fake))
        return loss * self.video_loss_gamma
    
    def reconstruction_loss(self, real_frames, fake_frames):
        '''Only calculate loss for bottom half'''
        real_frames = real_frames[:, :, real_frames.size(2)//2:, :]
        fake_frames = real_frames[:, :, fake_frames.size(2)//2:, :]
        loss = torch.sum(real_frames - fake_frames) * self.reconstruction_loss_gamma
        return loss
    
    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        opt_des_frame = torch.optim.Adam(self.frame_discriminator.parameters(), lr=1e-4)
        opt_des_vid = torch.optim.Adam(self.video_discriminator.parameters(), lr=1e-5)
        opt_des_sync = torch.optim.Adam(self.sync_discriminator.parameters(), lr=1e-5)
        # TODO Add scehduler
        return ([opt_gen, opt_des_frame, opt_des_vid, opt_des_sync], [])
                
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=True)
    
    def forward(self, batch):
        # Generate fake samples
        # batch = self.squeeze_batch(batch)
        starting_frame, audio_frames = batch['first_frame'], batch['audio_generator_input']
        fake_samples = self.generator(starting_frame, 
                                      audio_frames.squeeze(0).transpose(1,2))
        batch['fake_video_all'] = fake_samples
        batch['fake_frames_subset'] = sample_frames(fake_samples, batch['real_frames_subset'].size(1))
        batch['fake_video_blocks'] = cut_video_sequence(fake_samples, 
                                                        batch['real_video_blocks'].size(2))

        # Discriminator inference
        frame_fake_out = self.frame_discriminator(batch['fake_frames_subset'], starting_frame)
        frame_real_out = self.frame_discriminator(batch['real_frames_subset'].squeeze(0), starting_frame)
        sync_fake_output = self.sync_discriminator(batch['fake_video_blocks'], batch['audio_chunks'])
        sync_real_output = self.sync_discriminator(batch['real_video_blocks'].squeeze(0), batch['audio_chunks'])
        unsync_real_output = self.sync_discriminator(batch['real_video_blocks'].squeeze(0), shuffle_audio(batch['audio_chunks']))
        print(batch['fake_video_all'].size())
        video_fake_output = self.video_discriminator(batch['fake_video_all'].unsqueeze(0))
        video_real_ouptut = self.video_discriminator(batch['real_video_all'].unsqueeze(0))
        # Losses
        frame_loss = self.frame_loss(frame_fake_out, frame_real_out)
        sync_loss = self.sync_loss(sync_real_output, unsync_real_output, sync_fake_output)
        video_loss = self.video_loss(video_real_ouptut, video_fake_output)
        recon_loss = self.reconstruction_loss(batch['real_video_all'], batch['fake_video_all'])
        total_loss = frame_loss + sync_loss + video_loss + recon_loss
        return total_loss


    def training_step(self, batch, batch_idx, optimizer_idx):
        loss = self.forward(batch)
        # Generator Optimization
        if optimizer_idx == 0:
            # Minimize the loss 
            return loss
        # Discriminator Optimization
        else:
            # Maximize the loss
            return -loss
        

def train():
    root = os.path.abspath('avatar/realistic')
    module = TrainModule(f'{root}/configs/models.yaml',
                         f'{root}/configs/data.yaml')    
    with open(f'{root}/configs/trainer.yaml', 'r') as f:
        trainer_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    ckpt_path = trainer_config['checkpoint_path']
    trainer_config.pop('checkpoint_path')
    
    trainer = Trainer(**trainer_config)
    trainer.fit(module, ckpt_path=ckpt_path)
    
    os.makedirs(f'saved_models/{trainer_config["epochs"]}')
    torch.save(module.generator.state_dict(), f'saved_models/{trainer_config["epochs"]}/generator.pth')
    torch.save(module.sync_discriminator.state_dict(), f'saved_models/{trainer_config["epochs"]}/sync_discriminator.pth')
    torch.save(module.frame_discriminator.state_dict(), f'saved_models/{trainer_config["epochs"]}/frame_discriminator.pth')
    torch.save(module.video_discriminator.state_dict(), f'saved_models/{trainer_config["epochs"]}/video_discriminator.pth')
    
    
    
    
# TODO: Adjust dims for batch processing