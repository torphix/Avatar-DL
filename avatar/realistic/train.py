import os
import torch
import yaml
import pytorch_lightning as ptl
from .data.utils import cut_video_sequence, sample_frames, shuffle_audio
from .models.generator import Generator
from .models.discriminators import VideoDiscriminator, SyncDiscriminator, FrameDiscriminator
from .data.data import GANDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from PIL import Image
from torchvision.utils import save_image
import torch.nn.functional as F
from collections import OrderedDict

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
          
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(
                        list(self.frame_discriminator.parameters()) + 
                        list(self.video_discriminator.parameters()) + 
                        list(self.sync_discriminator.parameters()), 
                                 lr=1e-5)
        # TODO Add scehduler
        return ([opt_g, opt_d], [])
        # return opt_gen
                
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=True)
    
    def d_fake_inference(self, batch, generated_images):
        batch['fake_video_all'] = generated_images
        batch['fake_frames_subset'] = sample_frames(generated_images,
                                                    batch['real_frames_subset'].size(1))
        batch['fake_video_blocks'] = cut_video_sequence(generated_images, 
                                                        batch['real_video_blocks'].size(2))
        # Discriminator inference
        frame_fake_out = self.frame_discriminator(batch['fake_frames_subset'],
                                                  batch['identity_frame'])
        sync_fake_output = self.sync_discriminator(batch['fake_video_blocks'],
                                                   batch['audio_chunks'])
        unsync_real_output = self.sync_discriminator(batch['real_video_blocks'].squeeze(0),
                                                     shuffle_audio(batch['audio_chunks']))
        video_fake_output = self.video_discriminator(batch['fake_video_all'].unsqueeze(0))
        outputs = OrderedDict({
            'frame_fake_out':frame_fake_out,
            'sync_fake_output':sync_fake_output,
            'unsync_real_output':unsync_real_output,
            'video_fake_output':video_fake_output,
        })
        return outputs
    
    def d_real_inference(self, batch):
        frame_real_out = self.frame_discriminator(batch['real_frames_subset'].squeeze(0), 
                                                  batch['identity_frame'])
        sync_real_output = self.sync_discriminator(batch['real_video_blocks'].squeeze(0),
                                                   batch['audio_chunks'])
        unsync_real_output = self.sync_discriminator(batch['real_video_blocks'].squeeze(0),
                                                     shuffle_audio(batch['audio_chunks']))
        video_real_output = self.video_discriminator(batch['real_video_all'])     
        outputs = OrderedDict({
            'frame_real_out':frame_real_out,
            'sync_real_output':sync_real_output,
            'unsync_real_output':unsync_real_output,
            'video_real_output':video_real_output,
        })
        return outputs
    
    def g_loss(self, d_outputs):
        # Binary cross entropy for all fake outputs
        targets = torch.cat((
                torch.ones((d_outputs['frame_fake_out'].size(0), 1)),
                torch.ones((d_outputs['sync_fake_output'].size(0), 1)),
                torch.ones((d_outputs['unsync_real_output'].size(0), 1)),
                torch.ones((d_outputs['video_fake_output'].size(0), 1)),
        ), dim=0).type_as(d_outputs['frame_fake_out'])
        outputs = torch.cat((
            d_outputs['frame_fake_out'].unsqueeze(1),
            d_outputs['sync_fake_output'],
            d_outputs['unsync_real_output'],
            d_outputs['video_fake_output'].unsqueeze(1),
        ))
        return F.binary_cross_entropy(outputs, targets)
    
    def d_loss(self, d_outputs):
        # Binary cross entropy for all fake outputs
        real_targets = torch.cat((
                torch.ones((d_outputs['frame_real_out'].size(0), 1)),
                torch.ones((d_outputs['sync_real_output'].size(0), 1)),
                torch.ones((d_outputs['video_real_output'].size(0), 1)),
        ), dim=0).type_as(d_outputs['frame_real_out'])

        real_outputs = torch.cat((
            d_outputs['frame_real_out'].unsqueeze(1),
            d_outputs['sync_real_output'],
            d_outputs['video_real_output'].unsqueeze(1),
        ))
        fake_targets = torch.cat((
                torch.zeros((d_outputs['frame_fake_out'].size(0), 1)),
                torch.zeros((d_outputs['sync_fake_output'].size(0), 1)),
                torch.zeros((d_outputs['unsync_real_output'].size(0), 1)),
                torch.zeros((d_outputs['video_fake_output'].size(0), 1)),
        ), dim=0).type_as(d_outputs['frame_fake_out'])
        fake_outputs = torch.cat((
            d_outputs['frame_fake_out'].unsqueeze(1),
            d_outputs['sync_fake_output'],
            d_outputs['unsync_real_output'],
            d_outputs['video_fake_output'].unsqueeze(1),
        ))
        real_loss = F.binary_cross_entropy(real_outputs, real_targets)
        fake_loss = F.binary_cross_entropy(fake_outputs, fake_targets)
        return (real_loss + fake_loss) / 2
    
    def reconstruction_loss(self, real_frames, fake_frames):
        '''Only calculate loss for bottom half'''
        real_frames = real_frames[:, :, real_frames.size(-2)//2:, :]
        fake_frames = fake_frames[:, :, fake_frames.size(-2)//2:, :]
        save_image(fake_frames, 'fake.png')
        loss = F.l1_loss(fake_frames, real_frames)
        return loss
    
    def forward(self, identity_frame, audio_frames):
        # Generate fake samples
        # batch = self.squeeze_batch(batch)
        generated_images = self.generator(
            identity_frame, 
            audio_frames.squeeze(0).transpose(1,2))
        return generated_images

    def training_step(self, batch, batch_idx, optimizer_idx):
        '''
        Right now only using the fake loss for generator
        Is worth trying using both fake and real loss to train the generator
        '''
        # Unpack batch
        identity_frame, audio_frames = batch['identity_frame'], batch['audio_generator_input']
        # Train generator
        if optimizer_idx == 0:        
            generated_images = self.forward(identity_frame, audio_frames)
            # Losses 
            recon_loss = self.reconstruction_loss(
                batch['real_video_all'].squeeze(0), 
                generated_images)
            g_loss = self.g_loss(self.d_fake_inference(batch, generated_images))
            g_loss = (recon_loss + g_loss) / 2
            
            tqdm_dict = {'g_loss':g_loss, 
                         'recon_loss':recon_loss}
            output = OrderedDict({
                'loss': recon_loss + g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
            })
            # Save images
            save_image((generated_images + 0.5)*0.5, 'fake_images.png')
            save_image(batch['real_video_all'].squeeze(0), 'real_images.png')
            return output
        # Train discriminator
        elif optimizer_idx == 1:
            generated_images = self.forward(identity_frame, audio_frames)
            fake_outputs = self.d_fake_inference(batch, generated_images)
            real_outputs = self.d_real_inference(batch)
            outputs = fake_outputs | real_outputs
            d_loss = self.d_loss(outputs)
            tqdm_dict = {'d_loss', d_loss}
            output = OrderedDict({
                'loss':d_loss,
                'progress_bar':tqdm_dict,
                'log':tqdm_dict,
            })
            return output
        else: raise Exception('''
                              Optimizer Idx should be 1 or 0, 
                              combine discriminator parameters''')

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
    
    os.makedirs(f'saved_models/{trainer_config["min_epochs"]}', exist_ok=True)
    torch.save(module.generator.state_dict(), f'saved_models/{trainer_config["min_epochs"]}/generator.pth')
    torch.save(module.sync_discriminator.state_dict(), f'saved_models/{trainer_config["min_epochs"]}/sync_discriminator.pth')
    torch.save(module.frame_discriminator.state_dict(), f'saved_models/{trainer_config["min_epochs"]}/frame_discriminator.pth')
    torch.save(module.video_discriminator.state_dict(), f'saved_models/{trainer_config["min_epochs"]}/video_discriminator.pth')
    
    
    # Next:
        # - Check all disc inputs
        # - Add the disc ouptuts loss (oppposite ) to the gneenrator loss target
    