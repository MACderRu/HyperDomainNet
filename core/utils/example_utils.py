import torch
import torch.nn as nn
import torch.nn.functional as F

from core.uda_models import uda_models
from core.utils.common import get_trainable_model_state, get_stylegan_conv_dimensions
from core.parametrizations import BaseParametrization
from core.mappers import mapper_registry

    
class Inferencer(nn.Module):
    def __init__(self, ckpt, device):
        super().__init__()
        
        self.device = device
        self.model_type = ckpt['model_type']
        self.da_type = ckpt['da_type']
        
        self.sg2_source = uda_models['stylegan2'](
            img_size=ckpt['sg2_params']['img_size'],
            latent_size=ckpt['sg2_params']['latent_size'],
            map_layers=ckpt['sg2_params']['map_layers'],
            channel_multiplier=ckpt['sg2_params']['channel_multiplier'], 
            checkpoint_path=ckpt['sg2_params']['ckpt_path']
        )
        
        self.sg2_source.patch_layers(ckpt['patch_key'])
        self.sg2_source.freeze_layers()
        self.sg2_source.to(self.device)
        
        if self.model_type == 'original':
            self.model_da = uda_models['stylegan2'](
                img_size=ckpt['sg2_params']['img_size'],
                latent_size=ckpt['sg2_params']['latent_size'],
                map_layers=ckpt['sg2_params']['map_layers'],
                channel_multiplier=ckpt['sg2_params']['channel_multiplier'], 
                checkpoint_path=ckpt['sg2_params']['ckpt_path']
            )
            self.model_da.freeze_layers()
        elif self.model_type == 'mapper':
            self.model_da = mapper_registry[
                ckpt['mapper_config']['mapper_type']
            ](
                ckpt['mapper_config'],
                get_stylegan_conv_dimensions(ckpt['sg2_params']['img_size']),
            )
        else:
            self.model_da = BaseParametrization(
                ckpt['patch_key'],
                get_stylegan_conv_dimensions(ckpt['sg2_params']['img_size']),
            )
        self.model_da.load_state_dict(ckpt['state_dict'])
        self.model_da.to(self.device)
        
        if self.da_type == 'im2im':
            self.style_latents = ckpt['style_latents'].to(self.device)
        
    @torch.no_grad()
    def forward(self, latents, **kwargs):
        if not kwargs.get('input_is_latent', False):
            latents = self.sg2_source.style(latents)
            kwargs['input_is_latent'] = True
        
        src_imgs, _ = self.sg2_source(latents, **kwargs)
        
        if not kwargs.get('truncation', False):
            kwargs['truncation'] = 1
        
        if self.da_type == 'im2im':
            latents = self._mtg_mixing_noise(latents, truncation=kwargs['truncation'])
            kwargs.pop('truncation')
        
        if self.model_type == 'original':
            trg_imgs, _ = self.model_da(latents, **kwargs)
        elif self.model_type == 'mapper':
            trg_imgs, _ = self.sg2_source(latents, offsets=self.model_da(kwargs['mapper_input']), **kwargs)
        else:
            trg_imgs, _ = self.sg2_source(latents, offsets=self.model_da(), **kwargs)
        
        return src_imgs, trg_imgs
    
    def _mtg_mixing_noise(self, latents, truncation=1):
        w_styles = latents[0].unsqueeze(1).repeat(1, 18, 1)
        gen_mean = self.sg2_source.mean_latent.unsqueeze(1).repeat(1, 18, 1)
        
        style_mixing_latents = truncation * (w_styles - gen_mean) + gen_mean
        style_mixing_latents[:, 7:, :] = self.style_latents
        return [style_mixing_latents]
    