import torch
import torch.nn as nn
import torch.nn.functional as F

from core.uda_models import uda_models
from core.utils.common import get_trainable_model_state, build_from_checkpoint


class InferenceWrapper(nn.Module):
    def __init__(self, ckpt, config):
        self.device = config.training.device
        self.model = build_from_checkpoint(ckpt)
        
        if ckpt['model_type'] != 'original':
            self.sg2 = uda_models['stylegan2'](
                img_size=1024, latent_size=512, 
                map_layers=8, channel_multiplier=2, 
                device='cuda:0', checkpoint_path=
            )
        
        self.source_generator = uda_models[self.config.training.generator](
            **self.config.generator_args[self.config.training.generator]
        )
        self.source_generator.patch_layers(self.config.training.patch_key)
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)
        
    def forward(self, latents, **kwargs):
        ...
