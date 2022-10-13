import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.utils import make_grid
from core.uda_models import uda_models
from core.utils.common import get_trainable_model_state, get_stylegan_conv_dimensions, align_face
from core.parametrizations import BaseParametrization
from core.mappers import mapper_registry
from restyle_encoders.psp import pSp
from argparse import Namespace

    
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
    

@torch.no_grad()
def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    return avg_image.to('cuda').float().detach()


def run_on_batch(inputs, net, opts, avg_image):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        y_hat, latent = net.forward(x_input,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=opts.resize_outputs)

        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].unsqueeze(0))

        # resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)

    return results_batch, results_latent


def load_latent(path):
    return torch.from_numpy(np.load(path)).unsqueeze(0)


def get_celeb_latents(names):
    if not isinstance(names, list):
        return load_latent(f'examples/celeb_latents/{names}.npy')
    
    return torch.cat([
        load_latent(f'examples/celeb_latents/{name}.npy') for name in names
    ], dim=0)


def to_im(torch_image):
    return transforms.ToPILImage()(
        make_grid(torch_image, value_range=(-1, 1), normalize=True)
    )


def load_inversion_model(inversion_args):
    model_path = inversion_args['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    opts.n_iters_per_batch = 5
    opts.resize_outputs = False
    
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    return net, opts


def run_alignment(image_path):
    import dlib
    if not os.path.exists("pretrained/shape_predictor_68_face_landmarks.dat"):
        print('dlib shape predictor is not downloaded; launch `python download.py --load_type=dlib`')
    predictor = dlib.shape_predictor("pretrained/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image