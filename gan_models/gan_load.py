import json
import numpy as np
import torch
import torch.nn as nn

from gan_models.BigGAN import BigGAN, utils
from gan_models.ProgGAN.model import Generator as ProgGenerator
from gan_models.SNGAN.load import load_model_from_state_dict

try:
    from gan_models.StyleGAN2.model import Discriminator as StyleGan2Discriminator
    from gan_models.StyleGAN2.model import Generator as StyleGAN2Generator
except Exception as e:
    print('StyleGAN2 load fail: {}'.format(e))

from core.utils.class_registry import ClassRegistry

generator_registry = ClassRegistry()


class ConditionedBigGAN(nn.Module):
    def __init__(self, big_gan, target_classes=(239, )):
        super(ConditionedBigGAN, self).__init__()
        self.big_gan = big_gan
        self.target_classes = nn.Parameter(torch.tensor(target_classes, dtype=torch.int64),
            requires_grad=False)

        self.dim_z = self.big_gan.dim_z

    def set_classes(self, cl):
        try:
            cl[0]
        except Exception:
            cl = [cl]
        self.target_classes.data = torch.tensor(cl, dtype=torch.int64)

    def mixed_classes(self, batch_size):
        device = next(self.parameters()).device
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size).cuda()
        else:
            return torch.from_numpy(
                np.random.choice(self.target_classes.cpu(), [batch_size])).to(device)

    def forward(self, z, classes=None):
        if classes is None:
            classes = self.mixed_classes(z.shape[0]).to(z.device)

        cl_emb = self.big_gan.shared(classes).to(z.device)
        return self.big_gan(z, cl_emb)


class StyleGAN2Wrapper(nn.Module):
    def __init__(self, g, shift_in_w):
        super(StyleGAN2Wrapper, self).__init__()
        self.style_gan2 = g
        self.shift_in_w = shift_in_w
        self.dim_z = 512
        self.dim_shift = self.style_gan2.style_dim if shift_in_w else self.dim_z

    def forward(self, input, input_is_latent=False):
        return self.style_gan2([input], input_is_latent=input_is_latent)[0]

    def gen_shifted(self, z, shift):
        if self.shift_in_w:
            w = self.style_gan2.get_latent(z)
            return self.forward(w + shift, input_is_latent=True)
        else:
            return self.forward(z + shift, input_is_latent=False)


@generator_registry.add_func_to_registry("stylegan2")
def make_style_gan2(size, weights, latent_dim=512, n_layers_mlp=8, shift_in_w=True):
    G = StyleGAN2Generator(size, latent_dim, n_layers_mlp)
    G.load_state_dict(torch.load(weights, map_location='cpu')['g_ema'])
    G.cuda().eval()

    return StyleGAN2Wrapper(G, shift_in_w=shift_in_w)


def make_style_gan2_discriminator(size, weights_path):
    D = StyleGan2Discriminator(size)
    D.load_state_dict(torch.load(weights_path, map_location='cpu')['d'])
    return D


@generator_registry.add_func_to_registry("biggan")
def make_big_gan(config_path, weights_path, target_classes):
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config['skip_init'] = True
    config['no_optim'] = True

    G = BigGAN.Generator(**config)
    G.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=True)

    return ConditionedBigGAN(G, target_classes).eval()


@generator_registry.add_func_to_registry("proggan")
def make_proggan(weights_root):
    model = ProgGenerator()
    model.load_state_dict(torch.load(weights_root, map_location='cpu'))
    model.cuda()

    setattr(model, 'dim_z', [512, 1, 1])
    return model


@generator_registry.add_func_to_registry("sn_anime")
def make_sngan(gan_dir):
    gan = load_model_from_state_dict(gan_dir)
    G = gan.model.eval()
    setattr(G, 'dim_z', gan.distribution.dim)
    return G


@generator_registry.add_func_to_registry("sn_mnist")
def make_sngan(gan_dir):
    gan = load_model_from_state_dict(gan_dir)
    G = gan.model.eval()
    setattr(G, 'dim_z', gan.distribution.dim)
    return G
