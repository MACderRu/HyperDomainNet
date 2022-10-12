import math
import random
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .op import FusedLeakyReLU

from gan_models.StyleGAN2.model import (
    PixelNorm, EqualLinear, Blur, ModulatedConv2d,
    NoiseInjection, ConstantInput, ToRGB
)

from core.stylegan_patches import modulation_patches, decomposition_patches


class ModModulatedConv2d(nn.Module):
    def __init__(
        self,
        patch_key,
        conv_to_patch
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate
        self.weight = conv_to_patch.weight

        self.offsets_modulation = modulation_patches[patch_key](
            conv_to_patch.weight
        )
        # self.matrix_parametrizator.matrix_decomposition(conv_to_patch.weight)

    def forward(self, input, style, offsets=None):
        batch, in_channel, height, width = input.shape

        # further code is from original ModulatedConv2d
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        if offsets is not None:
            weight = self.offsets_modulation(weight, offsets)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class DecModulatedConv2d(nn.Module):
    def __init__(
        self,
        patch_key,
        conv_to_patch,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate
        self.matrix_parametrizator = decomposition_patches[patch_key](
            conv_to_patch.weight
        )

    def forward(self, input, style, offsets=None):
        batch, in_channel, height, width = input.shape
        weight = self.matrix_parametrizator(offsets)

        # further code is from original ModulatedConv2d
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyleModulatedConv2d(nn.Module):
    def __init__(
            self,
            patch_key,
            conv_to_patch
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate
        self.weight = conv_to_patch.weight

        self.offsets_modulation = style_patches[patch_key](
            conv_to_patch.weight
        )

    def forward(self, input, style, offsets=None):
        batch, in_channel, height, width = input.shape
        
        if self.offsets_modulation.style_space() == 'w' and offsets is not None:
            style = self.offsets_modulation(style, offsets)
        
        style = self.modulation(style) # style_space = modulation(w+) == affine(w+)
        
        if self.offsets_modulation.style_space() == 's' and offsets is not None:
            style = self.offsets_modulation(style, offsets)
        
        # further code is from original ModulatedConv2d
        style = style.view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class OffsetsStyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, offsets, noise=None):
        out = self.conv(input, style, offsets)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class OffsetsGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = OffsetsStyledConv(
            self.channels[4],
            self.channels[4],
            3, style_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer('noise_{}'.format(layer_idx), torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                OffsetsStyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                OffsetsStyledConv(
                    out_channel,
                    out_channel,
                    3,
                    style_dim,
                    blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        offsets=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=False
    ):

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, 'noise_{}'.format(i)) for i in range(self.num_layers)]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(
            out, latent[:, 0],
            offsets['conv_0'] if offsets is not None else None,
            noise=noise[0]
        )

        skip = self.to_rgb1(out, latent[:, 1])
        i = 1

        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(
                out, latent[:, i],
                offsets['conv_{}'.format(i)] if offsets is not None else None,
                noise=noise1
            )
            out = conv2(
                out, latent[:, i + 1],
                offsets['conv_{}'.format(i + 1)] if offsets is not None else None,
                noise=noise2
            )
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None