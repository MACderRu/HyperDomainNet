import torch

from core.utils.common import requires_grad
from core.utils.class_registry import ClassRegistry

from gan_models.StyleGAN2.offsets_model import (
    OffsetsGenerator,
    ModModulatedConv2d,
    DecModulatedConv2d,
    StyleModulatedConv2d
)

from core.stylegan_patches import decomposition_patches, modulation_patches, style_patches

uda_models = ClassRegistry()

# default_arguments = Omegaconf.structured(uda_models.make_dataclass_from_args("GenArgs"))
# default_arguments.GenArgs.stylegan2.size ...


@uda_models.add_to_registry("stylegan2")
class OffsetsTunningGenerator(torch.nn.Module):
    def __init__(self, img_size=1024, latent_size=512, map_layers=8,
                 channel_multiplier=2, device='cuda:0', checkpoint_path=None):
        super().__init__()

        self.generator = OffsetsGenerator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
        ).to(device)

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.generator.load_state_dict(checkpoint["g_ema"], strict=False)

        self.generator.eval()

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def patch_layers(self, patch_key):
        """
        Modify ModulatedConv2d Layers with <<patch_key>> patch
        """
        if patch_key in decomposition_patches:
            self._patch_modconv_key(patch_key, DecModulatedConv2d)
        elif patch_key in modulation_patches:
            self._patch_modconv_key(patch_key, ModModulatedConv2d)
        elif patch_key in style_patches:
            self._patch_modconv_key(patch_key, StyleModulatedConv2d)
        elif patch_key == 'original':
            ...
        else:
            raise ValueError(
                f'''
                Incorrect patch_key. Got {patch_key}, possible are {
                {decomposition_patches}, {modulation_patches}, {style_patches}
                }
                '''
            )
        return self
    
    def _patch_modconv_key(self, patch_key, mod_conv_class):
        self.generator.conv1.conv = mod_conv_class(
            patch_key, self.generator.conv1.conv
        )

        for conv_layer_ix in range(len(self.generator.convs)):
            self.generator.convs[conv_layer_ix].conv = mod_conv_class(
                patch_key, self.generator.convs[conv_layer_ix].conv
            )

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):
        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])
        if phase == 'shape':
            # layers 1-2
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-3
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'mapping':
            return list(self.get_all_layers())[0]
        if phase == 'affine':
            styled_convs = list(self.get_all_layers())[4]
            return [s.conv.modulation for s in styled_convs]
        if phase == 'conv_kernel':
            styled_convs = list(self.get_all_layers())[4]
            return [s.conv.weight for s in styled_convs]
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers()
        else:
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])

    def freeze_layers(self, layer_list=None):
        """
        Disable training for all layers in list.
        """
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        """
        Enable training for all layers in list.
        """
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        """
        Convert z codes to w codes.
        """
        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)

    def modulation_layers(self):
        return self.generator.modulation_layers

    def forward(self,
                styles,
                offsets=None,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                noise=None,
                randomize_noise=True):
        return self.generator(styles,
                              offsets=offsets,
                              return_latents=return_latents,
                              truncation=truncation,
                              truncation_latent=self.mean_latent,
                              noise=noise,
                              randomize_noise=randomize_noise,
                              input_is_latent=input_is_latent)
