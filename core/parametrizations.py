import torch
import torch.nn as nn

from core.utils.common import requires_grad
from core.utils.class_registry import ClassRegistry


base_heads = ClassRegistry()


@base_heads.add_to_registry('svd_s')
class KernelSplit(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.singular_shift = nn.Parameter(torch.zeros(c_out))

    def forward(self):
        return {
            "singular": self.singular_shift,
        }


@base_heads.add_to_registry('cink_mult')
class ChannelInKernel(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.ch_in_wise_shift = nn.Parameter(torch.zeros(1, 1, c_in, 1, 1))
        self.kernel_shift = nn.Parameter(torch.zeros(1, 1, 1, 3, 3))

    def forward(self):
        return {
            "in": self.ch_in_wise_shift,
            "kernel": self.kernel_shift
        }


@base_heads.add_to_registry('coutk_mult')
class ChannelOutKernel(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.ch_out_wise_shift = nn.Parameter(torch.zeros(1, c_out, 1, 1, 1))
        self.kernel_shift = nn.Parameter(torch.zeros(1, 1, 1, 3, 3))

    def forward(self):
        return {
            "out": self.ch_out_wise_shift,
            "kernel": self.kernel_shift
        }
    

@base_heads.add_to_registry('cout_mult')
class ChannelOut(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.ch_out_wise_shift = nn.Parameter(torch.zeros(1, c_out, 1, 1, 1))

    def forward(self):
        return {
            "out": self.ch_out_wise_shift
        }


@base_heads.add_to_registry('aff_cout')
class ChannelOutAffine(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.beta = nn.Parameter(torch.zeros(1, c_out, 1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, c_out, 1, 1, 1))

    def forward(self):
        return {
            "beta": self.beta,
            "gamma": self.gamma
        }


@base_heads.add_to_registry('aff_cout_no_beta')
class ChannelOutAffineNoBeta(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.gamma = nn.Parameter(torch.ones(1, c_out, 1, 1, 1))

    def forward(self):
        return {
            "gamma": self.gamma
        }  

    
@base_heads.add_to_registry(['cfull_mult', 'cfull_delta'])
class ChannelFull(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.shift = nn.Parameter(torch.zeros(1, c_out, c_in, 1, 1))

    def forward(self):
        return {
            "shift": self.shift
        }


@base_heads.add_to_registry(['csep_mult', 'csep_delta'])
class ChannelSplit(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.ch_out_wise_shift = nn.Parameter(torch.zeros(1, c_out, 1, 1, 1))
        self.ch_in_wise_shift = nn.Parameter(torch.zeros(1, 1, c_in, 1, 1))

    def forward(self):
        return {
            "in": self.ch_in_wise_shift,
            "out": self.ch_out_wise_shift
        }


@base_heads.add_to_registry(['cin_mult', 'cin_delta', 'cin_offset'])
class ChannelIn(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.ch_in_wise_shift = nn.Parameter(torch.zeros(1, 1, c_in, 1, 1))

    def forward(self):
        return {
            "in": self.ch_in_wise_shift
        }


class BaseParametrization(nn.Module):
    level_to_conv_name_map = {
        "coarse": ["conv_0", "conv_1", "conv_2", "conv_3", "conv_4"],
        "medium": ["conv_5", "conv_6", "conv_7", "conv_8", "conv_9"],
        "fine": ["conv_10", "conv_11", "conv_12", "conv_13", "conv_14", "conv_15", "conv_16"]
    }

    def __init__(
        self,
        parameterization_type,
        conv_dimensions
    ):
        super().__init__()
        assert parameterization_type in base_heads, \
            f"""
            got param type - {parameterization_type}, available - {base_heads}
            """

        self.parameterization_type = parameterization_type
        self.heads = nn.ModuleDict({
            f"conv_{idx}": self._construct_head(conv_dimension) for idx, conv_dimension in enumerate(conv_dimensions)
        })
    
    def _construct_head(self, conv_dimension):
        return base_heads[self.parameterization_type](conv_dimension)

    def forward(self):
        return {key: model() for key, model in self.heads.items()}

    def freeze_layers(self, keys=None):
        """
        Disable training for all layers in list.
        """
        if keys is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for key in keys:
                requires_grad(self.heads[key], False)

    def unfreeze_layers(self, keys=None):
        """
        Enable training for all layers in list.
        """
        if keys is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for key in keys:
                requires_grad(self.heads[key], True)

    def get_training_layers(self, phase):
        keys = list(list(self.children())[0].keys())
        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return keys[2:4]
        if phase == 'shape':
            # layers 1-2
            return keys[1:2]
        if phase == 'no_fine':
            # const + layers 1-10
            return keys[:10]
        if phase == 'shape_expanded':
            # const + layers 1-3
            return keys
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers()
        else:
            # everything except mapping and ToRGB
            return keys

    def get_all_layers(self):
        return list(list(self.children())[0].keys())
