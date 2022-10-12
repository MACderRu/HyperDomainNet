import torch
import torch.nn as nn
import numpy as np

from core.utils.class_registry import ClassRegistry

modulation_patches = ClassRegistry()
decomposition_patches = ClassRegistry()


class Patch(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device
    
    def to(self, device):
        super().to(device)


class BaseModulationPatch(Patch):
    def __init__(self, conv_weight: torch.Tensor):
        super().__init__()
        self.shape = conv_weight.shape
        self.register_buffer('ones', torch.ones(self.shape))
        _, self.c_out, self.c_in, k_x, k_y = self.shape

    def forward(self, weight, offsets):
        raise NotImplementedError()
        

@modulation_patches.add_to_registry("csep_delta")
class ChannelSeparateDelta(BaseModulationPatch):
    def forward(self, weight, offsets):
        return weight + offsets['in'] + offsets['out']


@modulation_patches.add_to_registry("csep_mult")
class ChannelwiseSepMult(BaseModulationPatch):
    def forward(self, weight, offsets):
        mult = self.ones + offsets['in'] + offsets['out']
        return weight * mult


@modulation_patches.add_to_registry("cfull_mult")
class ChannelwiseFullMult(BaseModulationPatch):
    def forward(self, weight, offsets):
        mult = self.ones + offsets['shift']
        return weight * mult


@modulation_patches.add_to_registry("cfull_delta")
class ChannelwiseFullDelta(BaseModulationPatch):
    def forward(self, weight, offsets):
        return weight + offsets['shift']
    

@modulation_patches.add_to_registry("aff_cout")
class ChannelwiseFullDelta(BaseModulationPatch):
    def forward(self, weight, offsets):
        return offsets['gamma'] * weight + offsets['beta']


@modulation_patches.add_to_registry("aff_cout_no_beta")
class ChannelwiseFullDelta(BaseModulationPatch):
    def forward(self, weight, offsets):
        return offsets['gamma'] * weight


@modulation_patches.add_to_registry("coutk_mult")
class ChanneloutKernelMult(BaseModulationPatch):
    def forward(self, weight, offsets):
        mult = self.ones + offsets['out'] + offsets['kernel']
        return weight * mult


@modulation_patches.add_to_registry("cout_mult")
class ChannelOutMult(BaseModulationPatch):
    def forward(self, weight, offsets):
        mult = self.ones + offsets['out']
        return weight * mult

    
@modulation_patches.add_to_registry("cin_mult")
class ChannelINMult(BaseModulationPatch):
    def forward(self, weight, offsets):
        mult = self.ones + offsets['in']
        return weight * mult
    
    
@modulation_patches.add_to_registry("cink_mult")
class ChannelINKernelMult(BaseModulationPatch):
    def forward(self, weight, offsets):
        mult = self.ones + offsets['in'] + offsets['kernel']
        return weight * mult


class BaseDecompositionPatch(nn.Module):
    def __init__(self, weight):
        super().__init__()
        weight_matrix = weight.cpu().detach().numpy().reshape((weight.shape[-4:]))
        self.c_out, self.c_in, self.k_x, self.k_y = weight_matrix.shape
        weight_matrix = np.transpose(weight_matrix, (2, 3, 1, 0))
        weight_matrix = np.reshape(weight_matrix, (self.k_x * self.k_y * self.c_in, self.c_out))
        self._decompose_weight(weight_matrix)

    def _decompose_weight(self, weight_matrix: np.ndarray):
        raise NotImplementedError()

    def reconstruct(self, offsets):
        raise NotImplementedError()

    def forward(self, offsets):
        weight = self.reconstruct(offsets)
        weight = weight.view(1, self.k_x, self.k_y, self.c_in, self.c_out)
        weight = weight.permute(0, 4, 3, 1, 2).contiguous()
        return weight


class SvdDecompositionPatch(BaseDecompositionPatch):
    def _decompose_weight(self, weight: np.ndarray):
        u, s, vh = np.linalg.svd(weight, full_matrices=False)
        u = torch.FloatTensor(u)
        vh = torch.FloatTensor(vh)
        s = torch.FloatTensor(s)

        self.register_buffer('s', s)
        self.register_buffer('u', u)
        self.register_buffer('vh', vh)

    def reconstruct(self, offsets):
        raise NotImplementedError()


@decomposition_patches.add_to_registry("svd_s")
class SvdSingularDecomposePatch(SvdDecompositionPatch):
    def reconstruct(self, offsets):
        if offsets is None:
            return self.u @ torch.diag_embed(self.s) @ self.vh

        shifted_s = (self.s + offsets['singular'])
        return self.u @ torch.diag_embed(shifted_s) @ self.vh


@decomposition_patches.add_to_registry("svd_u_k")
class USVDFirstK(SvdDecompositionPatch):
    def reconstruct(self, offsets):
        # offsets is like [vector_dim, k]
        # return torch.cat([self.u_trainable, self.u_frozen], dim=1) @ torch.diag_embed(self.s) @ self.vh
        ...
