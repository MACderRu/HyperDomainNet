try:
    from .fused_act import FusedLeakyReLU, fused_leaky_relu
    from .upfirdn2d import upfirdn2d
except:
    from .upfirdn2d_torch_native import upfirdn2d
    from .fused_act_torch_native import FusedLeakyReLU, fused_leaky_relu
