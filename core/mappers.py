import torch
import torch.nn as nn
import typing as tp

from core.utils.class_registry import ClassRegistry


mapper_registry = ClassRegistry()


def initialize_linear_layer(lin_layer):
    nn.init.xavier_uniform_(lin_layer.weight)
    lin_layer.weight.data = lin_layer.weight.data * 0.01
    lin_layer.bias.data.fill_(0.)


def initialize_mapper_weights(m):
    if isinstance(m, nn.Linear):
        initialize_linear_layer(m)


activations = {
    'lrelu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'relu': nn.ReLU,
    'id': nn.Identity,
}

kwargs = {
    'lrelu': {
        "negative_slope": 0.2
    },
    'relu': {},
    'prelu': {},
    'id': {}
}

base_blocks = ClassRegistry()


class ViewBlock(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        b = x.size(0)
        return x.view(b, *self.shape)


@base_blocks.add_to_registry("bnlinrelu")
class BNLinRelu(nn.Module):
    def __init__(self, input_features, output_features, bias=True, activation='relu'):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_features)
        self.lin = nn.Linear(input_features, output_features, bias=bias)
        self.act = activations[activation](**kwargs[activation])

    def forward(self, x):
        return self.bn(self.act(self.lin(x)))  # TODO: resnet-like block


@base_blocks.add_to_registry("resblock")
class DummyResBlock(nn.Module):
    def __init__(self, features, inner_features, activation='relu'):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Linear(features, inner_features),
            nn.BatchNorm1d(inner_features),
            activations[activation](**kwargs[activation]),
        )

        self.second_block = nn.Sequential(
            nn.Linear(inner_features, features),
            nn.BatchNorm1d(features),
        )

        self.out = activations[activation](**kwargs[activation])

    def forward(self, x):
        x_ = self.input_block(x)
        x_ = self.second_block(x_)
        return self.out(x_ + x)


class ZeroOutBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, x: torch.Tensor):
        batch_size, device = x.size(0), x.device
        return {
            "in": torch.zeros(batch_size, 1, self.c_in, 1, 1, device=device),
            "out": torch.zeros(batch_size, self.c_out, 1, 1, 1, device=device)
        }


class MapperBackbone(nn.Module):
    def __init__(self, mapper_config):
        super().__init__()

        self.backbone_type = mapper_config.backbone_type
        assert self.backbone_type in ['shared', 'levels'], \
            "incorrect backbone_type, possible are [shared, levels], current is {}".format(self.backbone_type)

        self.activation, self.input_dim, self.width, self.backbone_depth = (
            mapper_config.activation,
            mapper_config.input_dimension,
            mapper_config.width,
            mapper_config.backbone_depth,
        )

        self.no_coarse, self.no_medium, self.no_fine = (
            mapper_config.no_coarse,
            mapper_config.no_medium,
            mapper_config.no_fine
        )

        self.build()

    def build(self):
        if self.backbone_type == "shared":
            self._build_shared_backbone()
        elif self.backbone_type == "levels":
            self._build_levels_bacbone()

    def _build_shared_backbone(self):
        modules = [BNLinRelu(self.input_dim, self.width, activation=self.activation)]

        for _ in range(self.backbone_depth - 1):
            modules += [DummyResBlock(self.width, inner_features=512, activation=self.activation)]

        self.backbone = nn.Sequential(*modules)

    def _build_levels_bacbone(self):
        self.backbone = nn.ModuleDict({
            "fine": self._build_shared_backbone() if not self.no_fine else nn.Identity(),
            "coarse": self._build_shared_backbone() if not self.no_coarse else nn.Identity(),
            "medium": self._build_shared_backbone() if not self.no_medium else nn.Identity()
        })

    def forward(self, x):
        if isinstance(self.backbone, nn.ModuleDict):
            return {key: self.backbone[key](x) for key in self.backbone}
        elif isinstance(self.backbone, nn.Module):
            return self.backbone(x)


class BaseMapper(nn.Module):
    level_to_conv_name_map = {
        "coarse": ["conv1", "conv_1", "conv_2", "conv_3", "conv_4"],
        "medium": ["conv_5", "conv_6", "conv_7", "conv_8", "conv_9"],
        "fine": ["conv_10", "conv_11", "conv_12", "conv_13", "conv_14", "conv_15", "conv_16"]
    }

    def __init__(
        self,
        mapper_config
    ):
        super().__init__()
        self.activation, self.input_dim, self.width, self.backbone_depth, self.head_depth = (
            mapper_config.activation,
            mapper_config.input_dimension,
            mapper_config.width,
            mapper_config.backbone_depth,
            mapper_config.head_depth
        )

        self.backbone = MapperBackbone(mapper_config)
    
    def construct_nonresidual_head(self, c_in, c_out) -> nn.Module:
        in_head = [BNLinRelu(self.width, self.width, activation=self.activation) for _ in range(self.head_depth - 1)] + [nn.Linear(self.width, c_in),]
        out_head = [BNLinRelu(self.width, self.width, activation=self.activation) for _ in range(self.head_depth - 1)] + [nn.Linear(self.width, c_out),]

        heads = nn.ModuleDict({
            "out": nn.Sequential(*out_head, ViewBlock((c_out, 1, 1, 1))),
            "in": nn.Sequential(*in_head, ViewBlock((1, c_in, 1, 1)))
        })

        return heads

    def construct_residual_head(self, c_in, c_out) -> nn.Module:
        in_head = [DummyResBlock(self.width, inner_features=512, activation=self.activation) for _ in range(self.head_depth - 1)] + [nn.Linear(self.width, c_in), ]
        out_head = [DummyResBlock(self.width, inner_features=512, activation=self.activation) for _ in range(self.head_depth - 1)] + [nn.Linear(self.width, c_out), ]

        heads = nn.ModuleDict({
            "out": nn.Sequential(*out_head, ViewBlock((c_out, 1, 1, 1))),
            "in": nn.Sequential(*in_head, ViewBlock((1, c_in, 1, 1)))
        })

        return heads
    
    def construct_residual_head_cin(self, c_in, c_out) -> nn.Module:
        in_head = [DummyResBlock(self.width, inner_features=512, activation=self.activation) for _ in range(self.head_depth - 1)] + [nn.Linear(self.width, c_in), ]

        heads = nn.ModuleDict({
            "in": nn.Sequential(*in_head, ViewBlock((1, c_in, 1, 1)))
        })

        return heads

    def forward(self, x):
        backbone_out = self.backbone(x)
        answer = {}
        if not isinstance(backbone_out, dict):
            for key in self.heads:
                answer[key] = {
                    inner_key: self.heads[key][inner_key](backbone_out) for inner_key in self.heads[key]
                }
            return answer
        
        for out_key in backbone_out:
            for key in self.level_to_conv_name_map[out_key]:
                answer[key] = {
                    inner_key: self.heads[key][inner_key](backbone_out[out_key]) for inner_key in self.heads[key]
                }

        return answer


@mapper_registry.add_to_registry("residual_channelwise_sep")
class ResidualMapper(BaseMapper):
    def __init__(
        self,
        mapper_config,
        conv_dimensions
    ):
        super().__init__(mapper_config)

        self.heads = nn.ModuleDict({
            "conv_{}".format(idx): self.construct_residual_head(*dims) for idx, dims in enumerate(conv_dimensions)
        })

        self.apply(initialize_mapper_weights)


@mapper_registry.add_to_registry("residual_channelin")
class ResidualMapper(BaseMapper):
    def __init__(
        self,
        mapper_config,
        conv_dimensions
    ):
        super().__init__(mapper_config)

        self.heads = nn.ModuleDict({
            "conv_{}".format(idx): self.construct_residual_head_cin(*dims) for idx, dims in enumerate(conv_dimensions)
        })

        self.apply(initialize_mapper_weights)
        

@mapper_registry.add_to_registry("base_channelwise_sep")
class BaseChannelWiseMapper(BaseMapper):
    def __init__(
        self,
        mapper_config: tp.Dict,
        conv_dimensions: tp.List
    ):
        """
        mapper_config: Mappable - mapper config with core components
        conv1: - first StyledConv of StyleGAN
        convs: - list of further StyledConvs
        """

        super().__init__(mapper_config)
        self.heads = nn.ModuleDict({
            "conv_{}".format(idx): self.construct_nonresidual_head(*dims) for idx, dims in enumerate(conv_dimensions)
        })
        self.apply(initialize_mapper_weights)

    
@mapper_registry.add_to_registry("levelsheads_channelwise_sep")
class ChannelWiseMapperLevels(BaseMapper):
    def __init__(
        self,
        mapper_config,
        conv_dimension: tp.List
    ):
        """
        mapper_config: Mappable - mapper config with core components
        conv1: - first StyledConv of StyleGAN
        convs: - list of further StyledConvs
        """

        super().__init__(mapper_config)
        
        self.no_coarse, self.no_medium, self.no_fine = (
            mapper_config.no_coarse,
            mapper_config.no_medium,
            mapper_config.no_fine
        )

        idx_conv = list(enumerate(conv_dimension))

        self.heads = nn.ModuleDict({
            "conv_{}".format(idx): self.construct_residual_head(*dims) if not self.no_coarse else ZeroOutBlock(*dims)
            for idx, dims in idx_conv[:5]
        })
        
        self.heads.update({
            "conv_{}".format(idx): self.construct_residual_head(*dims) if not self.no_medium else ZeroOutBlock(*dims)
            for idx, dims in idx_conv[5:9]
        })
        
        self.heads.update({
            "conv_{}".format(idx): self.construct_residual_head(*dims) if not self.no_fine else ZeroOutBlock(*dims)
            for idx, dims in idx_conv[9:]
        })

        self.apply(initialize_mapper_weights)
