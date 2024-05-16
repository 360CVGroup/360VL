from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage


from .configuration_honeybee import HoneybeeVisualProjectorConfig
from torch.nn import functional as F
import math

def build_pos_embeds(
    config: HoneybeeVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    # true
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: HoneybeeVisualProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    # 0
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: HoneybeeVisualProjectorConfig):
    # false
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    # 16,24
    src_size = int(math.sqrt(abs_pos.size(1)))
    # 32,48
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: HoneybeeVisualProjectorConfig,
        num_input_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(config, output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            # self.pos_emb = self.pos_emb[:,1:]
            pos_emb = get_abs_pos(self.pos_emb[:,1:], x.size(1))
            pos_emb = pos_emb.to(device=x.device)
            x += pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        return x


class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        # x = x[:, 1:]  # drop cls token and 2d forward

        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x


class CAbstractor(ConvProjector):
    """C-Abstractor"""
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_queries
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

