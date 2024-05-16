import torch
import torch.nn as nn
import re
from .projectors import CAbstractor
from transformers import PretrainedConfig
from .configuration_honeybee import HoneybeeConfig,HoneybeeVisualProjectorConfig
import torch.nn.functional as F

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_honeybee_projector(config, projector_type, num_tokens,lm_hidden_size):
    """Build projector (abstractor) and query_tokens (optionally for resampler)"""
    proj_config = config
    proj_type = projector_type
    num_tokens = num_tokens
    output_hidden_size = lm_hidden_size  # LM hidden size

    abstractor = {
        "c-abs": CAbstractor,
    }[
        proj_type
    ](proj_config, num_tokens, output_hidden_size)
    return abstractor


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'c-abs':

        local_config_path = config.mm_projector_config
        honeybee_config = HoneybeeVisualProjectorConfig.from_pretrained(local_config_path)

        num_tokens = config.mm_num_tokens

        lm_hidden_size = config.hidden_size

        abstractor = build_honeybee_projector(honeybee_config,projector_type,num_tokens,lm_hidden_size)
        return abstractor

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
