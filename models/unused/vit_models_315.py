# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
import einops

from timm.models.vision_transformer import Attention, Mlp


#################################################################################
#                                 Core ViT Model                                #
#################################################################################

class ViTBlock(nn.Module):
    """
    A ViT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, proj_drop=0, attn_drop=0, mlp_drop=0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, proj_drop=proj_drop,attn_drop=attn_drop, **block_kwargs)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_drop)
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        x = x + self.attn(self.norm1(x))       # LN + Self-attention
        x = x + self.mlp(self.norm2(x))     # LN + FFN
        return x


class ViT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        proj_drop=0.,
        attn_drop=0.,
        mlp_drop=0.,
    ):
        super().__init__()
        self.hid_dim = hidden_size
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, proj_drop=proj_drop, 
                attn_drop=attn_drop, mlp_drop=mlp_drop) for _ in range(depth)
        ])
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


    def forward(self, x):
        """
        Forward pass of ViT.
        x: (N, T, D)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        for block in self.blocks:
            x = block(x)                      # (N, T, D)

        return x


#################################################################################
#                                   ViT Configs                                  #
#################################################################################
def ViT_H(**kwargs):
    return ViT(depth=32, hidden_size=1280, num_heads=16, **kwargs)

def ViT_XL(**kwargs):
    return ViT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def ViT_L(**kwargs):
    return ViT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def ViT_B(**kwargs):
    return ViT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def ViT_S(**kwargs):
    return ViT(depth=12, hidden_size=384, num_heads=6, **kwargs)

# ViT_models
ViT_models = {'ViT-H': ViT_H,
    'ViT-XL': ViT_XL, 'ViT-L':  ViT_L,  
    'ViT-B':  ViT_B,  'ViT-S':  ViT_S,  
}

