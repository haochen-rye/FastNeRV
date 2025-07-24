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
import numpy as np
import einops
import torch.nn.functional as F

from timm.models.vision_transformer import Attention, Mlp

class AttentionFull(Attention):
    def forward(self, x, attn_mask=None):
        assert self.fused_attn == True, 'fused_attn not supported'
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        self.attn = AttentionFull(hidden_size, num_heads=num_heads, qkv_bias=True, proj_drop=proj_drop,attn_drop=attn_drop, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_drop)

    def forward(self, x, attn_mask=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)       # LN + Self-attention
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


    def forward(self, x, z_token_list=None, sequential_decode=False, cumsum_z=False):
        """
        Forward pass of ViT.
        x: (N, T, D)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if z_token_list == None or sum(z_token_list)==0:
            if isinstance(x, list):
                x = torch.cat(x, dim=1)
            for block in self.blocks:
                x = block(x)                      # (N, T, D)
            return x
        else:
            assert len(x) == 3, f'input should have 3 parts: patch_tokens, z_tokens, d_tokens'
            patch_tokens, z_tokens, d_tokens = x
            x = torch.cat([patch_tokens, d_tokens], dim=1)
            z_tokens_list = z_tokens.split(z_token_list, dim=1)
            def get_valid_levels(z_token_list):
                valid_levels = []
                for z_idx, z_tokens in enumerate(z_token_list):
                    if z_tokens != 0:
                        valid_levels.append(z_idx)
                    else:
                        valid_levels[-1] = z_idx
                return valid_levels
            valid_levels = get_valid_levels(z_token_list)
            # import pdb; pdb.set_trace; from IPython import embed; embed()  
            f_list = []
            for block_id, block in enumerate(self.blocks):
                if block_id < len(z_tokens_list):
                    if sequential_decode:
                        x = [x, z_tokens_list[block_id]]
                    else:
                        x = [patch_tokens, d_tokens] + z_tokens_list[:block_id+1]
                x = block(x)                      # (N, T, D)
                if block_id < len(z_token_list):
                    f_list.append(x.clone())
                elif block_id == len(z_token_list):
                    f_list = [f_list[i][:,:patch_tokens.size(1)] for i in valid_levels]
                    if cumsum_z:
                        def deterministic_cumsum(f_list):
                            stacked_tensors = torch.stack(f_list, dim=0)  # This will have shape [n, b, c, l] where n is the number of tensors in f_list
                            cumsum_tensors = torch.zeros_like(stacked_tensors)
                            cumsum_tensors[0] = stacked_tensors[0]
                            for i in range(1, len(f_list)):
                                cumsum_tensors[i] = cumsum_tensors[i - 1] + stacked_tensors[i]

                            return cumsum_tensors
                        x = deterministic_cumsum(f_list).flatten(end_dim=1)
                    else:
                        x = torch.cat(f_list, 0)
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

def ViT_T(**kwargs):
    return ViT(depth=6, hidden_size=384, num_heads=6, **kwargs)

# ViT_models
ViT_models = {'ViT-H': ViT_H, 'ViT-T': ViT_T,
    'ViT-XL': ViT_XL, 'ViT-L':  ViT_L,  
    'ViT-B':  ViT_B,  'ViT-S':  ViT_S,  
}
