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
import math
import einops

from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
import models

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Patch Embedding Layers           #
#################################################################################
class VidTokenizer(nn.Module):
    """ 3D Video to Patch Embedding
    """
    def __init__(self, num_frames=8, img_size=224, patch_size=16, in_chans=3, 
            embed_dim=768, norm_layer=None, flatten=True, bias=True):
        super().__init__()
        self.num_patches = num_frames * (img_size // patch_size)**2 
        self.video_shape = (in_chans, num_frames, img_size, img_size)
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size),
             stride=(1, patch_size, patch_size), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        assert x[0].shape == self.video_shape, f"Shape mismatch. Expected shape: {self.video_shape}, but got: {x[0].shape}"
        x = self.proj(x)
        if self.flatten:
            x = einops.rearrange(x, 'b c t h w -> b (t h w) c') # BCTHW -> BNC
        x = self.norm(x)
        return x

#################################################################################
#               Embedding Layers for Timesteps           #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, ada_LN=False, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.ada_LN = ada_LN
        if ada_LN:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        forward_func = self.adaLN_forward if self.ada_LN else self.simple_forward
        return forward_func(x, c) 

    def adaLN_forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def simple_forward(self, x, c):
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        token_num = x.size(1)  
        if c != None:
            x = torch.cat([x, c[:,None]], dim=1)
        x = x + self.attn(self.norm1(x))       # LN + Self-attention
        x = x + self.mlp(self.norm2(x))     # LN + FFN
        return x[:,:token_num] 


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cond_dropout_prob=0.1,
        cond_choice='no',
        ada_LN=False,
        t_embed=False,
        tokenizer={},
    ):
        super().__init__()
        self.hid_dim = hidden_size
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, ada_LN=ada_LN, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.cond_choice = cond_choice.lower()
        self.t_embed = t_embed
        if self.t_embed:
            self.t_embedder = TimestepEmbedder(hidden_size)
        if self.cond_choice not in ['no'] and tokenizer:
            self.cond_tokenizer = VidTokenizer(tokenizer['frame_num'], tokenizer['input_size'], 
                tokenizer['patch_size'], 3, hidden_size, bias=True)
            self.pos_embed = nn.Parameter(torch.randn(1, self.cond_tokenizer.num_patches, hidden_size))
        if '_attention' in self.cond_choice:
            self.att_blks = nn.ModuleList([
                Attention(hidden_size, num_heads=num_heads, qkv_bias=True) for _ in range(depth)
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

        # Initialize timestep embedding MLP:
        if self.t_embed:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if hasattr(block, 'self.ada_LN'):
                nn.init.constant_(block.self.ada_LN[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if hasattr(self, 'cond_tokenizer'):
            # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
            w = self.cond_tokenizer.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.cond_tokenizer.proj.bias, 0)


    def forward(self, x, t=None, cond=None):
        """
        Forward pass of DiT.
        x: (N, T, D)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # TODO: adding condition tokens: concat or cross attention
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        t = self.t_embedder(t) if (self.t_embed and t) else None        # (N, D)
        if cond:
            if self.cond_choice not in ['no']:
                cond = self.cond_tokenizer(cond) + self.pos_embed
            if self.cond_choice == 'concat':
                x = torch.cat([cond, x], dim=1)     # (N, T+M, D)
        for blk_id, block in enumerate(self.blocks):
            x = block(x, t)                      # (N, T, D)

        return x


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def DiT_H(**kwargs):
    return DiT(depth=32, hidden_size=1280, num_heads=16, **kwargs)

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)

# DiT_models
NeRV_Dit = {'DiT-H': DiT_H,
    'DiT-XL': DiT_XL, 'DiT-L':  DiT_L,  
    'DiT-B':  DiT_B,  'DiT-S':  DiT_S,  
}


