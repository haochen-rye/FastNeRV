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
import einops

# from timm.models.vision_transformer import Attention, Mlp
from .vit_model import ViTBlock
from models.transformers.dit_models import TimestepEmbedder, LabelEmbedder, FinalLayer
from utils import ClampTensor


#################################################################################
#                                   DiffuseViT Configs                                  #
#################################################################################
class DiffuseViT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        hidden_size=1152,
        depth=28,
        num_patch=4,
        num_heads=16,
        mlp_ratio=4.0,
        proj_drop=0.,
        attn_drop=0.,
        mlp_drop=0.,
        cond_dropout_prob=0.1,
        num_classes=1000,
        sum_t_y=False,
        att_every_block=False,
        in_channels=64,
        out_channels=3,
        learn_sigma=False,
        final_layer=True,
        final_modulate=False,
        final_bias='0',
        out_range='no'
    ):
        super().__init__()
        self.hid_dim = hidden_size
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, proj_drop=proj_drop, 
                attn_drop=attn_drop, mlp_drop=mlp_drop) for _ in range(depth)
        ])

        # Modules for diffusion
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, cond_dropout_prob)
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patch, hidden_size), requires_grad=True)
        if final_layer:
            self.final_modulate = final_modulate
            out_channels = out_channels * (2 if learn_sigma else 1)
            if final_modulate:
                self.final_prefc = nn.Linear(hidden_size, in_channels)
                self.final_layer = FinalLayer(in_channels, 1, out_channels)
            else:
                self.final_layer = nn.Sequential(
                    nn.LayerNorm(in_channels),
                    nn.Linear(in_channels, out_channels))
        self.sum_t_y = sum_t_y
        self.att_every_block = att_every_block
        self.final_bias = final_bias
        self.out_range = out_range
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # nn.init.normal_(self.pos_embed)  # Uses the default mean=0 and std=1

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        if self.final_modulate:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)


    def VidOut(self, x):
        if self.final_bias == 'tanh':
            x = torch.tanh(x)
        else:
            x = x + float(self.final_bias)
        x = ClampTensor(x, self.out_range)

        return x


    def final_forward(self, x, t, y):
        if hasattr(self, 'final_layer'):
            T, _, _, H, W = x.size()
            x = einops.rearrange(x, 't b c h w -> b (t h w) c')
            if self.final_modulate:
                t = self.t_embedder(t)                   # (N, D)
                y = self.y_embedder(y, self.training)    # (N, D)
                c = t + y                                # (N, D)
                x = self.final_layer(x, self.final_prefc(c))                # (b, t * h * w, out_c * 2)
            else:
                x = self.final_layer(x)                 # (b, t * h * w, out_c * 2)
            x = einops.rearrange(x, 'b (t h w) c -> b c t h w', t=T, h=H, w=W,)
        else:
            x = einops.rearrange(x, 't b c h w -> b c t h w')
        x = self.VidOut(x)
        return x
        # import pdb; pdb.set_trace; from IPython import embed; embed()   


    def forward(self, x, t, y, z_split_num=None):
        """
        Forward pass of ViT.
        x: (N, T, D)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # import pdb; pdb.set_trace; from IPython import embed; embed()   
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        if self.sum_t_y:
            c = (t + y).unsqueeze(-2)                                # (N, 1, D)
        else:
            c = torch.stack([t, y], dim=1)
        if z_split_num == None:
            return self.normal_forward(x, c)
        else:
            return self.tnerv_forward(x, c, z_split_num)


    def normal_forward(self, x, c):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        c_tokens = c.size(1)
        x = [x, c]
        for block in self.blocks:
            x = block(x)                      # (N, T, D)
            if self.att_every_block:
                x[:,-c_tokens:] = c
        
        return x


    def tnerv_forward(self, x, c, z_split_num):
        assert len(x) == 2, f'input should have 2 parts: patch_tokens, z_tokens'
        x, z_tokens = x
        z_tokens_list = z_tokens.split(z_split_num)
        c_tokens = c.size(1)
        for block_id, block in self.blocks:
            if block_id < len(z_tokens_list):
                x = [x, z_tokens_list[block_id], c]
            x = block(x)                      # (N, T, D)
            if self.att_every_block:
                x[:,-c_tokens:] = c
        
        return x


#################################################################################
#                                   DiffuseViT Configs                                  #
#################################################################################
def DiffuseViT_H(**kwargs):
    return DiffuseViT(depth=32, hidden_size=1280, num_heads=16, **kwargs)

def DiffuseViT_XL(**kwargs):
    return DiffuseViT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiffuseViT_L(**kwargs):
    return DiffuseViT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiffuseViT_B(**kwargs):
    return DiffuseViT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiffuseViT_S(**kwargs):
    return DiffuseViT(depth=12, hidden_size=384, num_heads=6, **kwargs)

# DiffuseViT_models
DiffuseViT_models = {'ViT-H': DiffuseViT_H,
    'ViT-XL': DiffuseViT_XL, 'ViT-L':  DiffuseViT_L,  
    'ViT-B':  DiffuseViT_B,  'ViT-S':  DiffuseViT_S,  
}

