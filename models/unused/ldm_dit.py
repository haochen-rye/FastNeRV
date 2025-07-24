import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

# from random import randint, random
import random
from torchvision.transforms.functional import rgb_to_grayscale
from torch.nn.functional import interpolate
from torchvision.transforms import GaussianBlur
from diffusers.models import AutoencoderKL

import models
from models import register, NeRVEnc


@register('ldm_dit')
class LDMDit(nn.Module):
    def __init__(self, name, cond_dropout_prob, vae_name, num_patch, in_channels):
        super().__init__()
        # Build transformer backbone for video dinoising
        self.transformer_denoiser = models.dit_models[name](
            cond_dropout_prob=cond_dropout_prob, 
            num_patch=num_patch,
            in_channels=in_channels,
        )
        self.vae = self.load_vae(vae_name)
        self.z_h = self.z_w = int(math.sqrt(num_patch))
        # import pdb; pdb.set_trace; from IPython import embed; embed() 

    @torch.no_grad()
    def load_vae(self, vae_name='ema'):
        model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

    @torch.no_grad()
    def rgb2z(self, x):
        # import pdb; pdb.set_trace; from IPython import embed; embed() 
        z = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        return z


    @torch.no_grad()
    def z2rgb(self, z):
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=self.z_h, w=self.z_w)
        x = self.vae.decode(z / 0.18215).sample
        return x[None]        # t b c h w 


    def forward(self, x, t=None, y=None, fwd_type='denoise', cfg_scale=None):
        if fwd_type=='denoise':
            return self.transformer_denoiser(x, t, y)
        elif fwd_type=='guide_sample':
            return self.transformer_denoiser.forward_with_cfg(x, t, y, cfg_scale)
        elif fwd_type=='encode':
            return self.rgb2z(x)
        elif fwd_type=='decode':
            return self.z2rgb(x)
        else:
            NotImplementedError

