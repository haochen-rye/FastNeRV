import math
import os

import torch
import torch.nn as nn

from models import register
from models.param_enc import ParamEnc
from models.dit_models import DiT_models


@register('diffuse_param')
class DiffuseParam(nn.Module):
    # def __init__(self, name, cond_dropout_prob, nerv_enc):
    def __init__(self, name, args):
        super().__init__()
        # Build transformer backbone for video dinoising
        self.nerv_enc = self.load_from_checkpoint(args['nerv_enc'])
        self.z_num, self.z_dim = self.nerv_enc.z_token_num, self.nerv_enc.z_dim
        self.transformer_denoiser = DiT_models[name](
            cond_dropout_prob = args['cond_dropout_prob'], 
            num_patch = self.z_num,
            in_channels = self.z_dim,
            split_num = args['split_num'],
            learn_sigma = args['learn_sigma'],
            learn_pe = args['learn_pe'],
        )
        self.bottleneck_str = self.nerv_enc.bottleneck_str
        self.n_frames = self.nerv_enc.frame_num
        # import pdb; pdb.set_trace; from IPython import embed; embed() 


    def load_from_checkpoint(self, ckpt):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f'checkpoint {ckpt} does not exist'
            ckpt = torch.load(ckpt, map_location= lambda storage, loc: storage)
        else:
            assert isinstance(ckpt, dict), f'checkpoint must be a dict or a path to a checkpoint'
        model = ParamEnc(**ckpt['model']['args'])
        model.load_state_dict(ckpt['model']['sd'])
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model


    @torch.no_grad()
    def rgb2z(self, x):
        z = self.nerv_enc.encode(x)['z']
        return z

    
    @torch.no_grad()
    def z2rgb(self, z):
        x = self.nerv_enc.decode(z)
        if isinstance(x, dict):
            x = x['pred_frames']
        return x

    # def forward_with_cfg(self, x, t, y, cfg_scale):
    #     return self.transformer_denoiser.forward_with_cfg(x, t, y, cfg_scale)

    def forward(self, x, t=None, y=None, fwd_type='denoise', cfg_scale=None):
        # import pdb; pdb.set_trace; from IPython import embed; embed()   
        if fwd_type=='denoise':
            return self.transformer_denoiser(x, t, y)
        elif fwd_type=='guide_sample':
            return self.transformer_denoiser.forward_with_cfg(x, t, y, cfg_scale)
        elif fwd_type=='encode':
            return self.rgb2z(x)
        elif fwd_type=='decode':
            return self.z2rgb(x)
        else:
            NotImplementedError(f'Unsupported fwd_type {fwd_type}')

