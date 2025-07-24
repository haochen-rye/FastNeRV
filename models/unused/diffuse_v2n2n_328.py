import math
import os
from einops import rearrange

import torch
import torch.nn as nn

from models import register, make_model
from models.nerv_enc import NeRVEnc
from models.param_enc import ParamEnc
from models.dit_models import DiT_models


@register('diffuse_v2n2n')
class DiffuseV2N2N(nn.Module):
    def __init__(self, name, args, enc):
        super().__init__()
        # Build transformer backbone for video dinoising
        self.fix_nerv_enc = args['fix_nerv_enc']
        # import pdb; pdb.set_trace; from IPython import embed; embed() 
        self.nerv_enc = make_model(enc)
        # self.load_from_checkpoint(args['nerv_enc'], args['enc_type'])
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


    def load_from_checkpoint(self, ckpt, enc_type='nerv'):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f'checkpoint {ckpt} does not exist'
            ckpt = torch.load(ckpt, map_location= lambda storage, loc: storage)
        else:
            assert isinstance(ckpt, dict), f'checkpoint must be a dict or a path to a checkpoint'
        assert enc_type in ['nerv', 'param']
        enc_func = NeRVEnc if enc_type=='nerv' else ParamEnc
        model = enc_func(**ckpt['model']['args'])
        model.load_state_dict(ckpt['model']['sd'])
        if self.fix_nerv_enc:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        return model


    def rgb2z(self, x):
        z_info = self.nerv_enc.encode(x)
        return z_info

    
    def z2rgb(self, z):
        # x = self.nerv_enc.decode(z)['pred_frames']
        x = self.nerv_enc.decode(z)
        if isinstance(x, dict):
            x = x['pred_frames']        
        x = rearrange(x, 't b c h w->b c t h w')
        return x


    def forward(self, x, t=None, y=None, fwd_type='denoise', cfg_scale=None):
        if fwd_type=='denoise':
            z_info = self.rgb2z(x)   
            z = z_info['z']
            reconstruct_x = self.z2rgb(z)
            z_pred = self.transformer_denoiser(z.transpose(1,2), t, y)
            denoised_x = self.z2rgb(z_pred.transpose(1,2))
            return {**z_info, 'reconstruct_x': reconstruct_x, 'denoised_x': denoised_x, 'z_pred': z_pred, 'x': x}
        elif fwd_type=='guide_sample':
            z_pred = self.transformer_denoiser.forward_with_cfg(x, t, y, cfg_scale)
            return z_pred
        elif fwd_type == 'decode':
            denoised_x = self.z2rgb(x.transpose(1,2))
            return denoised_x
        else:
            NotImplementedError(f'Unsupported fwd_type {fwd_type}')

