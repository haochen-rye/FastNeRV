import torch
import torch.nn as nn
import numpy as np
import einops
from math import ceil

from models import register
from .layers import batched_conv
from utils import split_str, get_sincos_pos_embed, ActLayer, ClampTensor


@register('nerv_model')
class NeRVModel(nn.Module):
    def __init__(self, frame_num, in_dim, out_dim, hid_dim, 
        strds, ks, pos_embed, pe_dim, learn_pe, act='relu', out_bias=0.5, out_range='no'):
        super().__init__()        
        self.frame_num, self.pos_embed, self.pe_dim = frame_num, pos_embed, pe_dim
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        self.pe_embed = nn.Parameter(self.get_pos_embed(), requires_grad=learn_pe)      # T D
        strds_list, ks_list = split_str(strds), split_str(ks)
        depth = len(strds_list)
        ks_list += ks_list[-1:] * (depth - len(ks_list))
        self.ps_layers = nn.ModuleList()
        self.conv_shape_list, self.param_shapes = [], {}
        last_dim, ch = in_dim*pe_dim, hid_dim
        for i, cur_strd in enumerate(strds_list):
            cur_dim = ch if i < depth - 1 else out_dim
            cur_ks = ks_list[i]
            ch_out = cur_dim*cur_strd**2
            self.param_shapes[f'wb{i}'] = (ch_out, last_dim, cur_ks)
            cur_pad = 0 if (cur_ks == cur_strd and i == 0) else ceil(cur_ks - 1) // 2
            self.conv_shape_list.append((ch_out, last_dim, cur_ks, cur_pad)) 
            self.ps_layers.append(nn.PixelShuffle(cur_strd))
            last_dim = cur_dim
        self.act = ActLayer(act)
        self.out_bias = out_bias
        self.out_range = out_range
        self.depth = depth
        self.params = None

    
    def get_pos_embed(self, ):
        if 'sincos' in self.pos_embed:
            pos_embed = get_sincos_pos_embed(self.pe_dim, self.frame_num)
        else:
            pe_sigma = float(self.pos_embed.split('_')[-1])
            if 'inr' in self.pos_embed:
                w = torch.exp(torch.linspace(0, np.log(pe_sigma), self.pe_dim // 2))
            else:
                w = pe_sigma ** torch.arange(self.pe_dim // 2)    
            pe_range = [0, 1] if 'positive' in self.pos_embed else [-1, 1]
            x = torch.linspace(start=pe_range[0], end=pe_range[1], steps=self.frame_num)
            x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0))
            pos_embed = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)

        return pos_embed

    def OutImg(self, x):
        if self.out_bias == 'sigmoid':
            out = torch.sigmoid(x)
        elif self.out_bias == 'tanh':
            out = (torch.tanh(x) * 0.5) + 0.5
        else:
            out = x + float(self.out_bias)
        out = ClampTensor(out, self.out_range)
        
        return out


    def forward(self, nerv_params, batch_size):
        B, T = batch_size, self.frame_num
        if 'embed' in nerv_params:
            # x = self.params['embed']
            NotImplementedError
        else:
            x = einops.repeat(self.pe_embed, 't d-> t (b d) h w', b=B, h=1, w=1)

        for i in range(self.depth):
            x = batched_conv(x, nerv_params[f'wb{i}'], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = x.view((T, B, -1) + x.shape[-2:])
        return x

