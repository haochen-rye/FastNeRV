import torch
import torch.nn as nn
import numpy as np
import einops
from math import ceil

from models import register
from .layers import batched_conv
from utils import make_coord_grid, split_str, get_sincos_pos_embed, ActLayer

@register('hypo_convnets')
class HypoConvnets(nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim, 
        strds, ks, pos_embed, pe_dim, act='relu', out_bias=0.5, ):
        super().__init__()
        if 'fix' in pos_embed:
            self.pe_sigma = float(pos_embed.split('_')[-1])
            self.pe_range = [0, 1] if 'positive' in pos_embed else [-1, 1]
        # self.pos_embed, self.pe_dim = pos_embed, pe_dim
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

        def ActLayer(act):
            if act == 'relu':
                return nn.ReLU()
            elif act == 'gelu':
                return nn.GELU()
            else:
                NotImplementedError
        self.act = ActLayer(act)
        self.params = None
        self.out_bias = out_bias
        self.depth = depth


    def set_params(self, params):
        self.params = params


    def old_convert_posenc(self, x):
        if 'inr' in self.pos_embed:
            w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        elif 'nerv' in self.pos_embed:
            w = self.pe_sigma ** torch.arange(self.pe_dim // 2, device=x.device)
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
      
        return x


    def convert_posenc(self, frames):
        # import pdb; pdb.set_trace; from IPython import embed; embed()    
        return get_sincos_pos_embed(self.pe_dim, frames)       # (T D)
        
    def get_coord(self, n_frames):
        return make_coord_grid([n_frames], self.pe_range)
    
    def OutImg(self, x):
        if self.out_bias == 'sigmoid':
            return torch.sigmoid(x)
        elif self.out_bias == 'tanh':
            return (torch.tanh(x) * 0.5) + 0.5
        else:
            return x + float(self.out_bias)


    def forward(self, x):
        B, T = x.size(0), x.size(1)
        if 'embed' in self.params:
            x = self.params['embed']
        else:
            x = self.old_convert_posenc(x.view(B, -1, x.shape[-1])) 
            x = x.permute(1,0,2).reshape(T, -1, 1, 1)    # (T, B*PE_DIM, 1, 1)
            # pe = self.convert_posenc(T).to(x)
            # x = einops.repeat(pe, 't d->t (b d) h w', b=B, h=1, w=1)

        for i in range(self.depth):
            x = batched_conv(x, self.params[f'wb{i}'], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = x.view((T, B, -1) + x.shape[-2:])
        return x
