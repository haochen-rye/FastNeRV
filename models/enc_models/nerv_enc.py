import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.transformers.vit_model import ViT_models
from utils import NormLayer, split_str, init_wb, DiagonalGaussianDistribution

@register('nerv_enc')
class NeRVEnc(nn.Module):
    def __init__(self, tokenizer, hyponet, nerv_tokens, transformer_encoder, kl_layer):
        super().__init__()
        # Build transformer backbone and video tokenizer
        self.transformer_encoder = self.create_transformer(transformer_encoder)
        hid_dim = self.transformer_encoder.hid_dim
        self.tokenizer = models.make(tokenizer, args={'dim': hid_dim})

        # build bottleneck layers
        self.z_dim = nerv_tokens['z_dim']
        self.z_type = nerv_tokens['z_type'].lower()
        assert self.z_type in ['patch', 'nerv', 'extra', 'no', 'prefc']
        if self.z_type in ['no', 'prefc']:
            self.quant_z_fc = nn.Linear(hid_dim, hid_dim*2) if self.z_type=='prefc' else nn.Identity()
            self.dequant_z_fc = nn.Identity()
        else:
            self.quant_z_fc = nn.Linear(hid_dim, self.z_dim*2)
            self.dequant_z_fc = nn.Linear(self.z_dim, hid_dim)

        # build hypo_net to generate NeRV weights
        self.hyponet = models.make(hyponet)
        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.repeat_dim = nerv_tokens['repeat_dim'] if 'repeat_dim' in nerv_tokens else -2
        self.wtoken_rng, self.postfc_dim_list = dict(), dict()
        self.ordered_hyponet_param_names = list(self.hyponet.param_shapes.keys())
        token_num_list, token_dim_list = [split_str(nerv_tokens[x]) for x in ['token_nums', 'token_dims']]
        self.moment_shape = dict()
        n_wtokens = 0
        self.num_modulate_w = {'modulate_w': 0, 'modulate_w_prefc': 0, 'modulate_w_postfc': 0 }
        for layer_i, name in enumerate(self.ordered_hyponet_param_names):
            base_w = init_wb(*self.hyponet.param_shapes[name])       # model-agnostic weights: out_ch * in_ch * ks * ks
            self.base_params[name] = nn.Parameter(base_w)
            token_num, token_dim = token_num_list[layer_i], token_dim_list[layer_i]
            if token_num and token_dim:
                self.wtoken_postfc[name] = nn.Sequential(
                    nn.LayerNorm(hid_dim),
                    nn.Linear(hid_dim, token_dim),   
                )
                self.wtoken_rng[name] = (n_wtokens, n_wtokens + token_num)
                self.num_modulate_w['modulate_w'] += token_num * min(hid_dim, token_dim)
                self.num_modulate_w['modulate_w_prefc']  += token_num * hid_dim
                self.num_modulate_w['modulate_w_postfc'] += token_num * token_dim     
                n_wtokens += token_num 
                self.moment_shape[name] = (token_num , token_dim)
        self.wtokens = nn.Parameter(torch.empty(n_wtokens, hid_dim))
        self.token_init_std = nerv_tokens['token_init_std']
        self.token_num_list, self.token_dim_list = token_num_list, token_dim_list

        # collect bottleneck info for logging
        z_token_num = n_wtokens
        if self.z_type in ['prefc', 'no']:
            self.z_dim = hid_dim
        z_size = z_token_num * self.z_dim 
        self.z_token_num = z_token_num
        self.num_modulate_w['bottleneck_w'] = z_size
        self.bottleneck_str = f'{self.z_type}: (token_num: {z_token_num}, token_dim: {self.z_dim}) at {round(z_size/1e3, 1)}k'
        self.z_norm_first = nerv_tokens['z_norm_first']
        self.z_norm_layer = NormLayer(nerv_tokens['z_norm'], self.z_dim*(2 if self.z_norm_first else 1), z_token_num)
        self.frame_num = tokenizer['args']['frame_num']

        # add kl layer
        self.deterministic = kl_layer['deterministic']
        self.kl_weight = float(kl_layer['kl_weight'])
        self.avg_kl = kl_layer['avg_kl']
        self.initialize_weights()


    def initialize_weights(self):    
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize wtokens
        nn.init.normal_(self.wtokens, std=self.token_init_std)


    def create_transformer(self, transformer_encoder):
        model = ViT_models[transformer_encoder['name']](
            proj_drop = transformer_encoder['proj_drop'],
            attn_drop = transformer_encoder['attn_drop'],
            mlp_drop = transformer_encoder['mlp_drop'],
        )      
        return model  


    def forward(self, data, **kwargs):
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        z_info = self.encode(data, **kwargs)
        out_dict = self.decode(z_info['z'])
        out_frames_dict = {'pred': out_dict['pred_frames']}

        return {**out_dict, **z_info, 'out_frames': out_frames_dict, 
                 'gt_frames': data}


    def encode(self, data, **kwargs):
        # Video tokenization and go through transformer encoder
        dtokens = self.tokenizer(data)
        wtokens = self.repeat_input_tokens(self.wtokens, dtokens.size(0))
        trans_out = self.transformer_encoder([dtokens, wtokens], **kwargs)
        z_tokens = self.quant_z_fc(trans_out[:, -len(self.wtokens):])

        if self.z_type in ['no']:
            return {'z': z_tokens}
            #  z = z_tokens
        else:
            if self.z_norm_first:
                z_tokens = self.z_norm_layer(z_tokens)
            z_posterior = DiagonalGaussianDistribution(z_tokens, self.deterministic)
            z = z_posterior.sample()      
            base_kl = z_posterior.kl(avg_kl=self.avg_kl)
            if not self.z_norm_first:
                z = self.z_norm_layer(z)
            return {'z': z, 'kl_loss': base_kl*self.kl_weight, 'base_kl': base_kl}
        # import pdb; pdb.set_trace; from IPython import embed; embed() 


    def decode(self, z):
        z = self.dequant_z_fc(z)
        params_dict = dict()
        for name, (l,r) in self.wtoken_rng.items():
            modulate_w = self.wtoken_postfc[name](z[:, l: r, :])        # (B, token_num, total_fc_out_dim*2), where total_fc_out_dim = sum(in_ch*ks*ks*token_dim_list[i])
            params_dict[name] = modulate_w
        pred_frames = self.forward_hyponet(params_dict, z.size(0))      

        return {'params': params_dict, 'pred_frames': pred_frames}


    def forward_hyponet(self, modulate_params, batch_size):
        hyponet_params = {}
        for name in self.ordered_hyponet_param_names:
            wb = self.repeat_input_tokens(self.base_params[name], batch_size)
            # einops.repeat(self.base_params[name], 'n m -> b n m', b=batch_size)
            init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]       # (B, in_ch*ks*ks, out_ch), (B, 1, out_ch)
            if name in modulate_params:
                w = modulate_params[name]
                repeat_num = init_w.nelement() // w.nelement()
                assert repeat_num >= 1
                if self.repeat_dim == -1:
                    w = einops.repeat(w, 'B n m -> B n m d', d=repeat_num)
                elif self.repeat_dim == -2:
                    w = einops.repeat(w, 'B n m -> B n d m', d=repeat_num)
                elif self.repeat_dim == -3:
                    w = einops.repeat(w, 'B n m -> B d n m', d=repeat_num)
                else:
                    NotImplementedError(f'Unsupported repeat_dim {self.repeat_dim}')
                # import pdb; pdb.set_trace; from IPython import embed; embed()   
                final_w = w.reshape_as(init_w) * init_w
            else:
                final_w = init_w
            final_w = F.normalize(final_w, dim=1)
            hyponet_params[name] = torch.cat([final_w, init_b], dim=1)

        pred = self.hyponet(hyponet_params, batch_size)     # T B C H W

        return pred
        

    def repeat_input_tokens(self, in_tokens, batch_size):
        # repeat batch_size at the begining of dimension
        return einops.repeat(in_tokens, 'n d -> b n d', b=batch_size)


    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f'checkpoint {ckpt} does not exist'
            ckpt = torch.load(ckpt, map_location= lambda storage, loc: storage)
        else:
            assert isinstance(ckpt, dict), f'checkpoint must be a dict or a path to a checkpoint'
        model = cls(**ckpt['model']['args'])
        if load_state_dict:
            model.load_state_dict(ckpt['model']['sd'])
        return model



    # deprecated
