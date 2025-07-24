import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.vit_model import ViT_models
# from models.unused.nerv_dit import NeRV_ViT as ViT_models
from utils import MLP, NormLayer, split_str, init_wb, DiagonalGaussianDistribution


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
            self.quant_z_mlp = nn.Linear(hid_dim, hid_dim*2) if self.z_type=='prefc' else nn.Identity()
            self.dequant_z_mlp = nn.Identity()
        else:
            self.quant_z_mlp = nn.Linear(hid_dim, self.z_dim*2)
            mlp_expand_ratio = int(nerv_tokens['mlp_expand_ratio'])
            dequant_dim_list = [self.z_dim, self.z_dim*mlp_expand_ratio, ]
            if self.z_type in ['nerv']:
                self.dequant_z_mlp = MLP(dequant_dim_list + [hid_dim])
            else:
                self.btk_head = ViT_models[nerv_tokens['btk_head']]()
                hid_dim2 = self.btk_head.hid_dim
                self.dequant_z_mlp = MLP(dequant_dim_list + [hid_dim2])
                hid_dim = hid_dim2
                if self.z_type in ['extra']:
                    self.extra_z_num = nerv_tokens['extra_z_tokens']
                    assert self.extra_z_num > 0
                    self.extra_z_tokens = nn.Parameter(torch.randn(self.extra_z_num, hid_dim))

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
        nn.init.normal_(self.wtokens)

        # collect bottleneck info for logging
        if self.z_type in ['nerv', 'no', 'prefc']:
            z_token_num = n_wtokens
        elif self.z_type in ['patch']:
            z_token_num = self.tokenizer.patch_num
        elif self.z_type in ['extra']:
            z_token_num = self.extra_z_num
        if self.z_type in ['prefc']:
            self.z_dim = hid_dim
        z_size = z_token_num * self.z_dim 
        self.z_token_num = z_token_num
        self.num_modulate_w['bottleneck_w'] = z_size
        self.bottleneck_str = f'{self.z_type}: (token_num: {z_token_num}, token_dim: {self.z_dim}) at {round(z_size/1e3, 1)}k'
        # self.z_norm_layer = NormLayer(nerv_tokens['z_norm'], hid_dim, z_token_num)
        self.z_norm_layer = NormLayer(nerv_tokens['z_norm'], self.z_dim, z_token_num)

        # learning input embedding for each video
        self.frame_num = tokenizer['args']['frame_num']
        embed_type = hyponet['args']['pos_embed'].lower()
        if 'learn' in embed_type:
            spatial_size = [int(x) for x in embed_type.split('_')[1:3]] 
            if 'temporal_interpolate' in embed_type:
                embed_tokens = spatial_size + [2]
                fc_out_dim = self.hyponet.pe_dim
            elif 'frame_token' in embed_type:
                embed_tokens = [1,1,self.frame_num]
                fc_out_dim = self.hyponet.pe_dim * np.prod(spatial_size)
            elif 'pixel_token' in embed_type:
                embed_tokens = spatial_size + [self.frame_num]
                fc_out_dim = self.hyponet.pe_dim 
            self.embed_tokens = np.prod(np.array(embed_tokens))
            self.hyponet.param_shapes['embed'] = (self.embed_tokens, [-1])
            self.spatial_size = tuple(spatial_size)
            self.embed_fc = nn.Linear(hid_dim, fc_out_dim * 2)
            self.emb_tokens = nn.Parameter(torch.randn(self.embed_tokens, hid_dim)) 
        self.embed_type = embed_type
        self.token_num_list, self.token_dim_list = token_num_list, token_dim_list

        # build input tokens
        if 'embed_token' in self.embed_type:
            NotImplementedError
            # self.input_tokens = torch.cat([self.emb_tokens, self.wtokens], dim=0)
        else:
            self.input_tokens = self.wtokens

        # add kl layer
        self.deterministic = kl_layer['deterministic']
        self.kl_weight = kl_layer['kl_weight']
        self.avg_kl = kl_layer['avg_kl']

        # degration task
        # self.head_task = transformer_encoder['cond_task']

    def create_transformer(self, transformer_encoder):
        model = ViT_models[transformer_encoder['name']](
            proj_drop = transformer_encoder['proj_drop'],
            attn_drop = transformer_encoder['attn_drop'],
            mlp_drop = transformer_encoder['mlp_drop'],
        )      
        return model  


    def forward(self, data, **kwargs):
        n_frames = data.size(2) 
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        z_info = self.encode(data, **kwargs)
        out_dict = self.decode(z_info['z'], n_frames)
        out_frames_dict = {'pred': out_dict['pred_frames']}

        return {**out_dict, **z_info, 'out_frames': out_frames_dict, 
                 'gt_frames': data}


    def encode(self, data, **kwargs):
        # Video tokenization and go through transformer encoder
        dtokens = self.tokenizer(data)
        if self.z_type in ['nerv', 'no', 'prefc']:
            wtokens = self.repeat_input_tokens(self.input_tokens, dtokens.size(0))
            trans_out = self.transformer_encoder([dtokens, wtokens], **kwargs)
            z_tokens = self.quant_z_mlp(trans_out[:, -len(self.wtokens):])
        elif self.z_type in ['patch']:            
            z_tokens = self.quant_z_mlp(self.transformer_encoder(dtokens, **kwargs))
        elif self.z_type in ['extra']:            
            extra_z_tokens = self.repeat_input_tokens(self.extra_z_tokens, dtokens.size(0))
            trans_out = self.transformer_encoder([dtokens, extra_z_tokens], **kwargs)
            z_tokens = self.quant_z_mlp(trans_out[:,-self.extra_z_num:])
        else:
            NotImplementedError

        if self.z_type in ['no']:
             z, base_kl = z_tokens, 0
        else:
            z_tokens = self.z_norm_layer(z_tokens)
            z_posterior = DiagonalGaussianDistribution(z_tokens, self.deterministic)
            z = z_posterior.sample()      
            base_kl = self.get_kl_loss(z_posterior)
            # z = self.z_norm_layer(z_posterior.sample())      # Normalize the bottleneck_z            
        
        return {'z': z, 'kl_loss': base_kl*self.kl_weight, 'base_kl': base_kl}


    def get_kl_loss(self, z_posterior):
        if self.deterministic:
            return 0
        else:
            kl_loss = z_posterior.kl()
            if self.avg_kl:
                return kl_loss.mean()
            else:
                return kl_loss.sum() / kl_loss.size(0)
            # import pdb; pdb.set_trace; from IPython import embed; embed()     


    def decode(self, z, n_frames):
        z = self.dequant_z_mlp(z)
        if self.z_type in ['patch', 'extra']:            
            wtokens = self.repeat_input_tokens(self.input_tokens, z.size(0))
            if self.z_type in ['patch']:            
                trans_out = self.btk_head([z, wtokens])
            else:
                trans_out = self.btk_head([z, wtokens])
            z = trans_out[:, -len(self.wtokens):]

        # Then, use post_fc to generate modulated weights for each NeRV layer
        params_dict = dict()
        for name, (l,r) in self.wtoken_rng.items():
            modulate_w = self.wtoken_postfc[name](z[:, l: r, :])  # (B, token_num, total_fc_out_dim*2), where total_fc_out_dim = sum(in_ch*ks*ks*token_dim_list[i])
            params_dict[name] = modulate_w

        if 'learn' in self.embed_type:
            if 'embed_token' in self.embed_type:
                NotImplementedError
            else:
                embed_out = self.emb_tokens
            params_dict['embed'] = self.embed_fc(embed_out)      #   B, embed_tokens, embed_dim

        hyponet_out = self.forward_hyponet(params_dict, n_frames)      

        return {'params': params_dict, **hyponet_out}


    def repeat_input_tokens(self, in_tokens, batch_size):
        return einops.repeat(in_tokens, 'n d -> b n d', b=batch_size)


    def set_hyponet(self, modulate_params):
        hyponet_params, batch_size = {}, self.GetBatchFromMdoulateW(modulate_params)
        for name in self.ordered_hyponet_param_names:
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=batch_size)
            init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]       # (B, in_ch*ks*ks, out_ch), (B, 1, out_ch)
            if name in modulate_params:
                w = modulate_params[name]
                repeat_num = init_w.nelement() // w.nelement()
                # if repeat_num < 1:
                #     import pdb; pdb.set_trace; from IPython import embed; embed()    
                assert repeat_num >= 1
                if self.repeat_dim == -1:
                    w = einops.repeat(w, 'B n m -> B n m d', d=repeat_num)
                elif self.repeat_dim == -2:
                    w = einops.repeat(w, 'B n m -> B n d m', d=repeat_num)
                elif self.repeat_dim == -3:
                    w = einops.repeat(w, 'B n m -> B d n m', d=repeat_num)
                else:
                    NotImplementedError(f'Unsupported repeat_dim {self.repeat_dim}')
                final_w = w.reshape_as(init_w) * init_w
            else:
                final_w = init_w
            final_w = F.normalize(final_w, dim=1)
            hyponet_params[name] = torch.cat([final_w, init_b], dim=1)

        if 'embed' in modulate_params:
            hyponet_params['embed'] = modulate_params['embed']
        self.hyponet.set_params(hyponet_params)             # (B, in_ch, ks, ks, out_ch)
        
        return self.hyponet 


    @staticmethod
    def GetBatchFromMdoulateW(modulate_params):
        return modulate_params[list(modulate_params.keys())[0]].size(0)           


    def forward_hyponet(self, modulate_params, n_frames, pred_value_range = (0, 1)):
        # modulate_params, n_frames = out_dict['w_modulate'], out_dict['n_frames']
        device = modulate_params[list(modulate_params.keys())[0]].device
        batch_size = self.GetBatchFromMdoulateW(modulate_params)
        hyponet = self.set_hyponet(modulate_params)
        coord = hyponet.get_coord(n_frames).to(device)
        coord = einops.repeat(coord, 't d -> b t d', b=batch_size)
        pred = hyponet(coord)                               # t b c h w
        pred = pred.clamp(*pred_value_range)
        return {'hyponet': hyponet, 'pred_frames': pred}


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

