import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
from models import register
from models.transformers.vit_model import ViT_models
from utils import split_str, VQEmbedding
from utils import NormLayer, DiagonalGaussianDistribution
from utils.common import get_sincos_pos_embed, ClampTensor

@register('param_enc')
class ParamEnc(nn.Module):
    def __init__(self, tokenizer, encoder, decoder, kl_layer, vq_layer):
        super().__init__()
        # Build transformer backbone and video tokenizer
        self.encoder = self.create_transformer(encoder)
        enc_hid_dim = self.encoder.hid_dim
        self.tokenizer = models.make(tokenizer, args={'dim': enc_hid_dim})

        # build decoder
        self.decoder = self.create_transformer(decoder)
        dec_hid_dim = self.decoder.hid_dim
        self.imgwise_decode = decoder['imgwise_decode']
        self.expand_decoder = decoder['expand_decoder']

        self.patch_num, self.frame_num, v_hw, self.p_t, self.p_hw = \
            self.tokenizer.patch_num, self.tokenizer.frame_num, self.tokenizer.input_size, self.tokenizer.patch_t, self.tokenizer.patch_hw
        self.nh = self.nw = v_hw // self.p_hw
        self.nt = self.frame_num // self.p_t
        p_posemb = get_sincos_pos_embed(dec_hid_dim, (self.nt, self.nh, self.nw))
        self.p_posemb = nn.Parameter(p_posemb.view(-1, dec_hid_dim), requires_grad=decoder['learn_p_pe'])
        self.v_token_list = split_str(decoder['use_v_tokens'])
        v_token_num = np.prod(self.v_token_list)
        v_dim = decoder['v_dim']

        # layers to output video
        self.final_ch, final_ks = decoder['final_ch'], decoder['final_ks']
        out_ch = self.final_ch * self.p_t * self.p_hw**2
        self.final_fc = nn.Sequential(
            nn.LayerNorm(dec_hid_dim),
            nn.Linear(dec_hid_dim, out_ch),
            nn.GELU(),
            nn.LayerNorm(out_ch),
        )
        self.conv_out = nn.Conv3d(self.final_ch, 3, kernel_size=final_ks, padding=(final_ks-1)//2)
        self.out_bias = decoder['out_bias']
        self.out_range = decoder['out_range']

        # bottleneck 
        self.d_tokens = nn.Parameter(torch.randn(decoder['d_tokens'], dec_hid_dim))
        self.token_init_std = decoder['token_init_std']
        self.z_bottleneck = decoder['z_bottleneck']
        self.deterministic = kl_layer['deterministic']
        self.use_vq = vq_layer['use_vq']   
        self.z_norm_first = decoder['z_norm_first']

        # extra z_tokens
        z_token_num, z_dim, z_norm = decoder['z_tokens'], decoder['z_dim'], decoder['z_norm'] 
        self.z_tokens = nn.Parameter(torch.randn(z_token_num, enc_hid_dim))
        z_posemb = get_sincos_pos_embed(dec_hid_dim, z_token_num)
        self.z_posemb = nn.Parameter(z_posemb, requires_grad=decoder['learn_z_pe'])

        def quant_layers(z_dim, enc_dim=enc_hid_dim, dec_dim=dec_hid_dim):
            if self.z_bottleneck:
                proj_dim = z_dim * (1 if self.deterministic else 2)
                quant_layer = nn.Linear(enc_dim, proj_dim)
                dequant_layer = nn.Linear(z_dim, dec_dim)
                norm_layer = NormLayer(z_norm, proj_dim if self.z_norm_first else z_dim, z_token_num)
            else:
                quant_layer, dequant_layer, norm_layer = nn.Linear(enc_dim, dec_dim), nn.Identity(), nn.Identity()
            return quant_layer, dequant_layer, norm_layer
        self.z_quant_fc, self.z_dequant_fc, self.z_norm = quant_layers(z_dim)
        self.v_quant_fc, self.v_dequant_fc, self.v_norm = quant_layers(v_dim)

        # vq_layer
        if self.use_vq:
            assert z_dim % v_dim ==0, f'z_dim {z_dim} mush be divided by v_dim {v_dim}'
            self.z_codebook = VQEmbedding(vq_layer['codebook_size'],
                                        v_dim, 
                                        using_znorm = vq_layer['using_znorm'],
                                        ema = vq_layer['ema'],
                                        decay = vq_layer['ema_decay'], 
                                        )
            self.vq_weight = vq_layer['vq_weight']
            self.commit_loss = vq_layer['commit_loss']
            self.codebook_loss = vq_layer['codebook_loss']
            self.ema_type = vq_layer['ema_type']

        # kl layer
        if not self.deterministic:
            self.kl_weight = kl_layer['kl_weight']
            self.avg_kl = kl_layer['avg_kl']
            
        # bottleneck info for log
        bottleneck_size = z_token_num * z_dim + v_token_num * v_dim
        self.bottleneck_str = f'z: ({z_token_num}, {z_dim}), v: ({v_token_num}, {v_dim}), norm {z_norm} at {round(bottleneck_size/1e3, 1)}k'
        self.num_modulate_w = {'modulate_w': bottleneck_size, 'bottleneck_w': bottleneck_size, 
            'modulate_w_prefc': z_token_num*enc_hid_dim, 'modulate_w_postfc': z_token_num*dec_hid_dim}
        self.z_token_num, self.z_dim = z_token_num, z_dim
        self.v_token_num, self.v_dim = v_token_num, v_dim

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
        for init_token in [self.z_tokens, self.d_tokens]:
            nn.init.normal_(init_token, std=self.token_init_std)

        # Zero-out output layers:
        nn.init.constant_(self.conv_out.weight, 0)
        nn.init.constant_(self.conv_out.bias, 0)


    def create_transformer(self, transformer_cfg):
        model = ViT_models[transformer_cfg['name']](
            proj_drop = transformer_cfg['proj_drop'],
            attn_drop = transformer_cfg['attn_drop'],
            mlp_drop = transformer_cfg['mlp_drop'],
        )      
        return model  


    def forward(self, data, **kwargs):
        z_info = self.encode(data, **kwargs)
        vid_pred = self.decode(z_info)
        pred_list = vid_pred.split(data.size(0), dim=1)
        out_frames_dict = {f'pred{k}': v for k,v in enumerate(pred_list)}

        return {**z_info, 'out_frames': out_frames_dict, 'gt_frames': data, 'pred_frames': pred_list[-1]}


    def bottleneck_forward(self, z, z_type='z'):
        if z_type == 'z':
            quant_fc, norm_layer = self.z_quant_fc, self.z_norm
        elif z_type == 'v':
            quant_fc, norm_layer = self.v_quant_fc, self.v_norm
        else:
            raise ValueError(f'bottleneck forward only support [z,v], {z_type} found')
        
        z_info = {}
        z = quant_fc(z)
        if self.z_norm_first:
            z = norm_layer(z)
        
        # kl for regularization
        if not self.deterministic:
            z_posterior = DiagonalGaussianDistribution(z, self.deterministic)
            z = z_posterior.sample()
            base_kl = z_posterior.kl(avg_kl=self.avg_kl)
            z_info.update({f'{z_type}_base_kl': base_kl, f'{z_type}_kl_loss': base_kl * self.kl_weight})
        
        # vq for regularization
        if self.use_vq and z.numel() > 0:
            z_orig = z.clone()
            z_quant, z_code = self.z_codebook(z_orig.contiguous(), update_ema=(z_type in self.ema_type))
            commit_loss = F.mse_loss(z_quant.detach(), z_orig) 
            codebook_loss = F.mse_loss(z_quant, z_orig.detach())
            vq_loss = self.vq_weight * (self.commit_loss * commit_loss + self.codebook_loss * codebook_loss)
            z = (z_quant - z_orig).detach() + z_orig
            z_info.update({
                f'{z_type}_code': z_code, f'{z_type}_commit_loss': commit_loss, 
                f'{z_type}_codebook_loss': codebook_loss, f'{z_type}_vq_loss': vq_loss, 
            })            
     
        if not self.z_norm_first:
            z = norm_layer(z)

        return {**z_info, z_type: z}


    def pool_and_expand(self, v_tokens):
        # avg pool patch tokens and expand back to the original shape
        if self.v_token_num:
            tn, hn, wn = self.tokenizer.num_t, self.tokenizer.num_h, self.tokenizer.num_w
            ts, hs, ws = self.v_token_list
            pool_t, pool_h, pool_w = tn//ts, hn//hs, wn//ws
            v_ztokens = einops.rearrange(v_tokens, 'b (tp ts hp hs wp ws) c -> b ts hs ws (tp hp wp) c',
                ts=ts, hs=hs, ws=ws, tp=pool_t, hp=pool_h, wp=pool_w,) 
            v_ztokens = v_ztokens.mean(-2, keepdim=False)
            v_ztokens = einops.repeat(v_ztokens, 'b ts hs ws c -> b (tp ts) (hp hs) (wp ws) c', 
                tp=pool_t, hp=pool_h, wp=pool_w)
            v_ztokens = einops.rearrange(v_ztokens, 'b t h w c -> b (t h w) c')
            return v_ztokens
        else:
            return v_tokens
        

    def encode(self, data, **kwargs):
        # Video tokenization and go through transformer encoder
        dtokens = self.tokenizer(data)
        z_tokens = self.repeat_input_tokens(self.z_tokens, dtokens.size(0))
        trans_out = self.encoder([dtokens, z_tokens])
        v_tokens, z_tokens = trans_out.split([dtokens.size(1), z_tokens.size(1)], dim=1)
        # prepare patch tokens for bottleneck layers
        v_tokens = self.pool_and_expand(v_tokens)        
        v_info, z_info = self.bottleneck_forward(v_tokens, z_type='v'), self.bottleneck_forward(z_tokens, z_type='z')

        return {**v_info, **z_info}


    def prepare_tokens(self, z_info):
        # extra video-specific z_tokens
        z_tokens = self.z_dequant_fc(z_info['z'])
        B = z_tokens.size(0)

        # patch_tokens
        if 'v' in z_info:
            patch_tokens = z_info['v']
            patch_tokens = self.v_dequant_fc(patch_tokens)
            if self.imgwise_decode:
                patch_tokens = einops.rearrange(patch_tokens, 'b (t h w) c -> (b t) (h w) c', t=self.nt, h=self.nh, w=self.nw)
                z_tokens = einops.repeat(z_tokens, 'b l c -> (b t) l c', t=self.nt)
        else:
            patch_tokens = 0

        z_tokens = z_tokens + self.z_posemb[None]
        patch_tokens += self.repeat_input_tokens(self.p_posemb, B).reshape_as(patch_tokens)
        
        # extra video-agnostic tokens
        d_tokens = self.repeat_input_tokens(self.d_tokens, z_tokens.size(0))

        return [patch_tokens, z_tokens, d_tokens]


    def decode(self, z):
        in_tokens = self.prepare_tokens(z)
        def get_dec_kargs():
            return {
                'z_token_list': None if self.expand_decoder else getattr(self, 'z_token_list', None),
                'sequential_decode': getattr(self, 'sequential_decode', True),
                'cumsum_z': getattr(self, 'cumsum_z', True),
            }
        decoder_args = get_dec_kargs()
        trans_out = self.decoder(in_tokens, **decoder_args)
        patch_tokens = self.final_fc(trans_out[:,:self.patch_num])
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        patch_shape = '(b nt) (nh nw)' if self.imgwise_decode else 'b (nt nh nw)'
        vid_pred = einops.rearrange(patch_tokens, f'{patch_shape} (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)',
            nt=self.nt, nh=self.nh, nw=self.nw, c=self.final_ch, pt=self.p_t, ph=self.p_hw, pw=self.p_hw)
        # if self.imgwise_decode: 
        #     vid_pred = einops.rearrange(patch_tokens, '(b nt) nh nw) (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)',
        #         nt=self.nt, nh=self.nh, nw=self.nw, c=self.final_ch, pt=self.p_t, ph=self.p_hw, pw=self.p_hw)
        # else:
        #     vid_pred = einops.rearrange(patch_tokens, 'b (nt nh nw) (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)',
        #         nt=self.nt, nh=self.nh, nw=self.nw, c=self.final_ch, pt=self.p_t, ph=self.p_hw, pw=self.p_hw)
        vid_pred = self.conv_out(vid_pred) + self.out_bias
        vid_pred = einops.rearrange(vid_pred, 'b c t h w -> t b c h w')
        vid_pred = ClampTensor(vid_pred, self.out_range)
        # import pdb; pdb.set_trace; from IPython import embed; embed()     

        return vid_pred


    def repeat_input_tokens(self, in_tokens, batch_size):
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
