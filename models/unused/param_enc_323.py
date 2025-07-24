import os

import einops
import torch
import torch.nn as nn

import models
from models import register
from models.vit_model import ViT_models
from utils import NormLayer, DiagonalGaussianDistribution
from utils.common import get_3d_sincos_pos_embed

@register('param_enc')
class ParamEnc(nn.Module):
    def __init__(self, tokenizer, encoder, decoder, kl_layer):
        super().__init__()
        # Build transformer backbone and video tokenizer
        self.encoder = self.create_transformer(encoder)
        enc_hid_dim = self.encoder.hid_dim
        self.tokenizer = models.make(tokenizer, args={'dim': enc_hid_dim})

        # build decoder
        self.decoder = self.create_transformer(decoder)
        dec_hid_dim = self.decoder.hid_dim

        # video-agnostic tokens and patch position embedding
        # self.pe = decoder['pe'].lower()
        # self.patch_num, self.frame_num, (vh, vw), (self.ph, self.pw) = \
        self.patch_num, self.frame_num, v_hw, self.p_hw = \
            self.tokenizer.patch_num, self.tokenizer.n_frames, self.tokenizer.input_size, self.tokenizer.patch_size
        self.nh = self.nw = v_hw // self.p_hw
        pos_emb = get_3d_sincos_pos_embed(dec_hid_dim, self.nh, self.frame_num)
        self.posemb = nn.Parameter(pos_emb, requires_grad=decoder.learn_p_pe)
        # if self.pe == 'learned':
        #     self.posemb = nn.Parameter(torch.randn(self.patch_num, dec_hid_dim))
        # elif 'fix' in self.pe:
        #     self.posemb = nn.Parameter(get_3d_sincos_pos_embed(dec_hid_dim, self.nh, self.frame_num), requires_grad=False)
        # else:
        #     raise ValueError(f'Unsupported position embedding {self.pe} for input tokens')
        self.use_v_tokens = decoder['use_v_tokens']

        # layers to output video
        self.final_ch, final_ks = decoder['final_ch'], decoder['final_ks']
        self.final_fc = nn.Sequential(
            nn.LayerNorm(dec_hid_dim),
            nn.Linear(dec_hid_dim, self.ph*self.pw*self.final_ch),
            nn.GELU(),
            nn.LayerNorm(self.ph*self.pw*self.final_ch),
        )
        self.conv_out = nn.Conv3d(self.final_ch, 3, kernel_size=final_ks, padding=(final_ks-1)//2)
        self.out_bias = decoder['out_bias']

        # build bottleneck layer
        self.deterministic = kl_layer['deterministic']
        self.d_tokens = nn.Parameter(torch.randn(decoder['d_tokens'], dec_hid_dim))
        z_token_num, z_dim, z_norm = decoder['z_tokens'], decoder['z_dim'], decoder['z_norm']
        self.z_tokens = nn.Parameter(torch.randn(z_token_num, enc_hid_dim))
        proj_dim = z_dim * (1 if self.deterministic else 2)
        self.quant_fc = nn.Linear(enc_hid_dim, proj_dim)
        if self.use_v_tokens:
            z_token_num += 1
        self.z_norm_first = decoder['z_norm_first']
        self.z_norm_layer = NormLayer(z_norm, proj_dim if self.z_norm_first else z_dim, z_token_num)
        self.dequant_fc = nn.Linear(z_dim, dec_hid_dim)
        bottleneck_size = z_token_num*z_dim
        self.bottleneck_str = f'(token_num: {z_token_num}, token_dim: {z_dim}), norm {z_norm} at {round(bottleneck_size/1e3, 1)}k'
        self.num_modulate_w = {'modulate_w': bottleneck_size, 'bottleneck_w': bottleneck_size, 
            'modulate_w_prefc': z_token_num*enc_hid_dim, 'modulate_w_postfc': z_token_num*dec_hid_dim}

        # add kl layer
        if not self.deterministic:
            self.kl_weight = kl_layer['kl_weight']
            self.avg_kl = kl_layer['avg_kl']


    def create_transformer(self, transformer_cfg):
        model = ViT_models[transformer_cfg['name']](
            proj_drop = transformer_cfg['proj_drop'],
            attn_drop = transformer_cfg['attn_drop'],
            mlp_drop = transformer_cfg['mlp_drop'],
        )      
        return model  


    def forward(self, data, **kwargs):
        z_info = self.encode(data, **kwargs)
        vid_pred = self.decode(z_info['z'])
        out_frames_dict = {'pred': vid_pred}

        return {**z_info, 'out_frames': out_frames_dict, 'gt_frames': data, 'pred_frames': vid_pred}


    def bottle_forward(self, z):
        z = self.quant_fc(z)
        if self.z_norm_first:
            z = self.z_norm_layer(z)
        z_posterior = DiagonalGaussianDistribution(z, self.deterministic)
        z = z_posterior.sample()
        if not self.z_norm_first:
            z = self.z_norm_layer(z)
        base_kl = z_posterior.kl()
        if not self.deterministic:
            base_kl = base_kl.sum() / base_kl.size(0)
        z_info = {'z': z, 'base_kl': base_kl, 'kl_loss': base_kl * self.kl_weight}

        return z_info


    def encode(self, data, **kwargs):
        # Video tokenization and go through transformer encoder
        dtokens = self.tokenizer(data)
        z_tokens = self.repeat_input_tokens(self.z_tokens, dtokens.size(0))
        trans_out = self.encoder([dtokens, z_tokens])
        v_tokens, z_tokens = trans_out.split([dtokens.size(1), z_tokens.size(1)], dim=1)
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        if self.use_v_tokens:
            v_ztokens = v_tokens.mean(1, keepdim=True)
            z_tokens = torch.cat([v_ztokens, z_tokens], dim=1)
        z_info = self.bottle_forward(z_tokens)

        return z_info


    def prepare_tokens(self, z):
        if self.use_v_tokens:
            pos_emb = z[:,:1] 
            z_tokens = z[:,1:] 
        else:
            pos_emb = torch.zeros_like(z[:,:1], requires_grad=False)
            z_tokens = z
        patch_tokens = pos_emb + self.posemb

        return torch.cat([patch_tokens, z_tokens], dim=1)


    def decode(self, z):
        z = self.dequant_fc(z)
        in_tokens = self.prepare_tokens(z)
        d_tokens = self.repeat_input_tokens(self.d_tokens, in_tokens.size(0))
        trans_out = self.decoder([in_tokens, d_tokens])
        patch_tokens = self.final_fc(trans_out[:,:self.patch_num])
        vid_pred = einops.rearrange(patch_tokens, 'b (t nh nw) (c ph pw) -> b c t (nh ph) (nw pw)', t=self.frame_num, 
            nh=self.nh, nw=self.nw, c=self.final_ch, ph=self.ph, pw=self.pw)
        # import pdb; pdb.set_trace; from IPython import embed; embed()     
        vid_pred = self.conv_out(vid_pred) + self.out_bias
        vid_pred = einops.rearrange(vid_pred, 'b c t h w -> t b c h w')
        vid_pred = vid_pred.clamp(min=0, max=1)

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
