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

import models
from models import register
from utils import split_str

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=True):
        self.device = parameters.device
        self.deterministic = deterministic
        self.mean, self.logvar = parameters[..., ::2], parameters[..., 1::2]
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        if self.deterministic:
            return self.mean
        else:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.device)
            return x

    def kl(self):
        return 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)

    def nll(self, sample, dims=[1]):
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


def init_wb(out_ch, in_ch, ks):
    weight = torch.empty(in_ch, out_ch, ks, ks) #  weight = torch.empty(in_ch, out_ch, ks, ks)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(out_ch)
    fan_out, fan_in = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    wb_list = [weight.permute(0,2,3,1).flatten(end_dim=-2), bias[None]]
    return torch.cat(wb_list, dim=0).detach()


@register('variational_diffuse_nerv')
class VDiffuseNeRV(nn.Module):
    def __init__(self, tokenizer, hyponet, token_nums, token_dims, 
        transformer_encoder,  kl_layer):
        super().__init__()
        # Build transformer backbone and video tokenizer
        self.transformer_encoder = models.DiT_models[transformer_encoder['name']](
            cond_dropout_prob=transformer_encoder['cond_dropout_prob'], 
            cond_choice=transformer_encoder['cond_choice'], 
            ada_LN=transformer_encoder['ada_LN'],
            tokenizer=tokenizer['args'],
        )
        dim = self.transformer_encoder.hid_dim
        self.tokenizer = models.make(tokenizer, args={'dim': dim})

        # build hypo_net to generate NeRV weights
        self.hyponet = models.make(hyponet)
        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng, self.postfc_dim_list = dict(), dict()
        self.ordered_hyponet_param_names = list(self.hyponet.param_shapes.keys())
        token_num_list, token_dim_list = [split_str(x) for x in [token_nums, token_dims]]
        self.moment_shape = dict()
        n_wtokens = 0
        self.num_modulate_w = {'modulate_w': 0, 'modulate_w_prefc': 0, 'modulate_w_postfc': 0 }
        for layer_i, name in enumerate(self.ordered_hyponet_param_names):
            base_w = init_wb(*self.hyponet.param_shapes[name])       # model-agnostic weights: out_ch * in_ch * ks * ks
            self.base_params[name] = nn.Parameter(base_w)
            token_num, token_dim = token_num_list[layer_i], token_dim_list[layer_i]
            if token_num and token_dim:
                self.wtoken_postfc[name] = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, token_dim * 2),   
                )
                self.wtoken_rng[name] = (n_wtokens, n_wtokens + token_num)
                self.num_modulate_w['modulate_w'] += token_num * min(dim, token_dim)
                self.num_modulate_w['modulate_w_prefc']  += token_num * dim
                self.num_modulate_w['modulate_w_postfc'] += token_num * token_dim     
                n_wtokens += token_num 
                self.moment_shape[name] = (token_num , token_dim)
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

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
            self.embed_fc = nn.Linear(dim, fc_out_dim * 2)
            self.emb_tokens = nn.Parameter(torch.randn(self.embed_tokens, dim)) 
        self.embed_type = embed_type
        self.token_num_list, self.token_dim_list = token_num_list, token_dim_list

        # build input tokens
        if 'embed_token' in self.embed_type:
            self.input_tokens = torch.cat([self.emb_tokens, self.wtokens], dim=0)
        else:
            self.input_tokens = self.wtokens

        # add kl layer
        self.deterministic = kl_layer['deterministic'].lower() in ['yes', 'true']
        if not self.deterministic:
            kl_weight_list = [float(x) for x in kl_layer['kl_weight'].split('_')]     # kl_weight for each layer
            if len(kl_weight_list)==1:
                kl_weight_list *= len(self.ordered_hyponet_param_names)
            else:
                assert len(kl_weight_list) == len(self.ordered_hyponet_param_names)
            self.kl_weight_list = kl_weight_list


    def encode(self, data, t, cond):
        # Video tokenization and go through transformer encoder
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.input_tokens, 'n d -> b n d', b=B)
        input_tokens = torch.cat([dtokens, wtokens], dim=1)
        trans_out = self.transformer_encoder(input_tokens, t, cond)
        weight_out = trans_out[:, -len(self.wtokens):]
        params_dict = dict()

        # Then, use post_fc to generate modulated weights for each NeRV layer
        for name, (l,r) in self.wtoken_rng.items():
            modulate_w = self.wtoken_postfc[name](weight_out[:, l: r, :])  # (B, token_num, total_fc_out_dim*2), where total_fc_out_dim = sum(in_ch*ks*ks*token_dim_list[i])
            params_dict[name] = modulate_w

        if 'learn' in self.embed_type:
            if 'embed_token' in self.embed_type:
                embed_out = trans_out[:,-len(self.wtokens)-self.embed_tokens:-len(self.wtokens)]
            else:
                embed_out = self.emb_tokens
            params_dict['embed'] = self.embed_fc(embed_out)      #   B, embed_tokens, embed_dim

        return {'params': params_dict}


    def rgb2nerv(self, data, t, cond,):
        n_frames = data.size(2) 
        encode_output = self.encode(data, t, cond)
        out_dict = self.sample_w(encode_output['params'])
        out_dict = self.prepare_embed(out_dict)
        return {**out_dict, 'n_frames': n_frames}


    def forward(self, data, t, cond=None,):
        out_dict = self.rgb2nerv(data, t, cond)
        hyponet_out = self.forward_hyponet(out_dict)      
        hyponet_out['pred_frames'].mul_(2).add_(-1)
        sample_frames = einops.rearrange(hyponet_out['pred_frames'], 't b c h w -> b c t h w')

        return {**out_dict, **hyponet_out, 'gt_frames': data, 'sample_frames': sample_frames}


    def sample_w(self, out_params, ):
        param_DGD = {k: DiagonalGaussianDistribution(v, self.deterministic) for k,v in out_params.items()}
        param_sample = {k:v.sample() for k,v in param_DGD.items()}
        kl_loss_dict = {k:v.kl() for k,v in param_DGD.items()}
        kl_loss = {f'kl_{k}':v.mean() for k,v in kl_loss_dict.items()}
        total_kl_loss = 0.
        if not self.deterministic:
            for idx, name in enumerate(self.ordered_hyponet_param_names):
                if name in kl_loss and kl_weight_list[idx]:
                    total_kl_loss += kl_loss[name].mean() * kl_weight_list[idx]
        out_dict = {'w_DGD': param_DGD, 'w_modulate': param_sample,
             'kl_loss': kl_loss, 'total_kl_loss': total_kl_loss}
        
        return out_dict
    

    def prepare_embed(self, out_dict):
        if 'embed' in out_dict['w_modulate']:
            final_embed = out_dict['w_modulate']['embed']
            B = final_embed.size(0)
            if 'token' in self.embed_type:                   # for embed_type: ['frame_token', 'pixel_token']
                final_embed = final_embed.view(B, self.frame_num, *self.spatial_size, -1)
            elif 'temporal_interpolate' in self.embed_type:
                final_embed = final_embed.view(B, 2, *self.spatial_size, -1)
                start_embed, end_embed = torch.chunk(final_embed, chunks=2, dim=1)              # each will have shape (b, 1, h, w, d)
                t_values = torch.linspace(0, 1, steps=self.frame_num).view(1, self.frame_num, 1, 1, 1)  # shape (1, T, 1, 1, 1)
                final_embed = start_embed + t_values.to(end_embed) * end_embed
            final_embed = einops.rearrange(final_embed, 'b t h w d -> t (b d) h w')
            out_dict['w_modulate']['embed'] = final_embed

        return out_dict
    
    
    def set_hyponet(self, modulate_params):
        hyponet_params, batch_size = {}, self.GetBatchFromMdoulateW(modulate_params)
        for name in self.ordered_hyponet_param_names:
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=batch_size)
            init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]       # (B, in_ch*ks*ks, out_ch), (B, 1, out_ch)
            if name in modulate_params:
                w = modulate_params[name]
                repeat_num = init_w.nelement()//w.nelement()
                assert repeat_num >= 1
                w = einops.repeat(w, 'B n m -> B n (d m)', d=repeat_num)
                final_w = w.view_as(init_w) * init_w
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


    def forward_hyponet(self, out_dict, pred_value_range = (0, 1)):
        modulate_params, n_frames = out_dict['w_modulate'], out_dict['n_frames']
        device = modulate_params[list(modulate_params.keys())[0]].device
        batch_size = self.GetBatchFromMdoulateW(modulate_params)
        hyponet = self.set_hyponet(modulate_params)
        coord = hyponet.get_coord(n_frames).to(device)
        coord = einops.repeat(coord, 't d -> b t d', b=batch_size)
        pred = hyponet(coord)                               # t b c h w
        pred = pred.clamp(*pred_value_range)
        return {'hyponet': hyponet, 'pred_frames': pred}


    @torch.no_grad()
    def degrade_video(self, data, head_task):
        raw_video = data.transpose(1,2).clone()   # b,t,c,h,w  
        if head_task == 'colorization':
            input_video = rgb_to_grayscale(raw_video, 3).transpose(1,2)
        elif 'sr' in head_task:
            t_scale, h_scale, w_scale = [int(x) for x in head_task.split('_')[-3:]]
            t_start, h_start, w_start = 0, 0, 0
            # t_start, h_start, w_start = [random.randint(0, x-1) for x in [t_scale, h_scale, w_scale]]
            downsample_vid = raw_video[:,t_start::t_scale,:,h_start::h_scale,w_start::w_scale]
            t, h, w = raw_video.size(1), raw_video.size(-2), raw_video.size(-1)
            downsample_vid = einops.rearrange(downsample_vid, 'b t c h w -> b c t h w')
            if 't_hw' in head_task:
                downsample_vid = F.interpolate(downsample_vid, size=(t, h//h_scale, w//w_scale), mode='trilinear', align_corners=True)
            elif 'hw_t' in head_task:
                downsample_vid = F.interpolate(downsample_vid, size=(t//t_scale, h, w), mode='trilinear', align_corners=True)
            input_video = F.interpolate(downsample_vid, size=(t, h, w), mode='trilinear', align_corners=True)
        elif 'blur' in head_task:
            blur_size, blur_sigma = [float(x) for x in head_task.split('_')[-2:]]
            blur_video = GaussianBlur(int(blur_size), blur_sigma)(raw_video.flatten(end_dim=1))
            input_video = blur_video.view_as(raw_video).transpose(1,2)
        elif 'noise' in head_task:
            noisy_m, noisy_p = [float(x) for x in head_task.split('_')[-2:]]
            noise = torch.randn_like(data) * noisy_m
            masked_noise = torch.rand_like(data) < noisy_p
            noise[~masked_noise] = 0
            input_video = data + noise.detach()
        elif 'inpaint' in head_task:
            def mask_video(mask_size, vid_data):
                clone_video = vid_data.clone()
                h, w, half_mask_size = clone_video.size(-2), clone_video.size(-1), int(mask_size // 2)
                for ctr_x, ctr_y in [(1/2, 1/2), (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)]:
                    ctr_x, ctr_y = int(ctr_x * h), int(ctr_y * w)
                    clone_video[..., ctr_x - half_mask_size: ctr_x + half_mask_size, ctr_y - half_mask_size: ctr_y + half_mask_size] = 0
                return clone_video            
            inpaint_size = float(head_task.split('_')[-1])
            input_video = mask_video(inpaint_size, data)  # b,c,t,h,w
        elif 'img2vid' in head_task:
            num_frames = raw_video.size(1)
            if self.training:
                # random pick a frame
                rand_frame_idx = random.randint(0, num_frames-1)
                select_frame = data[:,:,rand_frame_idx].clone().detach()
            else:
                # pick the first frame
                select_frame = data[:,:,0].clone().detach()
            input_video = select_frame.unsqueeze(2).repeat(1,1,num_frames,1,1)
        else:
            input_video = data

        return input_video.clamp(min=0, max=1)          # b,c,t,h,w




    # Deprecated
    def hyponet_pred_vid(self, hyponet, n_frames, batch_size, device, pred_value_range=(0, 1)):
        coord = hyponet.get_coord(n_frames).to(device)
        coord = einops.repeat(coord, 't d -> b t d', b=batch_size)
        pred = hyponet(coord)                               # t b c h w
        pred = pred.clamp(*pred_value_range)
        return pred    
    

    @staticmethod
    def ger_value(d, key, default_v):
        if key in d:
            return d[key]
        return default_v

    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f'checkpoint {ckpt} does not exist'
            ckpt = torch.load(ckpt, map_location= lambda storage, loc: storage)
        else:
            assert isinstance(ckpt, dict), f'checkpoint must be a dict or a path to a checkpoint'

        tokenizer = ckpt['model']['args']['tokenizer']
        hyponet = ckpt['model']['args']['hyponet']
        token_nums = ckpt['model']['args']['token_nums']
        token_dims = ckpt['model']['args']['token_dims']
        transformer_encoder = ckpt['model']['args']['transformer_encoder']
        model = cls(tokenizer, hyponet, token_nums, token_dims, transformer_encoder)
        if load_state_dict:
            model.load_state_dict(ckpt['model']['sd'])
        return model
