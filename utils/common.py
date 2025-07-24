import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import rgb_to_grayscale
import torch.distributed as dist


# Quantization layer
class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""
    def __init__(self, n_embed, embed_dim, using_znorm=True, ema=True, decay=0.99, eps=1e-5):
        super().__init__(n_embed + (1 if ema else 0), embed_dim)
        self.using_znorm = using_znorm
        self.ema, self.decay, self.eps = ema, decay, eps
        self.n_embed, self.embed_dim = n_embed, embed_dim

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

        # Initialize usage tracker
        self.register_buffer('usage_tracker', torch.zeros(n_embed))
        self.buffer_hit = 0

    @torch.no_grad()
    def compute_distances(self, inputs):
        assert inputs.size(-1) == self.embed_dim
        inputs_flat = inputs.reshape(-1, self.embed_dim)
        codebook_t = self.weight[:-1, :].t()
        if self.using_znorm:
            raise ValueError('Using_znorm mode not supported now')
            inputs_flat = F.normalize(inputs_flat, dim=-1)  # Corrected function name from normalized to normalize
            codebook_emb = F.normalize(codebook_t, dim=-1)            

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs.shape[:-1], -1)  # [B, L, n_embed or n_embed+1]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs
    
    def reinitialize_low_usage_embeddings(self):
        """
        Reinitialize embeddings that have a usage count below 1/10 of the average usage.
        """
        # Calculate the average usage across all embeddings
        normlized_usage = self.usage_tracker / self.usage_tracker.sum()
        low_usage_threshold = 1. / self.n_embed * self.unuse_token_thres
        
        # Find indices of embeddings with usage below the threshold
        low_usage_indices = torch.where(normlized_usage < low_usage_threshold)[0]
        
        # Reinitialize these embeddings randomly
        nn.init.trunc_normal_(self.weight.data[low_usage_indices], std=0.1)

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B -1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x   

    @torch.no_grad()
    def _update_ema_buffers(self, vectors, idxs):
        n_embed = self.n_embed
        vectors = vectors.reshape(-1, self.embed_dim)
        idxs = idxs.reshape(-1)
        
        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        
        # if self.unuse_token_thres > 0:
        if True:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            
            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)
        
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + self.n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    @torch.no_grad()
    def _update_buffers(self, idxs):
        # Update usage tracker
        idxs = idxs.view(-1)            # flatten the idxs matrix
        num_hit = torch.tensor(idxs.numel(), device=self.weight.device)
        hit_V = idxs.bincount(minlength=self.n_embed).float()
        if dist.is_initialized():
            dist.all_reduce(hit_V, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_hit, op=dist.ReduceOp.SUM)
        
        # Update the usage tracker based on the number of hits 
        self.usage_tracker.mul_(0.99).add_(hit_V * 0.01) 
        self.buffer_hit += num_hit
            
        # if self.unuse_token_thres > 0:
        #     self.reinitialize_low_usage_embeddings()
            # import pdb; pdb.set_trace; from IPython import embed; embed()   

    def forward(self, inputs, update_ema=True):
        # split inputs to multiple entities if needed (split_VQ)
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        inputs_split = inputs.view(inputs.size(0), -1, self.embed_dim)
        embed_idxs = self.find_nearest_embedding(inputs_split)
        if self.training:
            self._update_buffers(embed_idxs)
            if self.ema and update_ema:
                self._update_ema_buffers(inputs_split, embed_idxs)
        embeds = self.embed(embed_idxs).view_as(inputs)
        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds

    def extra_repr(self) -> str:
        return super().extra_repr() + f', znorm={self.using_znorm}'
    

# downsample video
def downsample_video(video_data, downsample_scale):
    t, h, w = video_data.shape[-3:]
    new_size = (t, h//downsample_scale, w//downsample_scale)
    resize_video = interpolate(video_data, size=new_size, mode='trilinear', align_corners=True)
    
    return resize_video


# get string for exp_name
def get_exp_name(cfg):
    if cfg['trainer'] in ['enc_trainer']:
        # Cycle{}_{}_{}_
        #     cfg['optimizer']['epoch_cycles'],
        #     cfg['optimizer']['cycle_decay'],
        #     cfg['optimizer']['full_cosine'],
        name_str = 'Pthw_{}_{}_{}_VQ_{}_{}_{}_KL_{}_{}'.format(
            cfg['model']['args']['tokenizer']['args']['patch_t'],
            cfg['model']['args']['tokenizer']['args']['patch_hw'],
            cfg['model']['args']['tokenizer']['args']['learn_pe'],
            cfg['model']['args']['vq_layer']['use_vq'], 
            cfg['model']['args']['vq_layer']['vq_weight'], 
            cfg['model']['args']['vq_layer']['commit_loss'], 
            cfg['model']['args']['kl_layer']['deterministic'], 
            cfg['model']['args']['kl_layer']['kl_weight'],
        )
        if cfg['model']['name'] ==  'nerv_enc':
            name_str += 'Z_{}_{}_{}_{}_token_{}_{}_{}'.format(
                cfg['model']['args']['nerv_tokens']['z_type'],
                cfg['model']['args']['nerv_tokens']['z_dim'], 
                cfg['model']['args']['nerv_tokens']['z_norm'], 
                cfg['model']['args']['nerv_tokens']['z_norm_first'], 
                cfg['model']['args']['nerv_tokens']['token_nums'], 
                cfg['model']['args']['nerv_tokens']['token_dims'],
                cfg['model']['args']['transformer_encoder']['name'], 
            )
        elif cfg['model']['name'] in ['param_enc', 'tnerv_enc']:
            name_str += '_Z{}'.format(cfg['model']['args']['decoder']['z_tokens'])
            if cfg['model']['args']['decoder']['z_bottleneck']:
                name_str += '_{}_{}_{}'.format( 
                    cfg['model']['args']['decoder']['z_dim'],
                    cfg['model']['args']['decoder']['z_norm'],
                    cfg['model']['args']['decoder']['z_norm_first'],
                )
            name_str += '_d{}_PE_{}_{}_H_{}_C{}_K{}_E{}_D{}'.format(
                cfg['model']['args']['decoder']['d_tokens'],
                cfg['model']['args']['decoder']['learn_p_pe'],
                cfg['model']['args']['decoder']['learn_z_pe'],
                cfg['model']['args']['decoder']['use_v_tokens'],
                cfg['model']['args']['decoder']['final_ch'],
                cfg['model']['args']['decoder']['final_ks'],
                cfg['model']['args']['encoder']['name'], 
                cfg['model']['args']['decoder']['name'], 
            )
    elif cfg['trainer'] in ['diffuse_v2n_trainer', 'diffuse_z2n_trainer']:
        # prob{}_att_{}_{}_out{}
        name_str = '{}_nerv_{}_{}_C{}_z_{}_diffuse_{}_{}_{}'.format(
            cfg['model']['args']['transformer_encoder']['name'],
            cfg['model']['args']['nerv_tokens']['token_nums'], 
            cfg['model']['args']['nerv_tokens']['token_dims'],
            # cfg['model']['args']['transformer_encoder']['cond_dropout_prob'],
            # cfg['model']['args']['transformer_encoder']['sum_t_y'],
            # cfg['model']['args']['transformer_encoder']['final_bias'],
            cfg['model']['args']['hyponet']['args']['out_dim'],
            cfg['model']['args']['nerv_tokens']['z_type'],
            cfg['model']['diffuser']['predict_xstart'],
            cfg['model']['diffuser']['learn_sigma'],
            cfg['model']['diffuser']['sigma_small'],
        )
    elif cfg['trainer'] in ['diffuse_trainer', 'diffuse_v2nn_trainer']:
        if cfg['model']['name'] == 'diffuse_z':
            name_str = '{}_porb{}'.format(cfg['model']['args']['name'], cfg['model']['args']['cond_dropout_prob'])
        elif cfg['model']['name'] == 'diffuse_zn':
            name_str = ''
        else:
            name_str = '{}_prob{}_S{}_LPE{}'.format(
                cfg['model']['args']['name'],
                cfg['model']['args']['args']['cond_dropout_prob'],
                cfg['model']['args']['args']['split_num'],
                cfg['model']['args']['args']['learn_pe'],
            )
        name_str += '_diffuse_{}_{}_{}'.format(
            cfg['model']['diffuser']['predict_xstart'],
            cfg['model']['diffuser']['learn_sigma'],
            cfg['model']['diffuser']['sigma_small'],
        )
    return name_str    


# importaance sampling for diffusion steps
def importance_sampling(a, T, N):
    """
    Perform importance sampling on a range of integers from 0 to T-1.

    :param a: Scaling factor for importance weights.
    :param T: The upper limit for the range of integers (0 to T-1).
    :param N: The number of samples to draw.
    :return: A tensor containing N sampled integers based on importance weights.
    """
    
    # Generate integers from 0 to T-1
    T, N = int(T), int(N)
    integers = torch.arange(T, dtype=torch.float)
    weights = torch.exp(a * integers / T)
    weights = weights / weights.sum()

    # Sample according to the importance weights
    sampled_indices = torch.multinomial(weights, N, replacement=True)
    sampled_integers = integers[sampled_indices].int()

    return sampled_integers


# clamp outputer to output reasonabl videos
def ClampTensor(x, out_range='no'):
    assert out_range in ['no', 'neg', 'zero']
    if out_range not in ['no']:
        v_max = 1
        v_min = 0 if out_range=='zero' else -1
        x = x.clamp(min=v_min, max=v_max)
    return x


def ActLayer(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'gelu':
        return nn.GELU()
    else:
        NotImplementedError


# get sin-cos postion embedding
def get_1d_sincos_pos_embed(embed_dim, embed_num):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = np.arange(embed_num).reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_sincos_pos_embed(embed_dim, grid_size):    
    """
    embed_dim: output dimension for each position
    grid_size: a tuple of positions to be encoded: size (M,N,P)
    out: (M N P D)
    """    
    # Initialize the result tensor with zeros
    if isinstance(grid_size, int):
        grid_size = (grid_size, )
    result = np.zeros((*grid_size, embed_dim))

    # Sum embeddings by expanding along new axes and broadcasting
    emb_list = [get_1d_sincos_pos_embed(embed_dim, x) for x in grid_size]
    for emb_id, emb in enumerate(emb_list):
        expanded_emb = emb.reshape((1,)*(len(grid_size)-1) + emb.shape)
        expanded_emb = np.swapaxes(expanded_emb, emb_id, -2)
        result += expanded_emb

    return torch.from_numpy(result).float()


# Norm layer for bottleneck
class NormLayer(nn.Module):
    def __init__(self, norm_type, z_dim, token_num):
        super().__init__()
        if norm_type == 'bn_bn':
            self.norm_layer = nn.SyncBatchNorm(z_dim)
        elif norm_type == 'bn_b':
            self.norm_layer = nn.SyncBatchNorm(z_dim*token_num)
        elif norm_type == 'ln_d':
            self.norm_layer = nn.LayerNorm(z_dim)
        elif norm_type == 'ln_nd':
            self.norm_layer = nn.LayerNorm((token_num, z_dim))
        elif norm_type == 'no':
            self.norm_layer = nn.Identity()
        else:
            raise NotImplementedError(f'Non-supported type {norm_type}')
        self.norm_type = norm_type
        self.token_num = token_num

    def forward(self, x):
        if self.norm_type == 'bn_bn':
            x = einops.rearrange(x, 'b n d -> b d n')
            x = self.norm_layer(x)
            x = einops.rearrange(x, 'b d n -> b n d')
        elif self.norm_type == 'bn_b':
            x = einops.rearrange(x, 'b n d -> b (n d)')
            x = self.norm_layer(x)
            x = einops.rearrange(x, 'b (n d) -> b n d', n=self.token_num)
        else:
            x = self.norm_layer(x)
        return x


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

    def kl(self, avg_kl=False):
        kl_loss = 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
        if avg_kl:
            return kl_loss.mean()
        else:
            return kl_loss.sum() / kl_loss.size(0)

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


class MLP(nn.Module):
    def __init__(self, dim_list, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        layers = [nn.Linear(dim_list[i], dim_list[i+1]) for i in range(len(dim_list)-1)]
        self.layers = nn.ModuleList(layers)
        self.act = act_layer()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x


def degrade_video(data, head_task, is_train=True):
    raw_video = data.transpose(1,2).clone()   # b,t,c,h,w  
    if head_task == 'colorization':
        input_video = rgb_to_grayscale(raw_video, 3).transpose(1,2)
    elif 'sr' in head_task:
        t_scale, h_scale, w_scale = [int(x) for x in head_task.split('_')[-3:]]
        t_start, h_start, w_start = 0, 0, 0
        downsample_vid = raw_video[:,t_start::t_scale,:,h_start::h_scale,w_start::w_scale]
        t, h, w = raw_video.size(1), raw_video.size(-2), raw_video.size(-1)
        downsample_vid = einops.rearrange(downsample_vid, 'b t c h w -> b c t h w')
        if 't_hw' in head_task:
            downsample_vid = interpolate(downsample_vid, size=(t, h//h_scale, w//w_scale), mode='trilinear', align_corners=True)
        elif 'hw_t' in head_task:
            downsample_vid = interpolate(downsample_vid, size=(t//t_scale, h, w), mode='trilinear', align_corners=True)
        input_video = interpolate(downsample_vid, size=(t, h, w), mode='trilinear', align_corners=True)
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
    elif 'outpaint' in head_task:
        outpaint_size = 1 - float(head_task.split('_')[-1])
        small_x, large_x = outpaint_size/2, 1- outpaint_size/2
        h, w = data.shape[-2:]
        input_video = torch.zeros_like(data)
        left, right = int(w * small_x), int(w * large_x)
        top, bottom = int(h * small_x), int(h * large_x)
        input_video[...,top:bottom,left:right] = data[...,top:bottom,left:right]
    elif 'img2vid' in head_task:
        num_frames = raw_video.size(1)
        select_frame_idx = random.randint(0, num_frames-1) if is_train else 0
        select_frame = data[:,:,select_frame_idx].clone().detach()
        input_video = torch.zeros_like(data).fill_(select_frame.mean())
        input_video[:,:,select_frame_idx] = select_frame
        # if self.training:
        #     # random pick a frame
        #     rand_frame_idx = random.randint(0, num_frames-1)
        #     select_frame = data[:,:,rand_frame_idx].clone().detach()
        # else:
        #     # pick the first frame
        #     select_frame = data[:,:,0].clone().detach()
        # input_video = select_frame.unsqueeze(2).repeat(1,1,num_frames,1,1)
    else:
        input_video = data

    return input_video              # b,c,t,h,w
    # .clamp(min=-1, max=1)          


def split_str(s, type='int'):
    if not isinstance(s, str):
        s = str(s)
    type = float if type=='float' else int
    return [type(x) for x in str(s).split('_')]    
    

def ensure_path(path, replace=True):
    if os.path.exists(path):
        if replace:
            print(f'removing {path}')
            shutil.rmtree(path)
        basename = os.path.basename(path.rstrip('/'))
        if basename.startswith('_'):
            os.makedirs(path)
    else:
        os.makedirs(path)


def put_breakpoint():
    import pdb; pdb.set_trace; from IPython import embed; embed()  


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
    

# Trace ModulateNeRV forward
class ModulateNeRV(torch.nn.Module):
    def __init__(self, hyper_nerv, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.hyper_nerv = hyper_nerv
        # deepcopy(hyper_nerv)
        for reduant_attr in ['tokenizer', 'transformer_encoder', 'wtoken_postfc']:
            if hasattr(self.hyper_nerv, reduant_attr):
                delattr(self.hyper_nerv, reduant_attr)

    def forward(self, modulate_params):
        device = modulate_params[list(modulate_params.keys())[0]].device
        self.hyper_nerv.to(device)
        pred_frames = self.hyper_nerv.forward_hyponet(modulate_params, self.n_frames)['pred_frames']
        return pred_frames  # t b c h w
    
    
# Trace NeRV forward
class BaseNeRV(torch.nn.Module):
    def __init__(self, hyponet, n_frames):
        super().__init__()
        self.hyponet = deepcopy(hyponet)
        self.n_frames = n_frames

    def forward(self, params):
        B, device = params['wb0'].size(0), params['wb0'].device
        self.hyponet.to(device)
        # hyponet = self.hyponet
        coord = self.hyponet.get_coord(self.n_frames)
        coord = einops.repeat(coord, 't d -> b t d', b=B)
        self.hyponet.set_params(params)
        pred_frames = self.hyponet(coord.to(device))
        return pred_frames  # t b c h w
    

# trace full nerv forward
class BaseNeRV2(torch.nn.Module):
    def __init__(self, hyponet, n_frames):
        super().__init__()
        coord = hyponet.get_coord(n_frames)
        self.embed = hyponet.convert_posenc(coord).view(n_frames, -1) # (T, PE_DIM)
        self.n_frames, self.depth, self.act, self.OutImg = n_frames, hyponet.depth, hyponet.act, hyponet.OutImg
        self.conv_shape_list, self.ps_layers = hyponet.conv_shape_list, hyponet.ps_layers

    @staticmethod
    def batched_conv(x, wb, conv_shape, ps_layer):
        B = wb.size(0)
        ch_out, ch_in, cur_ks, pad = conv_shape
        conv_weigts = wb[:,:-1].view((B, ch_in, cur_ks, cur_ks, ch_out))
        conv_weigts = conv_weigts.permute(0,-1,1,2,3).flatten(end_dim=1)
        conv_bias = wb[:,-1:].flatten()
        x = torch.nn.functional.conv2d(x, conv_weigts, conv_bias, stride=1, padding=pad, groups=B)
        return ps_layer(x)

    def forward(self, params):
        B = params['wb0'].size(0)
        x = self.embed.repeat(B, 1).view(self.n_frames, -1, 1, 1).to(params['wb0'].device)  # (T, B*PE_DIM, 1, 1)
        for i in range(self.depth):
            x = self.batched_conv(x, params[f'wb{i}'], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = x.view((self.n_frames, B, -1) + x.shape[-2:])
        return x  # t b c h w
    

################# Tensor quantization and dequantization #################
def quant_tensor(t, bits=8):
    tmin_scale_list = []

    # quantize over the whole tensor, or along each dimenstion
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / (2**bits-1)
    tmin_scale_list.append([t_min, scale])
    for axis in range(t.dim()):
        t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
        if t_min.nelement() / t.nelement() < 0.02:
            scale = (t_max - t_min) / (2**bits-1)
            # tmin_scale_list.append([t_min, scale]) 
            tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)]) 
    # import pdb; pdb.set_trace; from IPython import embed; embed() 
     
    quant_t_list, new_t_list, err_t_list = [], [], []
    for t_min, scale in tmin_scale_list:
        t_min, scale = t_min.expand_as(t), scale.expand_as(t)
        quant_t = ((t - t_min) / (scale)).round().clamp(0, 2**bits-1)
        new_t = t_min + scale * quant_t
        err_t = (t - new_t).abs().mean()
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append(err_t)   

    # choose the best quantization 
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_new_t = new_t_list[best_quant_idx]
    best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
    best_tmin = tmin_scale_list[best_quant_idx][0]
    best_scale = tmin_scale_list[best_quant_idx][1]
    quant_t = {'quant': best_quant_t, 'min': best_tmin, 'scale': best_scale}

    return {'quant_t': quant_t, 'best_new_t': best_new_t}
    # return quant_t, best_new_t             


def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'], quant_t['scale']
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t.float()


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, 'w')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def set_save_dir(save_dir, replace=True):
    ensure_path(save_dir, replace=replace)
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    logger = set_logger(os.path.join(save_dir, f'log_{time_str}.txt'))
    writer = SummaryWriter(os.path.join(save_dir, f'tensorboard'))
    return logger, writer, time_str


def compute_num_params(model, full_model=True, text=True):
    if full_model:
        tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    else:
        if hasattr(model, 'base_params'):
            tot = int(sum([v.nelement() for k,v in model.base_params.items()]))
        else:
            tot = 0

    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        elif tot >= 1e3:
            return '{:.1f}K'.format(tot / 1e3)
        else:
            return str(tot)
    else:
        return tot

def text2str(tot):
    if tot >= 1e6:
        return '{:.1f}M'.format(tot / 1e6)
    elif tot >= 1e3:
        return '{:.1f}K'.format(tot / 1e3)
    else:
        return str(tot)

def make_optimizer(params, optimizer_spec, load_sd=False):
    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']](params, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class EpochTimer():

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(secs):
    if secs >= 3600:
        return f'{secs / 3600:.1f}h'
    elif secs >= 60:
        return f'{secs / 60:.1f}m'
    else:
        return f'{secs:.1f}s'


# Deprecated
# get fix 3d pos_embed
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return torch.from_numpy(pos_embed).float()


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

    