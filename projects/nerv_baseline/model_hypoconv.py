import torch
import torch.nn as nn
import numpy as np
import math

def batched_conv(x, wb, conv_shape, ps_layer):
    B = wb.size(0)
    ch_out, ch_in, cur_ks, pad = conv_shape
    # import pdb; pdb.set_trace; from IPython import embed; embed()      
    conv_weigts = wb[:,:-1].view((B, ch_in, cur_ks, cur_ks, ch_out))
    conv_weigts = conv_weigts.permute(0,-1,1,2,3).flatten(end_dim=1)
    conv_bias = wb[:,-1:].flatten()
    x = torch.nn.functional.conv2d(x, conv_weigts, conv_bias, stride=1, padding=pad, groups=B)
    return ps_layer(x)

def init_wb(out_ch, in_ch, ks):
    weight = torch.empty(in_ch, out_ch, ks, ks)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(out_ch)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    wb_list = [weight.permute(0,2,3,1).flatten(end_dim=-2), bias[None]]
    return torch.cat(wb_list, dim=0).detach()

class HypoConvnets(nn.Module):

    def __init__(self, args):
        super().__init__()
        from math import ceil
        in_dim, out_dim, self.use_pe = 1, 3, True
        hid_dim, strds_list, ks_list, act = args.hid_dim, args.strds, args.ks, args.act
        self.pe_dim, self.pe_sigma, self.out_bias = args.pe_dim, args.pe_sigma, args.out_bias
        last_dim = in_dim * self.pe_dim

        depth = len(strds_list)
        ks_list += ks_list[-1:] * (depth - len(ks_list))
        ch = hid_dim

        self.ps_layers = nn.ModuleList()
        self.params = nn.ParameterDict()
        self.conv_shape_list = []
        for i, cur_strd in enumerate(strds_list):
            cur_dim = ch if i < depth - 1 else out_dim
            cur_ks = ks_list[i]
            ch_out = cur_dim*cur_strd**2
            self.params[f'wb{i}'] = nn.Parameter(init_wb(ch_out, last_dim, cur_ks))
            cur_pad = 0 if (cur_ks == cur_strd and i == 0) else ceil(cur_ks - 1) // 2
            self.conv_shape_list.append((ch_out, last_dim, cur_ks, cur_pad)) 
            self.ps_layers.append(nn.PixelShuffle(cur_strd))
            last_dim = cur_dim

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            NotImplementedError
        self.depth = depth

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def OutImg(self, x):
        if self.out_bias == 'sigmoid':
            return torch.sigmoid(x)
        elif self.out_bias == 'tanh':
            return (torch.tanh(x) * 0.5) + 0.5
        else:
            return x + float(self.out_bias)

    def forward(self, x):
        B, T = 1, x.size(0)
        x = self.convert_posenc(x) 
        x = x.view(T, -1, 1, 1)  # (T, B*PE_DIM, 1, 1)

        for i in range(self.depth):
            x = batched_conv(x, self.params[f'wb{i}'][None], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = x.view((T, B, -1) + x.shape[-2:])
        # import pdb; pdb.set_trace; from IPython import embed; embed()      
        return x.squeeze(1)

def make_coord_grid(shape, range=(-1, 1), device=None):
    """
        Args:
            shape: tuple
            range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
        Returns:
            grid: shape (*shape, )
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    return grid

from decord import VideoReader
from torchvision.transforms import CenterCrop, Resize, Compose
from torch.utils.data import Dataset

class VideoDataSet(Dataset):
    def __init__(self, args):
        vr = VideoReader(args.vid_path)
        frame_num = args.frames
        if frame_num == -1:
            frame_idx = list(range(len(vr)))
        else:
            frame_idx = [int(x*len(vr)/frame_num) for x in range(frame_num)]
        video = torch.from_numpy(vr.get_batch(frame_idx).asnumpy())
        video = video.permute(0,-1,1,2).float() / 255. # # T,H,W,C -> T,C,H,W
        transform = Compose([Resize(size=args.crop_size, antialias=False), CenterCrop(args.crop_size)])
        self.video = transform(video)
        
    def __len__(self):
        return self.video.size(0)

    def __getitem__(self, idx):
        return {'frame': self.video[idx], 'idx': idx}

def adjust_lr(optimizer, cur_epoch, args):
    # cur_epoch = (cur_epoch + cur_iter) / args.epochs
    if 'hybrid' in args.lr_type:
        up_ratio, up_pow, down_pow, min_lr, final_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 1 - (1 - final_lr) * ((cur_epoch - up_ratio) / (1. - up_ratio))**down_pow
    elif 'cosine' in args.lr_type:
        up_ratio, up_pow, min_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - up_ratio)/ (1 - up_ratio)) + 1.0)
    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult        

import random
def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

import torch.distributed as dist
def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors
