import os
import time
from copy import deepcopy

import einops
import torch
from pytorch_msssim import ssim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from trainers import register
from utils import split_str
from trainers.base_trainer import BaseTrainer


@register('enc_trainer')
class EncTrainer(BaseTrainer):
    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        self.loss_weights = split_str(cfg['optimizer']['loss_weights'], 'float')
        self.max_epochs = cfg['max_epoch']
        self.epoch_cycles, self.cycle_decay, self.full_cosine = (cfg['optimizer'][key] for key in ('epoch_cycles', 'cycle_decay', 'full_cosine'))
        self.base_epochs = self.max_epochs // self.epoch_cycles
        def ssim_fn(pred, target):
            return 1 - ssim(pred, target, data_range=1, size_average=True)
        self.loss_funcs = [F.mse_loss, F.l1_loss, ssim_fn]        


    def get_cur_lr(self):
        cur_epoch = self.step / self.epoch_steps 
        cycle_decay = self.cycle_decay if (cur_epoch > self.base_epochs) else 1
        if not self.full_cosine:
            cur_epoch = cur_epoch % self.base_epochs
        lr = self.base_lr * self.get_lr_mult(self.lr_type, cur_epoch, self.base_epochs) * cycle_decay
        return lr


    def _iter_step(self, data, is_train):
        start, batch_size = time.time(), data['video'].size(0)
        video_data = data.pop('video').cuda(non_blocking=True)  # b c t h w
        output = self.model_ddp(video_data)
        output = self.compute_loss(output)
        output = self.measure_pred(output)
        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(output['loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()            
            # output['loss'].backward()
            # self.optimizer.step()
        output = self.add_output(output, start, batch_size)
        return self.filter_dict(output)


    def compute_loss(self, output):
        r_loss = self.get_reconstruction_loss(output)
        total_loss = r_loss
        extra_loss_list = [v for k, v in output.items() for loss_k in ['kl_loss', 'vq_loss'] if k.endswith(loss_k)]
        total_loss = total_loss + sum(extra_loss_list)

        return {**output, 'r_loss': r_loss, 'loss': total_loss}


    def get_reconstruction_loss(self, output):
        gt_frames = einops.rearrange(output['gt_frames'], 'b c t h w -> t b c h w')
        loss_list = []
        for k, pred_frames in output['out_frames'].items():
            if '_in' not in k:
                flat_pred, flat_gt = [x.flatten(end_dim=1) for x in [pred_frames, gt_frames]]
                cur_loss = [loss_func(flat_pred, flat_gt)*loss_weights for loss_func, loss_weights in zip(self.loss_funcs, self.loss_weights) if loss_weights]
                loss_list.append(sum(cur_loss))
        return sum(loss_list) / len(loss_list)

    
    @torch.no_grad()
    def measure_pred(self, output):
        gt = einops.rearrange(output['gt_frames'], 'b c t h w -> t b c h w')
        for k, pred in output['out_frames'].items():
            psnr = self.psnr(pred, gt, data_range=2 if self.zero_center_vid else 1)
            flat_pred, flat_gt = pred.detach().flatten(end_dim=1), gt.detach().flatten(end_dim=1)
            ssim_v = ssim(flat_pred, flat_gt, data_range=1)
            output.update({f'psnr_{k}': psnr, f'ssim_{k}': ssim_v, })

        return output


    def train_step(self, data):
        return self._iter_step(data, is_train=True)


    @torch.no_grad()
    def evaluate_step(self, data):
        return self._iter_step(data, is_train=False)


    @torch.no_grad()
    def visualize_epoch(self):
        self.get_vis_samples()
        self.model_ddp.eval()
        self.make_vis_dirs()
        for tag, vids in self.vis_samples.items():
            out_frames = self.model_ddp(vids.cuda())['pred_frames'].cpu().detach()
            res = einops.rearrange(out_frames, 't b c h w -> b t c h w')
            res = torch.stack([vids.transpose(1,2), res], dim=1).flatten(end_dim=1)
            self.dump_results(tag, res)
            self.vid2tensorboard(tag, res)
        # import pdb; pdb.set_trace; from IPython import embed; embed()    


    def get_vis_samples(self,):
        if not hasattr(self, 'vis_samples'):
            n_vis = self.cfg['vis_n']
            self.vis_samples = {}
            for k, test_loader in self.test_loader_dict.items():
                cur_dataset = test_loader.dataset
                sub_idx = torch.linspace(0, len(cur_dataset)-1, n_vis).int()
                sub_dataset = torch.utils.data.Subset(cur_dataset, sub_idx)
                vid_list = [x['video'] for x in sub_dataset]
                self.vis_samples[k] = torch.stack(vid_list, dim=0)

