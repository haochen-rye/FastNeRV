"""
    A basic trainer.

    The general procedure in run() is:
        make_datasets()
            create . train_loader, test_loader, dist_samplers
        make_model()
            create . model_ddp, model
        train()
            create . optimizer, epoch, log_buffer
            for epoch = 1 ... max_epoch:
                adjust_learning_rate()
                train_epoch()
                    train_step()
                evaluate_epoch()
                    evaluate_step()
                visualize_epoch()
                save_checkpoint()
"""

import os
import os.path as osp
from math import sqrt
import time
from pandas import DataFrame
import numpy as np
from pytorch_msssim import ssim

from torchvision.utils import make_grid
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image     
from torchvision.io import write_video
from torch.utils.data import Subset, ConcatDataset
import numpy as np
from math import cos, pi
from lpips import LPIPS

import datasets
import models
import utils
from trainers import register
from utils import FeatureStats, FVDCalculator

@register('base_trainer')
class BaseTrainer():

    def __init__(self, rank, cfg):
        self.rank = rank
        self.cfg = cfg
        self.is_master = (rank == 0)

        env = cfg['env']
        self.tot_gpus = env['tot_gpus']
        self.distributed = (env['tot_gpus'] > 1)
        self.use_amp = cfg['use_amp']

        # Setup log, tensorboard, wandb
        if self.is_master:
            logger, writer, self.time_str = utils.set_save_dir(env['save_dir'], replace=False)
            with open(osp.join(env['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)

            self.log = logger.info
            self.train_ret, self.val_ret = {}, {}

            self.enable_tb = True
            self.writer = writer

            if env['wandb_upload']:
                self.enable_wandb = True
                with open('wandb.yaml', 'r') as f:
                    wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                os.environ['WANDB_DIR'] = env['save_dir']
                os.environ['WANDB_NAME'] = env['exp_name']
                if 'api_key' in wandb_cfg:
                    os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
                wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfg)
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        # Setup distributed devices
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        if self.distributed:
            dist_url = f"tcp://localhost:{env['port']}"
            dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=self.tot_gpus, rank=rank)
            self.log(f'Distributed training enabled.')

        cudnn.benchmark = env['cudnn']
        self.dump_metrics = ['psnr', 'ssim', 'lpips', 'kl', 'acc', 'fps']
        self.save_visual_dir = os.path.join(self.cfg['env']['save_dir'], 'visualize')
        self.base_lr = cfg['optimizer']['args']['lr']
        self.lr_type = cfg['optimizer']['lr_type']
        self.zero_center_vid = cfg['zero_center']
        self.log_every_n = cfg['log_every_n']

        # load fvd stats for real data
        self.fvd_calculator = FVDCalculator(zero_center_vid=self.zero_center_vid, device=self.device)
        self.fvd_real_stats = None
        self.fvd_real_stats_path = cfg['eval_cfgs']['fvd_real_stats_path']
        if self.fvd_real_stats_path not in [None, 'no', ''] and os.path.exists(self.fvd_real_stats_path):
            self.fvd_real_stats = FeatureStats.load(self.fvd_real_stats_path)

        self.log(f'Environment setup done.')


    def valid_cfg_attr(self, key, invalid_attrs=['no', 'false']):
        if key in self.cfg and self.cfg[key].lower() not in invalid_attrs:
            return True
        return False

    def run(self):
        self.make_datasets()
        self.epoch_steps = len(self.train_loader)
        self.starting_epoch = 1
        resume_ckt_path = os.path.join(self.cfg['env']['save_dir'], 'epoch-last.pth') 
        latest_ckt = None
        if self.valid_cfg_attr('resume_ckt'):
            resume_ckt_path = self.cfg['resume_ckt']
        if os.path.exists(resume_ckt_path):
            latest_ckt = torch.load(resume_ckt_path)
            self.make_model(latest_ckt['model'], load_sd=True)
            self.starting_epoch = int(latest_ckt['epoch']) + 1
        else:
            self.make_model()

        if self.valid_cfg_attr('eval_model'):
            self.epoch, self.step, self.t_data, self.t_model = 0, 0, 0, 0
            self.log_buffer = []
            self.evaluate_epoch()
            self.visualize_epoch()
            self.log(', '.join(self.log_buffer))
            self.dump_csv(self.cfg, extra_str='eval_')
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            if latest_ckt is not None and self.use_amp:
                assert 'scaler_sd' in latest_ckt
                self.scaler.load_state_dict(latest_ckt['scaler_sd'])            
            self.train()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

            Cfg example:

            train/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        cfg = self.cfg
        self.dist_samplers = []

        def make_distributed_loader(dataset, loader_cfg, shuffle=False,
                drop_last=False, persistent_workers=True, init_fn=None):
            # print('distributed: ', self.distributed)
            sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None         
            batch_size, num_workers = [loader_cfg[x] for x in ['batch_size', 'num_workers']]
            loader = DataLoader(
                dataset,
                batch_size // self.tot_gpus,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers // self.tot_gpus,
                worker_init_fn=init_fn,
                pin_memory=True,
                persistent_workers=persistent_workers)
            return loader, sampler

        if cfg.get('train_dataset') is not None:
            train_dataset_cfg = cfg['train_dataset']
            train_dataset = datasets.make(train_dataset_cfg, args={'zero_center': self.zero_center_vid})
            self.log(f'Train dataset: len={len(train_dataset)}')
            self.cfg.update({'TrainSize': len(train_dataset),})
            self.train_loader, train_sampler = make_distributed_loader(train_dataset, 
                train_dataset_cfg['loader'], shuffle=True, drop_last=True)
            self.dist_samplers.append(train_sampler)

        if cfg.get('test_dataset') is not None:
            test_dataset_cfg = cfg['test_dataset']
            self.test_loader_dict = {}
            for dataset_name, dataset_csv in test_dataset_cfg['csv_paths'].items():
                test_dataset = datasets.make(test_dataset_cfg, 
                    args={'csv_file': dataset_csv, 'split': 'val', 'zero_center': self.zero_center_vid})
                self.log(f'Test dataset: {dataset_name}, len={len(test_dataset)}')
                self.cfg.update({f'TestSize_{dataset_name}': len(test_dataset),})
                test_loader, test_sampler = make_distributed_loader(test_dataset, 
                    test_dataset_cfg['loader'], shuffle=False, drop_last=False, )
                self.test_loader_dict.update({dataset_name: test_loader})
                self.dist_samplers.append(test_sampler)


    def make_vis_datasets(self, distributed=False):
        def get_vislist(dataset, n_vis):
            sub_dataset = torch.utils.data.Subset(dataset, torch.linspace(0, len(dataset)-1, n_vis).int())
            vid_list = [x['video'] for x in sub_dataset]
            return torch.stack(vid_list, dim=0)

        self.vislist_dict = {}
        n_vis = self.cfg['vis_n']
        if n_vis and hasattr(self, 'test_loader_dict'):
            vislist_test = []
            for k, test_loader in self.test_loader_dict.items():
                vislist_test.append(get_vislist(test_loader.dataset), n_vis)
            self.vislist_dict['test'] = torch.cat(vislist_test, dim=0)
            # if hasattr(self, 'train_loader'):
            #     self.vislist_dict['train'] = get_vislist(self.train_loader.dataset)
       

    def make_model(self, model_spec=None, load_sd=False):
        if model_spec is None:
            model_spec = self.cfg['model']

        model = models.make(model_spec, load_sd=load_sd)
        self.log(model)
        hypernet_size = utils.compute_num_params(model, text=False)
        basemodel_size = utils.compute_num_params(model, False, False)
        embed_size = 0
        total_size = basemodel_size + embed_size   
        hypersize_str = utils.text2str(hypernet_size)
        size_str = f'{utils.text2str(total_size)}_{utils.text2str(basemodel_size)}_{utils.text2str(embed_size)}' 
        self.log(f'Model: #params={hypersize_str}')
        self.log(f'BaseModel: #params(total_base_embed)={size_str}')
        if hasattr(model, 'bottleneck_str'):
            self.log(f'Bottleneck {model.bottleneck_str}')

        if hasattr(model, 'num_modulate_w'):
            bottleneck_size, modulate_size, modulate_prefc, modulate_postfc = [model.num_modulate_w[x] 
                for x in ['bottleneck_w', 'modulate_w', 'modulate_w_prefc', 'modulate_w_postfc']]
            bottleneck_str = utils.text2str(bottleneck_size)
            self.cfg.update({'ModulateSize': bottleneck_str})
            self.log(f'BottleneckWeights: #params={bottleneck_str}')
            self.log(f'ModulateWeights: #params={utils.text2str(modulate_size)}')
            self.log(f'ModulateWeights_pre_fc: #params={utils.text2str(modulate_prefc)}')
            self.log(f'ModulateWeights_post_fc: #params={utils.text2str(modulate_postfc)}')
            if self.enable_wandb:
                wandb.run.summary['BottleneckWeights #params'] = bottleneck_str
                wandb.run.summary['ModulateWeights #params'] = utils.text2str(modulate_size)
                wandb.run.summary['ModulateWeights_pre_fc #params'] = utils.text2str(modulate_prefc)
                wandb.run.summary['ModulateWeights_post_fc #params'] = utils.text2str(modulate_postfc)            

        if self.enable_wandb:
            wandb.run.summary['Model #params'] = hypersize_str
            wandb.run.summary['BaseModel #params(total_base_embed)'] = size_str

        self.cfg.update({'HyperSize': hypersize_str})
        if hasattr(model, 'hyponet'):
            self.log(f'HyponetParam: [layername: (ch_out, ch_in, k)]. {model.hyponet.param_shapes}')
            self.log(f'Token_nums: {model.token_num_list}, Token_dims: {model.token_dim_list}')
            self.cfg.update({'Size': size_str, 'Hyponet': model.hyponet.param_shapes})

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters=True)
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp


    @classmethod
    def dump_cfg(self, cfg, kv_dict={}, prefix=''):
        for k,v in cfg.items():
            if k != 'sd':
                if isinstance(v, dict):
                    kv_dict = self.dump_cfg(v, kv_dict, f'{prefix}{k}_')
                else:
                    kv_dict.update({f'{prefix}{k}': v})
        return kv_dict
        

    @staticmethod
    def list2str(l, precision=None, joint_str='_'):
        return joint_str.join(str(round(x,precision) if precision else x) for x in l)
    

    def dump_csv(self, cfg, extra_str=''):
        if self.is_master:
            csv_dict = self.dump_cfg(cfg, {})
            epoch_list = sorted(self.train_ret.keys())  
            epoch_list_str = self.list2str(epoch_list, joint_str=',')
            if self.val_ret:
                # sample_ret = self.val_ret if hasattr(self, 'val_ret') else self.train_ret
                metrics_list = self.val_ret[next(iter(self.val_ret))].keys()
                test_datanames = sorted(self.val_ret.keys())
                out_dict = {}
                for metric in metrics_list:
                    precision_num = 3 if ('ssim' in metric or 'lpips' in metric) else 1
                    if len(self.train_ret) and metric in self.train_ret[epoch_list[0]]:
                        train_key, best_train_key = f'{metric}_train_e{epoch_list_str}', f'{metric}_best_train'
                        metric_v_list = [self.train_ret[x][metric] for x in epoch_list]
                        out_dict[train_key] = self.list2str(metric_v_list, precision=precision_num)
                        out_dict[best_train_key] = round(max(metric_v_list), precision_num)
                    for test_dataname in test_datanames:
                        out_dict[f'{test_dataname}_{metric}'] = round(self.val_ret[test_dataname][metric], precision_num)
                csv_dict.update(out_dict)
            csv_path = os.path.join(cfg['env']['save_dir'], f'{extra_str}results_e{int(self.epoch)}_{self.time_str}.csv')
            csv_df = DataFrame.from_dict(csv_dict, orient='index').T
            csv_df.to_csv(csv_path)
            self.log(f'CSV results dumped to {csv_path}')


    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), cfg['optimizer'])

        max_epoch = cfg['max_epoch']
        eval_epoch = cfg.get('eval_epoch', max_epoch + 1)
        vis_epoch = cfg.get('vis_epoch', max_epoch + 1)
        save_epoch = cfg.get('save_epoch', max_epoch + 1)
        epoch_timer = utils.EpochTimer(max_epoch)
        
        for epoch in range(self.starting_epoch, max_epoch + 1):
            self.epoch = epoch
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)

            self.t_data, self.t_model = 0, 0
            self.train_epoch()

            if epoch % vis_epoch == 0:
                self.visualize_epoch()

            if epoch % eval_epoch == 0:
                self.evaluate_epoch()
                self.dump_csv(cfg)

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')
            self.save_checkpoint('epoch-last.pth')
            self.extra_ops()

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model + 1e-10)
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

        self.dump_csv(cfg)


    def extra_ops(self):
        pass


    def adjust_learning_rate(self):
        lr = self.get_cur_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if self.step % self.epoch_steps == 0:
            self.log_temp_scalar('train/lr', lr, self.epoch)


    def get_cur_lr(self):
        return self.base_lr


    def get_lr_mult(self, lr_type, cur_epoch, max_epoch):
        if lr_type == 'cosine':
            warmup_epoch = 1
            if cur_epoch <= warmup_epoch:
                lr_mult = 0.1 + 0.9 * cur_epoch / warmup_epoch
            else:
                lr_mult = 0.5 * (cos(pi*(cur_epoch - warmup_epoch) / (max_epoch - warmup_epoch)) + 1)
        elif lr_type == 'step':
            lr_steps = np.array([0.9, 0.98]) * max_epoch
            lr_mult = 0.1 ** (sum(cur_epoch >= lr_steps))
        else:
            raise NotImplementedError(f'Non supported lr_type: {lr_type}')
        return lr_mult


    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({k: v}, step=t)


    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)


    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor(ave_scalars[k].item(), dtype=torch.float32, device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.item()
            ave_scalars[k].n *= self.tot_gpus


    def get_pbar_desc(self, ave_scalars, batch_idx):
        metric_names = ['gpu_usage', 'model_t', 'reserve_mem', 'allo_mem'][:1]
        metric_units = ['%', 's', 'GB', 'GB'][:1]
        for key in ave_scalars:
            if 'acc' in key:
                metric_names.append(key)
                metric_units.append('%')
            if 'psnr' in key:
                metric_names.append(key)
                metric_units.append('dB')
            if any(x in key for x in ['loss', 'base_kl']):
                metric_names.append(key)
                metric_units.append('')
        descrp_list = [f'{metric}={ave_scalars[metric].v:.2f}{metric_unit}' for metric, metric_unit in zip(metric_names, metric_units)]
        data_time_ratio = 100 * self.t_data / (self.t_data + self.t_model)
        avg_t_data, avg_t_model = [x/(batch_idx+1) for x in [self.t_data, self.t_model]]
        descrp_list += [f't_data: {avg_t_data:.2f}s', f't_model: {avg_t_model:.2f}s', f'data_t_ratio: {data_time_ratio:.1f}%']
        full_descrp_str = ' | '.join(descrp_list) 
        return full_descrp_str


    def ave_ret_scalars(self, ave_scalars, ret, batch_size):
        for k, v in ret.items():
            if ave_scalars.get(k) is None:
                ave_scalars[k] = utils.Averager()
            ave_scalars[k].add(v, n=batch_size)
        return ave_scalars


    def log_text(self, split, ave_scalars):
        logtext = ('' if 'train' in split else '\n') + f'{split}:'
        for k, v in ave_scalars.items():
            log_format = '.4f' if 'loss' in k else '.2f'
            logtext += f' {k}={v.item():{log_format}}'
            self.log_temp_scalar(f'{split}/' + k, v.item())
        self.log_buffer.append(logtext)


    def run_one_epoch(self, loader, split, step_func):
        ave_scalars = dict()
        self.step = (self.epoch-1) * self.epoch_steps
        t1 = time.time()
        for batch_idx, data in enumerate(loader):
            t0 = time.time()
            self.t_data += t0 - t1
            if 'train' in split:
                self.step += 1
                self.adjust_learning_rate()
            ret = step_func(data)
            torch.cuda.synchronize()
            self.t_model += time.time() - t0
            ave_scalars = self.ave_ret_scalars(ave_scalars, ret, len(next(iter(data.values()))))
            if self.is_master and batch_idx % self.log_every_n == 0:
                full_descrp_str = self.get_pbar_desc(ave_scalars, batch_idx)
                self.log(f'Epoch {self.epoch}, {batch_idx}/{len(loader)}: ' + full_descrp_str)
            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)
        self.log_text(split, ave_scalars)

        return ave_scalars


    def train_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        loss = self.model_ddp(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    

    def evaluate_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        with torch.no_grad():
            loss = self.model_ddp(data)
        return {'loss': loss.item()}


    def dump_ret(self, ave_scalars, metrics):
        dataset_dict = {}
        def valid_key(k, metrics=metrics):
            for metric in metrics:
                if metric in k:
                    return True
            return False

        for k,v in ave_scalars.items():
            if valid_key(k):
                dataset_dict[k] = v.v
        return dataset_dict

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = self.run_one_epoch(self.train_loader, 'train', self.train_step)
        if self.is_master:
            self.train_ret[self.epoch] = self.dump_ret(ave_scalars, self.dump_metrics)

    @torch.no_grad()
    def evaluate_epoch(self):
        self.model_ddp.eval()
        for dataset_name, test_loader in self.test_loader_dict.items():
            ave_scalars = self.run_one_epoch(test_loader, f'eval {dataset_name}', self.evaluate_step)
            if self.is_master:
                self.val_ret[dataset_name] = self.dump_ret(ave_scalars, self.dump_metrics)

    def visualize_epoch(self):
        pass


    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_spec = self.cfg['model']
        model_spec['sd'] = self.model.state_dict()
        optimizer_spec = self.cfg['optimizer']
        optimizer_spec['sd'] = self.optimizer.state_dict()
        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': self.epoch,
            'cfg': self.cfg,
        }
        if self.use_amp:
            checkpoint['scaler_sd'] = self.scaler.state_dict()
        torch.save(checkpoint, osp.join(self.cfg['env']['save_dir'], filename))


    def make_vis_dirs(self, sub_dirs=['vid', 'img', 'vid_grid']):
        dir_list = [os.path.join(self.save_visual_dir, x) for x in sub_dirs]
        if self.is_master:
            for cur_dir in dir_list:
                os.makedirs(cur_dir, exist_ok=True)
        self.vid_dir, self.img_dir, self.grid_dir = dir_list

   
    def dump_results(self, tag, res):
        res = self.rescale_video(res)
        self.dump_video_grid(tag, res)
        self.dump_videos(tag, res)
        self.dump_imgs(tag, res)

  
    def dump_video_grid(self, tag, res):
        if self.cfg['dump_vid_grid'] not in ['no']:
            nrows = int(sqrt(len(res) * 2))
            res_grid=make_grid([x.flatten(end_dim=1) for x in res], nrow=nrows)
            res_grid = res_grid.view(*res[0].shape[:2], *res_grid.shape[-2:])
            out_path = os.path.join(self.grid_dir, f'e{self.epoch}_{tag}_rank{self.rank}.mp4')
            self.log(f'Vid_grid dumped to {out_path}')
            write_video(out_path, res_grid.permute(0,2,3,1) * 255, fps=2, options={'crf':'15'})  # a uint8 tensor in[T, H, W, C] to write video


    def dump_videos(self, tag, res):
        if self.cfg['dump_video'] not in ['no']:
            self.log(f'dumped to {self.vid_dir}')
            for i in range(res.size(0)):
                batch_id, statues = i // 2, 'gt' if i %2==0 else 'pred'
                out_path = os.path.join(self.vid_dir, f'{batch_id:02d}_{statues}_{tag}.mp4')
                write_video(out_path, res[i].permute(0,2,3,1) * 255, fps=2, options={'crf':'15'})  # a uint8 tensor in[T, H, W, C] to write video


    def dump_imgs(self, tag, res):
        if self.cfg['dump_pred'] not in ['no']:
            self.log(f'dumped to {self.img_dir}')
            for i in range(res.size(0)):
                batch_id, statues = i // 2, 'gt' if i %2==0 else 'pred'
                out_path = os.path.join(self.img_dir, f'{batch_id:02d}_{statues}_{tag}.jpg')
                save_image(res[i], out_path)


    def track_gpus(self, start_time):
        # torch.cuda.synchronize()
        total_time = time.time() - start_time
        max_reserved_gb = torch.cuda.max_memory_reserved() / 10**9
        max_allocated_gb = torch.cuda.max_memory_allocated() / 10**9
        gpu_utilization = torch.cuda.utilization()  # Hypothetical function
        torch.cuda.reset_peak_memory_stats()     
        gpu_dict = {'total_time':total_time, 'gpu_usage': gpu_utilization, 
                'max_reserve_mem': max_reserved_gb, 
                 'max_allo_mem': max_allocated_gb}
        return gpu_dict


    def add_output(self, output, start, batch_size):
        gpu_dict = self.track_gpus(start)
        fps = batch_size / gpu_dict.pop('total_time')
        output.update({'fps': fps, **gpu_dict})
        return output
    

    def filter_dict(self, out_dict, filter_keys=[]):
        # remove un-necessary dict items
        for k,v in out_dict.items():
            if isinstance(v, torch.Tensor):
                if v.nelement() == 1:
                    out_dict[k] = v.detach().item()
                else:
                    filter_keys.append(k)

        for k,v in out_dict.items():
            if not isinstance(v, (int, float)):
                filter_keys.append(k)

        for filter_key in filter_keys:
            if filter_key in out_dict:
                del out_dict[filter_key]

        return out_dict
        

    @staticmethod
    def psnr(pred, target, data_range=1, start_dim=-3):
        mse = (pred.detach() - target.detach())**2
        psnr = -10 * torch.log10(mse.flatten(start_dim=start_dim).mean(-1) / data_range**2)
        psnr = torch.nan_to_num(psnr, posinf=50)
        return psnr.mean().cpu()    

    @torch.no_grad()
    def lpips(self, pred, target, data_range=1):
        # pred, target: b*t, c, h, w
        if not hasattr(self, 'lpips_fn'):
            self.lpips_fn = LPIPS(net='alex').to(self.device)
            self.lpips_fn.eval()            
        lpips_v = self.lpips_fn((pred-0.5)*2, (target-0.5)*2 ).mean()
        return lpips_v.cpu()
    
    def rescale_video(self, res):
        if self.zero_center_vid:
            return res * 0.5 + 0.5
        else:
            return res

    def vid2tensorboard(self, tag, res):
        res = self.rescale_video(res).clamp(min=0, max=1)
        if self.enable_tb:
            self.writer.add_video(tag, res, self.epoch, fps=1)    # res: `(N, T, C, H, W)` in [0, 1] for type `float`.
        if self.enable_wandb:
            wandb.log({tag: wandb.Video(res*255., fps=4, format='mp4')}, step=self.epoch,)
    