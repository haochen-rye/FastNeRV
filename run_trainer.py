"""
    Generate a cfg object according to a cfg file and args, then spawn Trainer(rank, cfg).
"""

import argparse
import os
from mergedeep import merge

import yaml
import torch
import torch.multiprocessing as mp
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import utils
import trainers
import numpy as np
import random

torch.use_deterministic_algorithms(True, warn_only=True) # for cudnn deterministic
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--data_path', default='data/k400')
    parser.add_argument('--csv_file', default='k400_train.js')
    parser.add_argument('--t_duration', type=float, default=1.0)
    parser.add_argument('--frame_num', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--num_workers', '-j', type=int, default=16)
    parser.add_argument('--out_path', default='output path')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--port_offset', '-p', type=int, default=0)
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    parser.add_argument('--opts', type=str, nargs='*', default=[], help='cfg args to update')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    return args

def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                if v.startswith('$') and v.endswith('$'):
                    v = getattr(args, v.replace('$', ''))
                d[k] = v
    translate_cfg_(cfg)

    if args.tag != '':
        args.tag = args.tag + '_'

    env = dict()
    tot_gpus = torch.cuda.device_count()
    env['tot_gpus'] = tot_gpus
    env['cudnn'] = args.cudnn
    env['wandb_upload'] = args.wandb_upload
    cfg['env'] = env

    def build_tree(tree_list):
        if len(tree_list)>=2:
            return {tree_list[0]: build_tree(tree_list[1:]) if len(tree_list)>2 else tree_list[-1]}

    def nested_v(dict, keys):
        for key in keys:
            dict = dict[key]
        return dict

    def convert(type, x):
        if type == bool and isinstance(x, str):
            if x.lower() == 'true':
                return True
            elif x.lower() == 'false':
                return False
            else:
                raise ValueError('Cannot convert {} to bool'.format(x))
        else:
            return type(x)

    assert len(args.opts) % 2 == 0
    for cur_cfg_key, v in zip(args.opts[::2], args.opts[1::2]):
        keys = cur_cfg_key.split('.')
        cur_v = nested_v(cfg, keys)
        try:
            v = convert(type(cur_v), v)
        except:
            raise AttributeError(f'{keys}: {cur_v}, {v}')
        cfg = merge(cfg, build_tree(keys+[v]))

    env['exp_name'] = '{}G{}_F{}_t{}_Size{}_B{}_E{}_lr{}_{}_N{}_{}'.format(
        args.tag, tot_gpus, args.frame_num, args.t_duration, args.input_size, 
        args.batch_size, cfg['max_epoch'],
        cfg['optimizer']['args']['lr'], 
        cfg['optimizer']['lr_type'],
        cfg['train_dataset']['args']['cls_vid_num'], 
        utils.get_exp_name(cfg))
    env['save_dir'] = os.path.join(args.out_path, env['exp_name'])
    env['port'] = 10000 + args.port_offset    
    cfg['manualSeed'] = args.manualSeed

    return cfg


def main_worker(rank, cfg):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    manualSeed = cfg['manualSeed'] 
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) # for cudnn deterministic

    trainer = trainers.trainers_dict[cfg['trainer']](rank, cfg)
    trainer.run()


def main():
    args = parse_args()

    cfg = make_cfg(args)
    utils.ensure_path(cfg['env']['save_dir'], args.replace)

    if cfg['env']['tot_gpus'] > 1:
        mp.spawn(main_worker, args=(cfg,), nprocs=cfg['env']['tot_gpus'])
    else:
        main_worker(0, cfg)


if __name__ == '__main__':
    main()
