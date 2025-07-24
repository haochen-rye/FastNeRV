import json
import os
import pickle
from hashlib import md5
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.distributed



'''
Example to pre-calculate FVD stats using a single GPU for a python dataset and save it as a pickle file. 
Note that the FVD stats change if you use a different frame rate / crop size / other configs.
Important: Here we assume data is in [0, 1] and BCTHW format.

ds = _  # define you dataset
fvdc = FVDCalculator()
stats_path = 'path/to/save/stats.pkl'
fvdc.get_feature_stats_for_dataset(ds, stats_pkl_path=stats_path)

'''



'''
Example to calculate FVD between a generated video data batch and a real video dataset.
Important: Here we assume data is in [0, 1] and BCTHW format.

dataset_feats_path = 'path/to/real/dataset/stats.pkl'
feat_real = FeatureStats.load(dataset_feats_path)

videos = _  # define your generated video data batch
fvdc = FVDCalculator()
feats_gen = fvdc.get_feature_stats_for_batch(videos)
fvd = fvdc.calculate_fvd(feats_gen, feat_real)


# If the real stats is not available, you can calculate it on the fly.
# The results will be automatically cached for future use.
# This is NOT recommanded as it is slow and sometimes crashes due to memory / DDP issues.
# Only use this if you are using a very small dataset or you are debugging.
fvd = fvdc.calculate_fvd_with_dataset(feats_gen, dataset_real)
'''



def calc_dataset_md5(dataset):
    return md5(json.dumps(dataset.__dict__, sort_keys=True).encode('utf-8')).hexdigest()

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = edict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj
    


class FVDCalculator:
    def __init__(self, i3d_path=None, zero_center_vid=True, device='cuda'):
        if i3d_path is None:
            i3d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
        self.zero_center_vid = zero_center_vid
        self.device = device
        self.i3d = torch.jit.load(i3d_path).eval().to(device)
        if torch.distributed.is_initialized():
            self.num_gpus = torch.distributed.get_world_size() # use world_size as num_gpus
        else:
            self.num_gpus = torch.cuda.device_count() # use local device count as num_gpus

    @torch.inference_mode()
    def get_feature_stats_for_batch(self, batch, feats=None, num_gpus=None):
        if num_gpus is None:
            num_gpus = self.num_gpus
        if feats is None:
            feats = FeatureStats(capture_mean_cov=True)

        # import pdb; pdb.set_trace; from IPython import embed; embed()   
        if isinstance(batch, Dict):
            data = batch['video']
        else:
            data = batch
            
        # Note: the used i3d torchscript model expects input in [-1, 1] and size=224x224.
        # if setting resize=True, the input will be resized to 224x224
        # if setting rescale=True, model expects input in [0, 255] and the model will rescale 
        # the input to [-1, 1] internally.
            
        # Here we assume data is in [0, 1] and BCTHW
        # so wee need to rescale data to [-1, 1] without setting rescale        
        assert isinstance(data, torch.Tensor) and data.ndim == 5 # BCTHW
        if not self.zero_center_vid:
            data = (data - 0.5) * 2
        features = self.i3d(data.to('cuda'), resize=True, return_features=True)
        feats.append_torch(features, num_gpus=num_gpus)
        return feats
    

    def get_feature_stats_for_dataset(
            self, 
            dataset: Dataset, 
            bs=32, 
            num_workers=4,
            stats_pkl_path=None
            # cache_stats=True,
        ): # always use a single gpu
        assert isinstance(dataset, Dataset), f'Expected a torch Dataset, but got {type(dataset)}'
        # dataset_md5 = calc_dataset_md5(dataset)
        # dataset_name = Path(dataset.csv_file).stem 

        # if stats_pkl_path is None:
        #     stats_pkl = f'fvd_stats_{dataset_name}_{dataset_md5}.pkl'
        #     stats_cache_path = Path(__file__).resolve().parent / 'stats_cache'
        #     stats_cache_path.mkdir(exist_ok=True)
        #     stats_pkl_path = stats_cache_path / stats_pkl

        if stats_pkl_path != None and os.path.exists(stats_pkl_path):
            return FeatureStats.load(stats_pkl_path)
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank == 0:
                feats = self.calculate_stats_for_dataset(dataset, bs, num_workers)
                if stats_pkl_path != None:
                    feats.save(stats_pkl_path)
            torch.distributed.barrier()
            if rank != 0:
                assert os.path.exists(stats_pkl_path)
                feats = FeatureStats.load(stats_pkl_path)
        else:
            feats = self.calculate_stats_for_dataset(dataset, bs, num_workers)
            if stats_pkl_path != None:
                feats.save(stats_pkl_path)
        
        return feats
    

    def calculate_stats_for_dataset(self, dataset, bs=32, num_workers=4):
        feats = FeatureStats(capture_mean_cov=True)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
        for batch in tqdm(loader, desc='Extracting features'):
            feats = self.get_feature_stats_for_batch(batch, feats, num_gpus=1)
        return feats


    def calculate_fvd(self, feats_gen, feats_real):
        mu_gen, cov_gen = feats_gen.get_mean_cov()
        mu_real, cov_real = feats_real.get_mean_cov()

        # updated for better numerical stability
        mu_gen = torch.from_numpy(mu_gen)
        cov_gen = torch.from_numpy(cov_gen)
        mu_real = torch.from_numpy(mu_real)
        cov_real = torch.from_numpy(cov_real)
        
        a = (mu_gen - mu_real).square().sum(dim=-1)
        b = cov_gen.trace() + cov_real.trace()
        c = torch.linalg.eigvals(cov_gen @ cov_real).sqrt().real.sum(dim=-1)
        fvd = a + b - 2 * c

        return fvd


    def calculate_fvd_with_dataset(self, feats_gen, dataset_real, bs=32, stats_pkl_path=None):
        feats_real = self.get_feature_stats_for_dataset(dataset_real, bs, stats_pkl_path=stats_pkl_path)
        return self.calculate_fvd(feats_gen, feats_real)