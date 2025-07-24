import os 
from glob import glob
import torch
import lmdb
import json
import argparse
from tqdm import tqdm
import time
from torchvision.utils import save_image     
import pandas as pd
import subprocess
from torch.utils.data import Subset
import torch.nn.functional as F
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, default='/fsx/haochen1/data/LMDB/k400_val_L19877_T8_sample_evenly_Size_256+0_ClsVidNum_400_1e4',)
parser.add_argument('--test_dir', type=str, default='/fsx/haochen1/data/LMDB/',)
parser.add_argument('--filter', type=str, nargs='+', default=['T8', 'L19877',])
parser.add_argument('--out_path', type=str, default='outputs/nvloader_results/quality', help='out path',)

args = parser.parse_args()

class LMDB_config:
    """Configuration options for the LMDB. These should be reasonable defaults.

    For large datasets with random access (for example ImageNet trained with shuffle=True), try setting readahead=False.
    """

    # Writing:
    map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    write_frequency = 10240  # how often to flush during database creation
    shuffle_while_writing = False  # Shuffle during DB creation
    db_channels_first = True
    rounds = 1  # Can write multiple rounds of the same dataset (for example with different augmentations).

    # reading:
    max_readers = 128
    readahead = True  # this should be beneficial for long sequential reads, disable when randomly accessing large DBs
    meminit = True
    max_spare_txns = 128

ANNO_FILE = 'anno.js'

class LmdbDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_dir, db_cfg=LMDB_config):
        self.db_list, self.num_per_shard = None, int(1e4)
        lmdb_dir = os.path.expanduser(lmdb_dir)
        self.db_files = sorted(glob(os.path.join(lmdb_dir, 'shard*.lmdb')))
        self.anno_file_path = os.path.join(lmdb_dir, ANNO_FILE)
        with open(self.anno_file_path, 'r') as f:
            anno_data = json.load(f)
        self.pathes, self.labels, self.shapes, self.keys = [anno_data[x] for x in ['path', 'label', 'shape', 'key']]
        self.db_cfg = db_cfg
        self.shape_str = '_'.join(str(x) for x in list(self.shapes[0]))
        result = subprocess.check_output(['du',  lmdb_dir], universal_newlines=True, stderr=subprocess.STDOUT)
        self.avg_vid_size = float(result.split()[0]) / 1e3 / len(self.labels)
        self.path_dict = {path:key for path, key in zip(self.pathes, self.keys)}

    def __len__(self):
        return len(self.labels)

    def init_db(self):
        self.db_list = []
        for file in self.db_files:
            db = lmdb.open(file,
                subdir=False,
                max_readers=self.db_cfg.max_readers,
                readonly=True,
                lock=False,
                readahead=self.db_cfg.readahead,
                meminit=self.db_cfg.meminit,
                max_spare_txns=self.db_cfg.max_spare_txns)
            self.db_list.append(db)

    def get_video_data(self, index, path=None):
        if path !=None:
            index = self.path_dict[path]
        if self.db_list is None:
            self.init_db()
            assert(len(self.db_list))
        db_idx = index // self.num_per_shard
        if db_idx >= len(self.db_list):
            print(db_idx, len(self.db_list))
        with self.db_list[db_idx].begin(write=False) as txn:
            key = str(self.keys[index]).encode('ascii')
            byteflow = txn.get(key)
        video = torch.frombuffer(byteflow, dtype=torch.uint8).view(self.shapes[index])  #  C,T,H,W

        return {'video':video, 'path': self.pathes[index]}

    def __getitem__(self, index):
        video = self.get_video_data(index)
        return {'video':video, 'path': self.pathes[index], 'label': self.labels[index]}

dataset_dict = {}
full_vid_dir = os.path.expanduser(args.test_dir)
for sub_dir in sorted(os.listdir(full_vid_dir)):
    if sum([x in sub_dir for x in args.filter]) == len(args.filter) and sub_dir not in args.gt_dir:
        codec_name = '_'.join(sub_dir.split('_')[:6])
        dataset_dict[codec_name] = LmdbDataset(os.path.join(full_vid_dir, sub_dir))
        cur_avg_vid_size = dataset_dict[codec_name].avg_vid_size
        # print(f'{codec_name}: {cur_avg_vid_size}M')
gt_dataset = LmdbDataset(args.gt_dir)

from pytorch_msssim import ssim
from lpips import LPIPS

def get_psnr(vid0, vid1, max_pixel_value = 255):
    # mse = F.mse_loss(vid0.float()/max_pixel_value, vid1.float()/max_pixel_value)
    # psnr = -10 * torch.log10(mse)
    mse = F.mse_loss(vid0.float(), vid1.float(), reduction='none').mean(dim=(0,2,3))
    psnr = 10 * torch.log10((max_pixel_value ** 2) / mse).mean()
    # psnr = torch.nan_to_num(psnr, posinf=50)
    return psnr

def get_ssim(vid0, vid1, max_pixel_value = 255):
    vid_list = [x.float().transpose(0,1) for x in [vid0, vid1]]
    ssim_v = ssim(vid_list[0], vid_list[1], data_range=max_pixel_value)
    return ssim_v

lpips_fn = LPIPS(net='alex', eval_mode=True)
def get_lpips_alex(vid0, vid1, max_pixel_value = 255):
    vid_list = [(x.float().transpose(0,1) / max_pixel_value - 0.5)*2 for x in [vid0, vid1]]
    lpips_v = lpips_fn(vid_list[0], vid_list[1]).mean()
    return lpips_v

results_dict = {}
metric_list = ['psnr', 'ssim', 'lpips'][:2]
func_list = [get_psnr, get_ssim, get_lpips_alex][:2]
metric_precision = {'psnr':1, 'ssim':3, 'lpips': 4}
pbar = tqdm(range(len(gt_dataset)))
for i in pbar:
    gt_data = gt_dataset.get_video_data(i)
    for cur_codec, cur_dataset in dataset_dict.items():
        test_data = cur_dataset.get_video_data(0, gt_data['path'])
        for metric, metric_func in zip(metric_list, func_list):
            if metric not in results_dict:
                results_dict[metric] = {}
            # c,t,h,w
            metric_v = metric_func(gt_data['video'].cuda(), test_data['video'].cuda())
            if cur_codec not in results_dict[metric]:
                results_dict[metric][cur_codec] = []
            results_dict[metric][cur_codec].append(metric_v.cpu())

    if i and i % 100 == 0:
        out_dict = {k:{kk:torch.stack(vv).mean().item() for kk, vv in v.items()} for k,v in results_dict.items()}
        out_df = pd.DataFrame(out_dict)
        print(out_df)

time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
os.makedirs(args.out_path, exist_ok=True)
out_path = f'{args.out_path}/quality_{time_str}.csv'
out_df.to_csv(out_path)
print(f'dumped results to {out_path}')