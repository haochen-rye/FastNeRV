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

parser = argparse.ArgumentParser()
parser.add_argument('--lmdb_dir', type=str, default='~/data/LMDB/k400_train_L240188_T8_sample_evenly_Size_256+0_ClsVidNum_400_1e4',)
parser.add_argument('--batch_size', '-b', type=int, nargs='+', default=[8,], help='batch size',)
parser.add_argument('--workers', '-j', type=int, nargs='+', default=[8,], help='num workers',)
parser.add_argument('--vid_num', '-n', type=float, default=1e4,)
parser.add_argument('--epochs', '-e', type=int, default=1, help='total epochs',)
parser.add_argument('--out_path', type=str, default='outputs/nvloader_results/lmdb', help='out path',)

args = parser.parse_args()
args.vid_num = int(args.vid_num)

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

    def get_video_data(self, index):
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

        return video

    def __getitem__(self, index):
        video = self.get_video_data(index)
        return {'video':video, 'path': self.pathes[index], 'label': self.labels[index]}

# add video decoding/dataloading results
vid_num_list, codec_list, time_list, fps_list, avg_vid_size_list, epoch_list = [[] for _ in range(6)]
def add_results(fps, total_time, v_codec_str, avg_vid_size, vid_num=args.vid_num, epochs=args.epochs):
    avg_vid_size_list.append(round(avg_vid_size, 3))
    vid_num_list.append(vid_num)
    epoch_list.append(epochs)
    fps_list.append(round(fps, 1))
    time_list.append(round(total_time,2))
    codec_list.append(v_codec_str)


# iterate over video list
def iterate_loading(iterate_item, batch_size, v_codec_str, dec_fun=None,):
    epoch_time_list, epoch_fps_list, i = [], [], 1
    for epoch in range(args.epochs):
        start = time.time()
        pbar = tqdm(iterate_item)
        for batch in pbar:
            if dec_fun != None:
                batch = dec_fun(batch)
            # if torch.cuda.is_available():
            #     batch = {k:v.cuda() for k,v in batch.items()}
            vps = batch_size * i / (time.time() - start)
            i += 1
            pbar.set_description(f'{v_codec_str}: VPS={vps:.1f}')
        total_time = time.time() - start
        epoch_time_list.append(total_time/60)
        epoch_fps_list.append(args.vid_num/total_time)
    avg_vps, total_time = sum(epoch_fps_list)/len(epoch_fps_list), sum(epoch_time_list)
    return avg_vps, total_time

def dump_results(vid_num_list, epoch_list, time_list, codec_list, avg_Vid_size_list, fps_list, dump_csv=False):
    result_df = pd.DataFrame.from_dict({'vid_num': vid_num_list, 'epoch': epoch_list,
        'time (min)': time_list, 'codec': codec_list, 'avg vid (MB)': avg_vid_size_list, 'FPS': fps_list, })    
    print(result_df)    
    if dump_csv:
        time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        exp_name = os.path.split(args.lmdb_dir)[-1]
        out_path = f'{args.out_path}/result_{time_str}_{exp_name}.csv'
        result_df.to_csv(out_path)
        print(f'dumped results to {out_path}')

vid_dataset = LmdbDataset(args.lmdb_dir)
shape_str, avg_vid_size = vid_dataset.shape_str, vid_dataset.avg_vid_size

#  evaluate decoding speed
codec_str = f'decode_{shape_str}'
avg_vps, total_time = iterate_loading(range(args.vid_num), 1, codec_str, vid_dataset.get_video_data)
add_results(avg_vps, total_time, codec_str, avg_vid_size)
dump_results(vid_num_list, epoch_list, time_list, codec_list, avg_vid_size, fps_list)

# import pdb; pdb.set_trace; from IPython import embed; embed()    
# dataloader speed
sub_vid_dataset = Subset(vid_dataset, list(range(args.vid_num)))
for batch_size in args.batch_size:
    for workers in args.workers:
        codec = f'dataloader_b{batch_size}_j{workers}'
        train_dataloader = torch.utils.data.DataLoader(sub_vid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, )
        avg_vps, total_time = iterate_loading(train_dataloader, batch_size, codec)
        add_results(avg_vps, total_time, codec, avg_vid_size)
        dump_results(vid_num_list, epoch_list, time_list, codec_list, avg_vid_size, fps_list)
dump_results(vid_num_list, epoch_list, time_list, codec_list, avg_vid_size, fps_list, True)
