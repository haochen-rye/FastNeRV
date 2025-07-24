import os
# from random import randint
import random
import math
import numpy as np
from tqdm import tqdm
import platform
import lmdb
import pickle
from glob import glob
import shutil
import itertools

import decord
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.io import read_image

import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Accurate seek is not implemented for pyav backend.*')

from datasets import register
import logging
log = logging.getLogger(__name__)

decord.bridge.set_bridge('torch')

# LMDB config setups
NUM_ATTEMPTS = 60*24
NUM_PER_SHARD = int(1e4)

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


@register('video_dataset')
class VideoDataSet(Dataset):
    def __init__(self, frame_num, t_duration, cls_vid_num, crop_size, csv_file='', zero_center=False,
        root_path='~/nerv_projects/FastNeRV/data/dataset_meta/meta', split='train', strd='evenly', 
        resize=False, rand_flip=False, decode_type='torchvision', lmdb_root='',):
        cls_num, vid_num = [int(float(x)) for x in cls_vid_num.split('_')]
        assert csv_file.endswith('.csv')
        csv_data = pd.read_csv(os.path.expanduser(os.path.join(root_path, csv_file)))
        # create subset: select the top VID_NUM videos for each class, then CLS_NUM videos per calss
        if 'label' in csv_data:
            vid_list = csv_data.sort_values(['label', 'path']).groupby('label').head(vid_num)   
            vid_list = pd.concat([group for _,group in vid_list.groupby('label')][:cls_num])
            self.vid_list, labels_list, self.text = [vid_list[k].tolist() for k in ['path', 'label', 'action']]
            # convert labels to start from 0
            label_idx = {y:x for x,y in enumerate(set(labels_list))}
            self.labels = [label_idx[y] for y in labels_list]
        else: 
            self.vid_list = csv_data.sort_values('path')['path'][:vid_num].tolist()
            self.labels = [0] * len(self.vid_list)
        self.n_frames, self.t_strd, self.t_duration = frame_num, strd, t_duration
        self.zero_center = zero_center
        self.transform = VideoTransform(frame_num, crop_size, split, resize, rand_flip, zero_center)
        self.use_lmdb = decode_type.lower() in ['lmdb']
        if self.use_lmdb:
            lmdb_dir = csv_file.replace('.csv', f'_{cls_vid_num}')
            lmdb_dir = os.path.join(os.path.expanduser(lmdb_root) , lmdb_dir)
            self.anno_file = os.path.join(lmdb_dir, 'tensor_shapes.pkl') 
            def GetShapeList():
                with open(self.anno_file, 'rb') as f:
                    return pickle.load(f)       
            if os.path.isfile(self.anno_file) and len(GetShapeList())==len(self):
                log.info(f'Reusing cached database at {lmdb_dir}')
            else:
                if os.path.isdir(lmdb_dir):
                    log.info(f'Removing {lmdb_dir}...')
                    shutil.rmtree(lmdb_dir)                    
                self.create_lmdb_dataset(lmdb_dir)
            self.video_shapes = GetShapeList()
            self.db_files = sorted(glob(os.path.join(lmdb_dir, 'shard*.lmdb')))
            self.db_list = None
            self.db_cfg = LMDB_config
        # import pdb; pdb.set_trace; from IPython import embed; embed()  


    def create_lmdb_dataset(self, out_dir, db_cfg=LMDB_config,
            name_fmt='shard_{:03d}.lmdb', num_per_shard=int(1e4)):

        os.makedirs(out_dir, exist_ok=True)
        if platform.system() != "Linux" and db_cfg.map_size == 1099511627776 * 2:
            raise ValueError("Provide a reasonable default map_size for your operating system.")
        else:
            map_size = db_cfg.map_size

        def open_lmdb(lmdb_path):
            db = lmdb.open(
                lmdb_path,
                subdir=False,
                map_size=map_size,
                readonly=False,
                meminit=db_cfg.meminit,
                writemap=True,
                map_async=True,
            )
            txn = db.begin(write=True)
            return db, txn

        # finalize dataset
        def close_lmdb(db, txn):
            if db and txn:
                txn.commit()
                db.close()

        db, txn = None, None
        shape_list = []
        class IterateData(Dataset):
            def __init__(self, vid_len, dec_func):
                self.vid_len = vid_len
                self.dec_func = dec_func

            def __len__(self):
                return self.vid_len

            def __getitem__(self, idx):
                return self.dec_func(idx)

        data_loader = DataLoader(IterateData(len(self), self.decode_video), 
            batch_size=1, shuffle=False, num_workers=min(os.cpu_count(), 64) )
        for idx, batch in enumerate(tqdm(data_loader, desc=f'Creating lmdb dataset at {out_dir}')):
            video = batch[0]
            if idx % num_per_shard == 0:
                close_lmdb(db, txn)
                shard_num = 1 + idx // num_per_shard
                lmdb_path = os.path.join(out_dir, name_fmt.format(shard_num))
                db, txn = open_lmdb(lmdb_path)
            video = np.asarray(video)       # c,t,h,w
            shape_list.append(video.shape)
            txn.put(str(idx).encode("ascii"), video.tobytes())

        close_lmdb(db, txn)
        with open(self.anno_file, 'wb') as f:
            pickle.dump(shape_list, f)


    def __len__(self):
        return len(self.vid_list)
        

    def __getitem__(self, idx):
        video = self.load_video(idx)        # C,T,H,W
        video = self.transform(video)
        return {'video': video, 'path': self.vid_list[idx], 'label': self.labels[idx]}


    def load_video(self, idx):
        load_func = self.get_video_from_LMDB if self.use_lmdb else self.decode_video
        return load_func(idx)

    
    def decode_video(self, idx):
        vid_path = self.vid_list[idx]
        if os.path.isfile(vid_path):
            reader = torchvision.io.VideoReader(vid_path, "video")
            duration = reader.get_metadata()['video']['duration'][0]
            if duration <= self.t_duration + 0.5:
                t_start, t_end = 0, duration
            else:
                t_start = random.uniform(0, duration - self.t_duration - 0.2)
                t_end = t_start + self.t_duration
            def extract_frames(reader, t_start, t_end, num_frame):
                frames = []
                for f in reader.seek(t_start):
                    if f['pts'] <= t_end or len(frames) < num_frame:
                        frames.append(f['data'])
                    else:
                        break
                return frames
            # frames = [f['data'] for f in itertools.takewhile(lambda x: x['pts'] <= t_end, reader.seek(t_start))]
            frames = extract_frames(reader, t_start, t_end, self.n_frames)
            video = torch.stack(frames, 1)      # C T H W
        else:
            path_list = sorted(glob(vid_path + '/*.png'))
            frame_list = [read_image(x) for x in path_list]
            video = torch.stack(frame_list, 1)
        return video


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


    def get_video_from_LMDB(self, index):
        if self.db_list is None:
            self.init_db()
            assert(len(self.db_list))
        # orig_idx = self.orig_idx[index]
        db_idx = index // NUM_PER_SHARD
        if db_idx >= len(self.db_list):
            print(db_idx, len(self.db_list))
        with self.db_list[db_idx].begin(write=False) as txn:
            key = str(index).encode('ascii')
            byteflow = txn.get(key)
        try:
            data_block = np.frombuffer(byteflow, dtype=np.uint8).reshape(self.video_shapes[index])
        except ValueError as e:
            log.info(f'id: {index}/(len(self))')
            
        video = torch.from_numpy(np.copy(data_block)) #  C,T,H,W
        # video = torch.frombuffer(byteflow, dtype=torch.uint8).view(self.video_shapes[index])  #  C,T,H,W
        return video
   

    # Deprecate
    def get_frame_idxs(self, total_frames,  strd):
        if isinstance(strd, str) and 'even' in strd:
            frame_strd = total_frames / self.n_frames
        else:
            frame_strd = float(strd)
        frame_idx = [int(x) for x in torch.arange(0, total_frames, frame_strd)]
        return frame_idx
    

class VideoTransform(object):
    def __init__(
        self, n_frames=8, crop_size=256, split='train', resize=False, rand_flip=False,  zero_center=False,
    ):
        self.n_frames = n_frames
        tfm_list = []
        if resize:
            tfm_list.append(transforms.Resize(crop_size))
        if split=='train':
            tfm_list.append(transforms.RandomCrop(crop_size))
            if rand_flip:
                tfm_list.append(transforms.RandomHorizontalFlip())
        else:
            tfm_list.append(transforms.CenterCrop(crop_size))
        self.vid_tfm = transforms.Compose(tfm_list)
        self.split = split
        self.zero_center = zero_center


    def __call__(self, video):
        video = self.pad_frames(video)
        video = self.vid_tfm(video) / 255.
        if self.zero_center:
            video.mul_(2).add_(-1)

        return video             # C,T,H,W


    def pad_frames(self, video):
        video = video.clone().float() 
        if video.size(1) >= self.n_frames:
            # C,T,H,W
            t_idx = torch.linspace(0, video.size(1)-1, self.n_frames).int()
            video = video[:,t_idx]
            # t_strd = video.size(1) // self.n_frames
            # extra_frames = video.size(1) - self.n_frames * t_strd
            # starting_idx = random.randint(0, extra_frames) if self.split=='train' else 0
            # end_idx = starting_idx + self.n_frames * t_strd
            # video = video[:,starting_idx:end_idx:t_strd]
        else:
            video = F.pad(video, (0,0,0,0,0,self.n_frames-video.size(1)), mode='replicate') 
        assert video.size(1) == self.n_frames
        return video
