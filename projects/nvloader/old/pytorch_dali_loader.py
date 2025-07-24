import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
from tqdm import tqdm
import pandas as pd

from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy, DALIGenericIterator
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def fast_collate(batch, memory_format):
    """Based on fast_collate from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

def parse():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')

    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--disable_dali', default=False, action='store_true',
                        help='Disable DALI data loader and use native PyTorch one instead.')
    parser.add_argument('--load_videos', default=False, action='store_true',
                        help='Video loading istead of images.')
    
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--length', '-l', default=16, type=int,
                        help='Frames per video.')
    parser.add_argument('--stride', '-s', default=4, type=int,
                        help='Video sequence strides.')
    parser.add_argument('--fill', default=256, type=int,
                        help='Buffer size for shuffling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--fp16-mode', default=False, action='store_true',
                        help='Enable half precision mode.')
    parser.add_argument('--loss-scale', type=float, default=1)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    return args

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

@pipeline_def
def create_video_pipeline(filenames, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    videos = fn.readers.video_resize(device='gpu', filenames=filenames, 
                            sequence_length=args.length, stride=args.stride,
                            random_shuffle=True, initial_fill=args.fill, size=crop, mode='not_smaller')
    videos = fn.crop(videos, crop_h=crop, crop_w=crop)
    return videos

@pipeline_def
def create_img_pipeline(filenames, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=filenames,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")        
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    # labels = labels.gpu()
    return images


args = parse()

args.distributed = False
args.world_size = 1
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])
else:
    args.local_rank = 0

crop_size, val_size = 224, 224
if args.load_videos:
    csv_path = '/data/home/haochen1/nerv_projects/OneNeRV/data/dataset_meta/meta/k400_val.csv'
    path_list = pd.read_csv(csv_path)['path'].tolist()[:500]
    pipeline_func = create_video_pipeline
else:
    home = os.path.expanduser('~') 
    path_list = f"{home}/data/davis/JPEGImages/480p"
    pipeline_func = create_img_pipeline

# pytorch dataloader
train_pipe = pipeline_func(batch_size=args.batch_size,
                            num_threads=args.workers,
                            device_id=args.local_rank,
                            seed=12 + args.local_rank,
                            filenames=path_list,
                            crop=crop_size,
                            size=val_size,
                            dali_cpu=args.dali_cpu,
                            shard_id=args.local_rank,
                            num_shards=args.world_size,
                            is_training=True)
train_pipe.build()
#  reader_name="Reader",
train_loader = DALIGenericIterator(train_pipe, ['data'],
                                            last_batch_policy=LastBatchPolicy.PARTIAL,
                                            auto_reset=True)

class data_prefetcher():
    """Based on prefetcher from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __iter__(self):
        return self

    def __next__(self):
        """The iterator was added on top of the orignal example to align it with DALI iterator
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        if input is None:
            raise StopIteration
        return input, target

if args.disable_dali:
    data_iterator = data_prefetcher(train_loader)
    data_iterator = iter(data_iterator)
else:
    data_iterator = train_loader

start_time = time.time()
pbar = tqdm(enumerate(data_iterator))
for i, data in pbar:
    vps = (i+1)*args.batch_size / (time.time() - start_time)
    pbar.set_description(f'VPS={vps:.1f}')
        # print(data[0]['data'].shape)
    # pass
        # if args.disable_dali:
        #     input, target = data
        #     train_loader_len = len(train_loader)
        # else:
        #     input = data[0]["data"]
        #     target = data[0]["label"].squeeze(-1).long()
        #     train_loader_len = int(math.ceil(data_iterator._size / args.batch_size))
        # print(f'input: {input.shape}, {input.dtype}')
