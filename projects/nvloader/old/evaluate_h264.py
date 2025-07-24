import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
from torch.utils.data import Dataset
from decoder import decode
from torchvision.transforms import CenterCrop
from torch.utils.data import Subset
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class VideoDataSet(Dataset):
    def __init__(self, num_frames, sampling_rate, backend, args):
        path_df=pd.read_csv(args.csv_path)
        self.num_frames, self.sampling_rate = num_frames, sampling_rate
        self.backend = backend
        self.vid_path_list = path_df['path']
        self.crop = CenterCrop(args.crop_size)
        # import pdb; pdb.set_trace; from IPython import embed; embed()    

    def __len__(self):
        return len(self.vid_path_list)

    def __getitem__(self, idx):
        try: 
            frames = decode(self.vid_path_list[idx], self.sampling_rate, self.num_frames, 
                backend=self.backend) 
            crop_frames = self.crop(frames)
            # print(f'crop_frames: {crop_frames.shape}')
            return crop_frames
        except Exception as e:
            print(e)


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default='/data/home/haochen1/nerv_projects/OneNeRV/data/dataset_meta/meta/k400_val.csv',
        help='csv file with video paths',)
parser.add_argument('--data_len', '-l', type=int, default=1000, help='dataset length',)
parser.add_argument('--crop_size', type=int, default=224, help='center crop size',)
parser.add_argument('--batch_size', '-b', type=int, nargs='+', default=[8,], help='batch size',)
parser.add_argument('--workers', '-j', type=int, nargs='+', default=[8,], help='num workers',)
parser.add_argument('--backend', type=str, nargs='+', default=['pyav', 'decord', 'torchvision', ],
        help='backend type',)
parser.add_argument('--epochs', '-e', type=int, default=5, help='total epochs',)
parser.add_argument('--sampling_rate', type=int, nargs='+', default=[4], help='dataset length',)
parser.add_argument('--frames', type=int, nargs='+', default=[16, ], help='dataset length',)
parser.add_argument('--out_path', type=str, default='output/h264_speed',
        help='out path',)
parser.add_argument('--pin_memory', action='store_true', help='pin memory for dataloader')

args = parser.parse_args()

frames_list, patch_list, data_list, time_list, fps_list = [], [], [], [], []
epoch_list, backend_list, batch_list, workers_list = [], [], [], []
for num_frames in args.frames:
    for sampling_rate in args.sampling_rate:
        for backend in args.backend:
            for batch_size in args.batch_size:
                for workers in args.workers:
                    frames_list.append(f'{num_frames}_{sampling_rate}')
                    data_list.append(args.data_len)
                    epoch_list.append(args.epochs)
                    batch_list.append(batch_size)
                    workers_list.append(workers)
                    patch_list.append(1)
                    backend_list.append(backend)
                    full_dataset = VideoDataSet(num_frames, sampling_rate, backend, args)
                    train_dataset =  Subset(full_dataset, list(range(args.data_len)))
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=args.pin_memory, sampler=None, drop_last=False)
                    epoch_time_list, epoch_fps_list = [], []
                    i = 1
                    for epoch in range(args.epochs):
                        start = time.time()
                        pbar = tqdm(train_dataloader)
                        for batch in pbar:
                        # for batch in tqdm(train_dataloader, desc=f'{backend} Epoch {epoch}:'):
                            vps = batch_size * i / (time.time() - start)
                            # input_vid = batch
                            i += 1
                            pbar.set_description(f'{backend}: VPS={vps:.1f}')
                        total_time = time.time() - start
                        epoch_time_list.append(total_time/60)
                        epoch_fps_list.append(args.data_len/total_time)
                    fps_list.append(sum(epoch_fps_list)/len(epoch_fps_list))
                    time_list.append(sum(epoch_time_list))

result_df = pd.DataFrame.from_dict({'frame': frames_list, 'data': data_list, 'epoch': epoch_list,
    'patch': patch_list, 'time (min)': time_list, 'FPS': fps_list, 'backend': backend_list})    
print(result_df)
time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
result_df.to_csv(f'{args.out_path}/result_{time_str}.csv')
