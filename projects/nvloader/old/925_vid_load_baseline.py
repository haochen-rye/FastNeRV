import torch
import os
from tqdm import tqdm
import argparse
import time
import pandas as pd
from torch.utils.data import Dataset
import cv2
from multiprocessing import Pool, cpu_count
import subprocess
from math import ceil
from video_decoder import decode_video
from functools import partial

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

parser = argparse.ArgumentParser()
parser.add_argument('--cls_vid_num', type=str, default='400_5',)
parser.add_argument('--dataset', type=str, default='k400_train',)
parser.add_argument('--crop_size', type=int, default=224, help='center crop size',)
parser.add_argument('--batch_size', '-b', type=int, nargs='+', default=[8,], help='batch size',)
parser.add_argument('--workers', '-j', type=int, nargs='+', default=[8,], help='num workers',)
parser.add_argument('--backend', type=str, nargs='+', default=['pyav', 'decord', 'torchvision', ],
        help='backend type',)
parser.add_argument('--epochs', '-e', type=int, default=3, help='total epochs',)
parser.add_argument('--sampling_rate', type=int, nargs='+', default=[4], help='dataset length',)
parser.add_argument('--frames', type=int, nargs='+', default=[8, ], help='dataset length',)
parser.add_argument('--out_path', type=str, default='output/h264_speed',
        help='out path',)
parser.add_argument('--pin_memory', action='store_true', help='pin memory for dataloader')

args = parser.parse_args()

# extract videos first
def basic_func(vid_path, out_dir, sample_frames,  short_size, codec, crf):
    vid_name = vid_path.split('/')[-1].strip()
    out_path = os.path.join(out_dir, vid_name)
    if not os.path.exists(out_path) or not cv2.VideoCapture(out_path).isOpened():
        vid = cv2.VideoCapture(vid_path)
        height, width, frames, fps = vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_FRAME_WIDTH) , vid.get(cv2.CAP_PROP_FRAME_COUNT), vid.get(cv2.CAP_PROP_FPS)
        target_fps = ceil(fps * sample_frames / frames ) 
        long_size = ceil(short_size * max(height, width) / min(height, width) / 2) * 2
        new_height, new_width = (long_size, short_size) if height > width else (short_size, long_size)
        h,w,x,y = short_size, short_size, int(new_width - short_size/2), int(new_height - short_size/2)
        full_str =  f' -vf fps={target_fps},scale={new_width}:{new_height},crop={h}:{w}:{x}:{y} '
        # full_str =  f'-vf fps={target_fps},scale={new_width}:{new_height}'
        full_cmd =  f'ffmpeg -hide_banner -loglevel error -i {vid_path} {full_str} \
                        -c:v libx{codec} -crf {crf} {out_path} -y'
        subprocess.run(full_cmd.split())


def create_dataset(dataset, cls_vid_num, sample_frames):
    short_size, codec, crf = 256, 264, 15
    home = os.path.expanduser('~') 
    out_dir = '{home}/data/kinetics_vid/{dataset}_h{codec}_{short_size}_crf{crf}_F{frames}'.format(
        home=home, dataset=dataset, codec=codec, short_size=short_size, crf=crf, frames=sample_frames)
    os.makedirs(out_dir, exist_ok=True)    

    csv_path = '{home}/nerv_projects/OneNeRV/data/dataset_meta/meta/{dataset}_full.csv'.format(
        home=home, dataset=dataset)
    csv_data = pd.read_csv(csv_path)

    cls_num, vid_num = [int(float(x)) for x in cls_vid_num.split('_')]
    # select the top VID_NUM videos for each class
    vid_list = csv_data.sort_values(['label', 'path']).groupby('label').head(vid_num)   
    # select the top CLS_NUM classes for the dataset
    new_csv_data = pd.concat([group for _,group in vid_list.groupby('label')][:cls_num])
    vid_path_list = new_csv_data['path'].tolist()
    count = cpu_count()
    with Pool(processes=count) as p:
        p.map(partial(basic_func, out_dir=out_dir, sample_frames=sample_frames, short_size=short_size, codec=codec, crf=crf), tqdm(vid_path_list, desc='Creating dataset'))       
    
    # import pdb; pdb.set_trace; from IPython import embed; embed() 
    source_base_name = os.path.dirname(vid_path_list[0])
    new_vid_pathes = [x.replace(source_base_name, out_dir) for x in new_csv_data['path']]
    return new_vid_pathes


# build dataloader and evaluate its speed
class VideoDataSet(Dataset):
    def __init__(self, vid_path_list, backend, num_frames):
        self.vid_path_list = vid_path_list
        self.backend = backend
        self.frames_num = num_frames
        # import pdb; pdb.set_trace; from IPython import embed; embed()    

    def __len__(self):
        return len(self.vid_path_list)

    def __getitem__(self, idx):
        try: 
            frames = decode_video(self.vid_path_list[idx], backend=self.backend, num_frames=self.frames_num) 
            return frames
        except Exception as e:
            print(e)


frames_list, mode_list, cls_vid_num_list, time_list, fps_list = [], [], [], [], []
epoch_list, backend_list, batch_list, workers_list = [], [], [], []
def add_results(num_frames, workers, mode, backend, batch_size, cls_vid_num=args.cls_vid_num, epochs=args.epochs):
    frames_list.append(num_frames)
    batch_list.append(batch_size)
    workers_list.append(workers)
    mode_list.append(mode)
    backend_list.append(backend)
    cls_vid_num_list.append(cls_vid_num)
    epoch_list.append(epochs)

for num_frames in args.frames:
    vid_path_list = create_dataset(args.dataset, args.cls_vid_num, num_frames)
    for backend in args.backend:
        # Evaluate decoding speed
        add_results(num_frames, 0, 'decoding', backend, 1)
        # Evaluate dataloader speed
        for batch_size in args.batch_size:
            for workers in args.workers:
                add_results(num_frames, 0, 'decoding', backend, 1)

                # frames_list.append(num_frames)
                # cls_vid_num_list.append(args.cls_vid_num)
                # epoch_list.append(args.epochs)
                # batch_list.append(batch_size)
                # workers_list.append(workers)
                # mode_list.append('dataloader')
                # backend_list.append(backend)
                # import pdb; pdb.set_trace; from IPython import embed; embed()    
                full_dataset = VideoDataSet(vid_path_list, backend, num_frames)
                train_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=args.pin_memory, sampler=None, drop_last=False)
                epoch_time_list, epoch_fps_list = [], []
                i = 1
                for epoch in range(args.epochs):
                    start = time.time()
                    pbar = tqdm(train_dataloader)
                    for batch in pbar:
                        # batch = batch.cuda()
                        vps = batch_size * i / (time.time() - start)
                        i += 1
                        pbar.set_description(f'{backend}: VPS={vps:.1f}')
                    total_time = time.time() - start
                    epoch_time_list.append(total_time/60)
                    epoch_fps_list.append(len(full_dataset)/total_time)
                fps_list.append(sum(epoch_fps_list)/len(epoch_fps_list))
                time_list.append(sum(epoch_time_list))

result_df = pd.DataFrame.from_dict({'frame': frames_list, 'cls_vid_num': cls_vid_num_list, 'epoch': epoch_list,
    'patch': patch_list, 'time (min)': time_list, 'FPS': fps_list, 'backend': backend_list})    
# print(result_df)
time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
result_df.to_csv(f'{args.out_path}/result_{time_str}.csv')
