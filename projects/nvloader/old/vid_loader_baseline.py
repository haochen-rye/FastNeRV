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
from torchvision.io import write_video
from torchvision.utils import save_image     

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

parser = argparse.ArgumentParser()
parser.add_argument('--cls_vid_num', '-n', type=str, default='400_5',)
parser.add_argument('--dataset', type=str, default='k400_train',)
parser.add_argument('--short_size', type=int, default=256, help='shorter video size',)
parser.add_argument('--batch_size', '-b', type=int, nargs='+', default=[8,], help='batch size',)
parser.add_argument('--workers', '-j', type=int, nargs='+', default=[8,], help='num workers',)
parser.add_argument('--codecs', type=str, nargs='+', default=['libx264', 'mpeg4', 'libvpx-vp9', ])
parser.add_argument('--backend', type=str, nargs='+', default=['decord', 'pyav', 'torchvision', 'cuda'][:1],
        help='backend type',)
parser.add_argument('--epochs', '-e', type=int, default=1, help='total epochs',)
parser.add_argument('--sampling_rate', type=int, nargs='+', default=[4], help='dataset length',)
parser.add_argument('--crfs', type=int, nargs='+', default=[50], help='crf for ffmpeg video encoding',)
parser.add_argument('--presets', type=str, nargs='+', default=['ultrafast'], help='preset for ffmpeg video encoding',)
parser.add_argument('--modes', type=str, nargs='+', default=['decode', 'load'][:0], help='evaluate modes',)
parser.add_argument('--frames', type=int, nargs='+', default=[8, ], help='dataset length',)
parser.add_argument('--dump_videos', '-v', type=int, default=0, help='video num to dump',)
parser.add_argument('--out_path', type=str, default='output/h264_speed',
        help='out path',)
parser.add_argument('--pin_memory', action='store_true', help='pin memory for dataloader')

args = parser.parse_args()

# extract videos with given video_code/frames/size first
def basic_func(vid_path, out_dir, sample_frames,  short_size, codec, crf, preset):
    vid_name = vid_path.split('/')[-1].strip()
    out_path = os.path.join(out_dir, vid_name)
    if not os.path.exists(out_path) or not cv2.VideoCapture(out_path).isOpened():
        vid = cv2.VideoCapture(vid_path)
        height, width, frames, fps = vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_FRAME_WIDTH) , vid.get(cv2.CAP_PROP_FRAME_COUNT), vid.get(cv2.CAP_PROP_FPS)
        long_size = ceil(short_size * max(height, width) / min(height, width) / 2) * 2
        new_height, new_width = (long_size, short_size) if height > width else (short_size, long_size)
        h,w,x,y = short_size, short_size, int(new_width - short_size/2), int(new_height - short_size/2)
        # change fps to select frames
        # target_fps = ceil(fps * sample_frames / frames ) 
        # full_str =  f' -vf fps={target_fps},scale={new_width}:{new_height},crop={h}:{w}:{x}:{y} '
        # select every n frames 
        time_strd = ceil(frames / sample_frames)
        sample_str = f"select=not(mod(n\,{time_strd}))"
        full_str =  f' -vf {sample_str},scale={new_width}:{new_height},crop={h}:{w}:{x}:{y} -vsync vfr'
        # full_str =  f'-vf fps={target_fps},scale={new_width}:{new_height}'
        full_cmd =  f'ffmpeg -hide_banner -loglevel error -i {vid_path} {full_str} \
                        -c:v {codec} -crf {crf} -preset {preset} {out_path} -y'
        # print(full_cmd)
        subprocess.run(full_cmd.split())


def create_dataset(dataset, cls_vid_num, sample_frames, codec, preset, crf):
    short_size = args.short_size
    home = os.path.expanduser('~') 
    csv_path = '{home}/nerv_projects/OneNeRV/data/dataset_meta/meta/{dataset}_full.csv'.format(
        home=home, dataset=dataset)
    csv_data = pd.read_csv(csv_path)

    cls_num, vid_num = [int(float(x)) for x in cls_vid_num.split('_')]
    # select the top VID_NUM videos for each class
    vid_list = csv_data.sort_values(['label', 'path']).groupby('label').head(vid_num)   
    # select the top CLS_NUM classes for the dataset
    new_csv_data = pd.concat([group for _,group in vid_list.groupby('label')][:cls_num])
    vid_path_list = new_csv_data['path'].tolist()
    out_dir = '{home}/data/kinetics_vid_evenly/{dataset}_{codec}_{preset}_crf{crf}_{short_size}_F{frames}'.format(
        home=home, dataset=dataset, codec=codec, preset=preset, short_size=short_size, crf=crf, frames=sample_frames)
    os.makedirs(out_dir, exist_ok=True)    

    with Pool(processes=cpu_count()) as p:
        p.map(partial(basic_func, out_dir=out_dir, sample_frames=sample_frames, 
            short_size=short_size, codec=codec, crf=crf, preset=preset), tqdm(vid_path_list, desc='Creating dataset'))       
    
    source_base_name = os.path.dirname(vid_path_list[0])
    new_vid_pathes = [x.replace(source_base_name, out_dir) for x in new_csv_data['path']]
    return new_vid_pathes

    # import pdb; pdb.set_trace; from IPython import embed; embed() 


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
        frames = decode_video(self.vid_path_list[idx], backend=self.backend, num_frames=self.frames_num) 
        return frames

# add video decoding/dataloading results
frames_list, mode_list, cls_vid_num_list, codec_list, time_list, fps_list, avg_vid_size_list = [], [], [], [], [], [], []
epoch_list, backend_list, batch_list, workers_list = [], [], [], []
def add_results(num_frames, mode, backend, fps, total_time, v_codec_str, avg_vid_size,
            cls_vid_num=args.cls_vid_num, epochs=args.epochs):
    frames_list.append(num_frames)
    mode_list.append(mode)
    backend_list.append(backend)
    avg_vid_size_list.append(round(avg_vid_size, 3))
    cls_vid_num_list.append(cls_vid_num)
    epoch_list.append(epochs)
    fps_list.append(round(fps, 1))
    time_list.append(round(total_time,2))
    codec_list.append(v_codec_str)

# iterate over video list
def iterate_loading(iterate_item, backend, mode, batch_size, total_len, v_codec_str, dec_fun=None,):
    epoch_time_list, epoch_fps_list, i, dump_video_num = [], [], 1, 0
    for epoch in range(args.epochs):
        start = time.time()
        pbar = tqdm(iterate_item)
        for batch in pbar:
            if dec_fun != None:
                out_jpeg_path = batch.replace('.mp4', '.jpeg')
                batch = dec_fun(batch)
                if dump_video_num <= args.dump_videos and not os.path.isfile(out_jpeg_path):
                    # import pdb; pdb.set_trace; from IPython import embed; embed() 
                    save_image(batch.permute(0,-1,1,2).float() / 255., out_jpeg_path)
                    # write_video(f'try.mp4',batch,fps=2)
            batch = batch.cuda()
            vps = batch_size * i / (time.time() - start)
            i += 1
            pbar.set_description(f'{v_codec_str} {backend} {mode}: VPS={vps:.1f}')
        total_time = time.time() - start
        epoch_time_list.append(total_time/60)
        epoch_fps_list.append(total_len/total_time)
    avg_vps, total_time = sum(epoch_fps_list)/len(epoch_fps_list), sum(epoch_time_list)
    return avg_vps, total_time
    
# import pdb; pdb.set_trace; from IPython import embed; embed()    
for num_frames in args.frames:
    for codec in args.codecs:
         for preset in args.presets:
        #  ['veryfast', 'superfast', 'ultrafast'][-1:]:
            for crf in args.crfs:
                vid_path_list = create_dataset(args.dataset, args.cls_vid_num, num_frames, codec, preset, crf)
                v_codec_str = f'{codec}_{preset}_crf{crf}'
                vid_size_list = [os.path.getsize(x)/1e6 for x in vid_path_list]
                avg_vid_size = sum(vid_size_list) / len(vid_size_list)
                for backend in args.backend:
                    if 'decode' in args.modes:
                        # Evaluate decoding speed
                        avg_vps, total_time = iterate_loading(vid_path_list, backend, 'decode', 1, len(vid_path_list), 
                                v_codec_str, partial(decode_video, num_frames=num_frames, backend=backend), )
                        add_results(num_frames, 'decoding', backend, avg_vps, total_time, v_codec_str, avg_vid_size)

                    if 'load' in args.modes:
                        # Evaluate dataloader speed
                        for batch_size in args.batch_size:
                            for workers in args.workers:
                                full_dataset = VideoDataSet(vid_path_list, backend, num_frames)
                                train_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=False, sampler=None, drop_last=False)
                                avg_vps, total_time = iterate_loading(train_dataloader, backend, 'dataloader', batch_size,
                                        len(vid_path_list), v_codec_str)
                                add_results(num_frames, f'dataloader_j{workers}_b{batch_size}', backend, avg_vps, total_time, v_codec_str, avg_vid_size)

                # dump results to dataframe
                result_df = pd.DataFrame.from_dict({'frame': frames_list, 'cls_vid_num': cls_vid_num_list, 'epoch': epoch_list,
                    'mode': mode_list, 'time (min)': time_list, 'codec': codec_list, 'avg vid (MB)': avg_vid_size_list, 'backend': backend_list, 'FPS': fps_list, })    
                print(result_df)

time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
exp_name = ','.join(args.presets) + '__crf' + ','.join([str(x) for x in args.crfs])
out_path = f'{args.out_path}/result_{time_str}_{exp_name}.csv'
result_df.to_csv(out_path)
print(f'dumped results to {out_path}')