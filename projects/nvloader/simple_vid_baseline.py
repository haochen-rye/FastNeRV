import torch
import os
from tqdm import tqdm
import argparse
import time
import pandas as pd
from torch.utils.data import Dataset
import glob
from multiprocessing import Pool, cpu_count
from torchvision.io import write_video
from video_decoder import decode_video
from functools import partial
from torchvision.utils import save_image     
import subprocess

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# export CUDA_DEVICE_MAX_CONNECTIONS=2
# time ~/local/ffmpeg-n4.4-latest-linux64-gpl-4.4/bin/ffmpeg  -hwaccel nvdec -c:v h264_cuvid -hwaccel_output_format cuda   -i try.mp4 -benchmark   -f null -

parser = argparse.ArgumentParser()
parser.add_argument('--vid_num', '-n', type=float, default=2e4,)
parser.add_argument('--filter', type=str, nargs='+', default=['crf', 'libx264'])
parser.add_argument('--backend', type=str, nargs='+', default=['decord', 'pyav', 'torchvision', 'cuda'][:1],  help='backend type',)
parser.add_argument('--epochs', '-e', type=int, default=1, help='total epochs',)
parser.add_argument('--modes', type=str, nargs='+', default=['decode', 'load'][:1], help='evaluate modes',)
parser.add_argument('--vid_dir', type=str, default='/fsx/haochen1/data/compress_vid/k400_val_resize',)
parser.add_argument('--batch_size', '-b', type=int, nargs='+', default=[8,], help='batch size',)
parser.add_argument('--workers', '-j', type=int, nargs='+', default=[8,], help='num workers',)
parser.add_argument('--dump_videos', '-v', type=int, default=0, help='video num to dump',)
parser.add_argument('--out_path', type=str, default='outputs/nvloader_results/video', help='out path',)
parser.add_argument('--dump_path', type=str, default='~/data/kinetics_vid_evenly/h264_1020', help='dir path to dump videos',)
parser.add_argument('--pin_memory', action='store_true', help='pin memory for dataloader')
parser.add_argument('--num_frames', type=int, default=8, help='video frame num',)

args = parser.parse_args()

# build dataloader and evaluate its speed
class VideoDataSet(Dataset):
    def __init__(self, vid_path_list, backend, num_frames):
        self.vid_path_list = vid_path_list
        self.backend = backend
        self.frames_num = num_frames

    def __len__(self):
        return len(self.vid_path_list)

    def __getitem__(self, idx):
        frames = decode_video(self.vid_path_list[idx], backend=self.backend, num_frames=self.frames_num) 
        return frames

# add video decoding/dataloading results
frames_list, mode_list, vid_num_list, codec_list, time_list, fps_list, avg_vid_size_list = [], [], [], [], [], [], []
epoch_list, backend_list, batch_list, workers_list = [], [], [], []
def add_results(num_frames, mode, backend, fps, total_time, v_codec_str, avg_vid_size,
            vid_num=args.vid_num, epochs=args.epochs):
    frames_list.append(num_frames)
    mode_list.append(mode)
    backend_list.append(backend)
    avg_vid_size_list.append(round(avg_vid_size, 3))
    vid_num_list.append(vid_num)
    epoch_list.append(epochs)
    fps_list.append(round(fps, 1))
    time_list.append(round(total_time,2))
    codec_list.append(v_codec_str)

# iterate over video list
args.dump_path = os.path.expanduser(args.dump_path) + '_'.join(args.filter)
os.makedirs(args.dump_path, exist_ok=True)
def iterate_loading(iterate_item, backend, mode, batch_size, total_len, v_codec_str, dec_fun=None,):
    epoch_time_list, epoch_fps_list, i, dump_video_num = [], [], 1, 0
    for epoch in range(args.epochs):
        start = time.time()
        pbar = tqdm(iterate_item)
        for batch in pbar:
            if dec_fun != None:
                raw_video_name = os.path.split(batch)[1]
                out_vid_path = os.path.join(args.dump_path, raw_video_name)
                out_jpeg_path = out_vid_path.replace('.mp4', '.jpeg')
                batch = dec_fun(batch)
                if dump_video_num < args.dump_videos and not os.path.isfile(out_jpeg_path):
                    save_image(batch.permute(0,-1,1,2).float() / 255., out_jpeg_path)
                    write_video(out_vid_path, batch, fps=2)
                # import pdb; pdb.set_trace; from IPython import embed; embed() 
            if torch.cuda.is_available():
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
def create_dataset(vid_dir, vid_num=int(args.vid_num)):
    vid_pathes_dict = {}
    full_vid_dir = os.path.expanduser(vid_dir)
    for sub_dir in sorted(os.listdir(full_vid_dir)):
        if sum([x in sub_dir for x in args.filter]) == len(args.filter):
            vid_pathes = glob.glob(os.path.join(full_vid_dir, sub_dir, '*.mp4'))
            vid_pathes_dict[sub_dir] = sorted(vid_pathes)[:vid_num]
    return vid_pathes_dict


vid_path_dict = create_dataset(args.vid_dir)
num_frames = args.num_frames
for v_codec_str, vid_path_list in vid_path_dict.items():
    vid_size_list = [os.path.getsize(x)/1e3 for x in vid_path_list]
    avg_vid_size = sum(vid_size_list) / len(vid_size_list)
    for backend in args.backend:
        if 'av1' in v_codec_str and backend=='decord':
            continue

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
                        num_workers=workers, )
                    avg_vps, total_time = iterate_loading(train_dataloader, backend, 'dataloader', batch_size,
                            len(vid_path_list), v_codec_str)
                    add_results(num_frames, f'dataloader_j{workers}_b{batch_size}', backend, avg_vps, total_time, v_codec_str, avg_vid_size)

        if 'no' in args.modes:
            base_vid_dir = '_'.join(args.vid_dir.split('/')[-2:])
            add_results(num_frames, base_vid_dir, 'no', 0, 0, v_codec_str, avg_vid_size)

        # dump results to dataframe
        result_df = pd.DataFrame.from_dict({'frame': frames_list, 'vid_num': vid_num_list, 'epoch': epoch_list,
            'mode': mode_list, 'time (min)': time_list, 'codec': codec_list, 'avg vid (KB)': avg_vid_size_list, 'backend': backend_list, 'FPS': fps_list, })    
        print(result_df)

time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
exp_name = os.path.split(args.vid_dir)[-1] + f'_e{args.epochs}_'
exp_name += '_'.join(args.filter + args.backend + args.modes) 
out_path = f'{args.out_path}/result_{time_str}_{exp_name}.csv'
result_df.to_csv(out_path)
print(f'dumped results to {out_path}')


# Get CPU model name using the 'lscpu' command (Linux specific)
def get_cpu_model_linux():
    try:
        output = subprocess.check_output(['lscpu']).decode()
        for line in output.splitlines():
            if 'Model name' in line:
                return line.split(":")[1].strip()
    except Exception as e:
        return "Error: " + str(e)
    return "Model name not found"
