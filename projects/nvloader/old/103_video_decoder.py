import torch
from random import randint
import torchvision.io as io
import os
import cv2
from multiprocessing import Pool, cpu_count
import subprocess
from math import ceil
from tqdm import tqdm
import pandas as pd
import av
import numpy as np
import torch.nn.functional as F

from pynviread import NvidiaReader
import pynviread.examples as ex

import sys
sys.path.insert(0, '/data/home/haochen1/local/decord/python')

os.environ['DECORD_EOF_RETRY_MAX'] = '204800'

import decord
from decord import VideoReader
decord.bridge.set_bridge('torch')
from decord import cpu, gpu

def temporal_sampling(video_duration, sampling_rate, num_frames):
    if sampling_rate == -1:
        random_starting_idx, end_idx = 0, video_duration-1
    else:
        clip_duration = sampling_rate * (num_frames - 1)
        random_starting_idx = randint(0, max(video_duration - clip_duration, 0))
        end_idx = min(random_starting_idx + clip_duration, video_duration-1)
    frame_idx_list = torch.linspace(random_starting_idx, end_idx, num_frames).round().int().tolist()
    return frame_idx_list


def decord_decode(
    path_to_vid,
    num_frames,    
    sampling_rate,
    ):
    vr = VideoReader(path_to_vid)
    # video_duration = len(vr)
    # frame_idx_list = temporal_sampling(video_duration, sampling_rate, num_frames)
    # frames = decord_vr.get_batch(frame_idx_list).asnumpy()
    # v_frames = torch.from_numpy(frames)
    v_frames = []
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        v_frames.append(vr[i])

    return torch.stack(v_frames, 0)         # t,h,w,c


def torchvision_decode(
    path_to_vid,
    num_frames,
    sampling_rate,
):
    # timestamp_list = io.read_video_timestamps(path_to_vid)[0]
    # frame_idx_list = temporal_sampling(len(timestamp_list), sampling_rate, num_frames)
    # start_end_pts = [timestamp_list[frame_idx_list[x]] for x in [0, -1]]
    # v_frames, _, _ = io.read_video(path_to_vid, start_pts=start_end_pts[0], 
    #             end_pts=start_end_pts[-1], pts_unit='pts')  
    # import pdb; pdb.set_trace; from IPython import embed; embed()  
    v_frames, _, _ = io.read_video(path_to_vid)   # t,h,w,c
    return v_frames


def pyav_decode(
    path_to_vid,
    num_frames,
    sampling_rate,
):
    v_frames = []
    container = av.open(path_to_vid)
    container.streams.video[0].thread_type = "AUTO"
    for frame in container.decode(video=0):
        v_frames.append(frame.to_rgb().to_ndarray())
    video_data = torch.from_numpy(np.stack(v_frames, 0))  # t,h,w,c
    container.close()

    return video_data

def cuda_decode(
    path_to_vid,
    num_frames,
    sampling_rate,
):
    # with NvidiaReader(src=path_to_vid, verbose=False, output_options=['-loglevel', 'error', '-hide_banner']) as reader:
    #     video = reader.read()
    # video = torch.from_numpy(video)             # t,h,w,c

    # import pdb; pdb.set_trace; from IPython import embed; embed() 
    vr = VideoReader(path_to_vid, ctx=gpu(0))
    v_frames = []
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        v_frames.append(vr[i])

    video = torch.stack(v_frames, 0)         # t,h,w,c
    
    return video

def decode_video(
    path_to_vid,
    num_frames=None,
    sampling_rate=None,
    backend='pyav'
):
    if backend == 'pyav':
        decode_func = pyav_decode
    elif backend == 'torchvision':
        decode_func = torchvision_decode
    elif backend == 'decord':
        decode_func = decord_decode
    elif backend == 'cuda':
        decode_func = cuda_decode
    else:
        raise NotImplementedError(f'Unknown decoding backend {backend}')

    try:
        video = decode_func(path_to_vid, sampling_rate, num_frames)
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        video = torch.ones(num_frames,256,256,3)
        # return None

    # remove reduant frames
    if video.size(0) > num_frames:
        video = video[:num_frames]
    elif video.size(0) == num_frames:
        video = video
    elif video.size(0) < num_frames:
        pad_size = num_frames - video.size(0)
        video = (video.transpose(0,-1).contiguous() /255.).to(torch.float)        
        video = F.pad(video, (0,pad_size,0,0,0,0), mode='replicate').transpose(0,-1).contiguous()
        video = (video*255.).to(torch.uint8)
    return video



# extract videos first
def create_dataset(dataset, cls_vid_num):
    short_size, sample_frames, codec, crf = 256, 8, 264, 15
    home = os.path.expanduser('~') 
    out_dir = '{home}/data/kinetics_samples/{dataset}_h{codec}_{short_size}_crf{crf}_F{frames}_{cls_vid_num}'.format(
        home=home, dataset=dataset, codec=codec, short_size=short_size, crf=crf, frames=sample_frames, cls_vid_num=cls_vid_num)
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

    def work(vid_path):
        vid_name = vid_path.split('/')[-1].strip()
        out_path = os.path.join(out_dir, vid_name)
        if not os.path.exists(out_path) or not cv2.VideoCapture(out_path).isOpened():
            vid = cv2.VideoCapture(vid_path)
            height, width, frames, fps = vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_FRAME_WIDTH) , vid.get(cv2.CAP_PROP_FRAME_COUNT), vid.get(cv2.CAP_PROP_FPS)
            # import pdb; pdb.set_trace; from IPython import embed; embed() 
            target_fps = ceil(fps * sample_frames / frames ) 
            long_size = ceil(short_size * max(height, width) / min(height, width) / 2) * 2
            new_height, new_width = (long_size, short_size) if height > width else (short_size, long_size)
            h,w,x,y = short_size, short_size, int(new_width - short_size/2), int(new_height - short_size/2)
            full_str =  f' -vf fps={target_fps},scale={new_width}:{new_height},crop={h}:{w}:{x}:{y} '
            # full_str =  f'-vf fps={target_fps},scale={new_width}:{new_height}'
            full_cmd =  f'ffmpeg -hide_banner -loglevel error -i {vid_path} {full_str} \
                            -c:v libx{codec} -crf {crf} {out_path} -y'
            subprocess.run(full_cmd.split())

    count = cpu_count()
    with Pool(processes=count) as p:
        p.map(work, tqdm(vid_path_list, desc='Creating dataset'))   
    source_base_name = os.path.dirname(vid_path_list[0])
    new_vid_pathes = new_csv_data['path'].replace(source_base_name, out_dir)

    return new_vid_pathes
