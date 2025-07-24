import cv2
import os
import glob
from tqdm import tqdm
import pandas as pd
import subprocess
import argparse
from multiprocessing import Pool, cpu_count
import os
from math import ceil

def work_list(cmd_list):
    for cmd in cmd_list:
        subprocess.run(cmd.split())
    
parser = argparse.ArgumentParser(description='Generate resized videos')
parser.add_argument('--short_side', '-s', type=int, nargs='+', default=[240,], help='short side list')
args = parser.parse_args()

home = os.path.expanduser('~') 
csv_path = '/data/home/haochen1/nerv_projects/OneNeRV/data/dataset_meta/meta/{dataset}_{split}.csv'
out_dir = '{home}/data/kinetics_samples/{dataset}_{split}_h{codec}_{short_size}_crf{crf}_F{frames}'

cmd_list = []
short_size, sample_frames, codec, crf, dataset, split = 256, 64, 264, 23, 'k400', 'train'
path_list = pd.read_csv(csv_path.format(dataset=dataset, split=split))['path'].tolist()
# for dataset in ['k400', 'k700'][:1]:
#     for split in ['train', 'val']:
#         path_list = pd.read_csv(csv_path.format(dataset=dataset, split=split))['path'].tolist()
        # for codec in [264, 265][:1]:
        #     for crf in [23, ]:
        #         for vid_path in tqdm(path_list):
        #             vid_name = vid_path.split('/')[-1].strip()
        #             out_path = os.path.join(out_dir.format(home=home, dataset=dataset, split=split, codec=codec, crf=crf, short_size=short_size, frames=sample_frames), vid_name, )
        #             if not os.path.exists(out_path) or not cv2.VideoCapture(out_path).isOpened():
        #                 vid = cv2.VideoCapture(vid_path)
        #                 height, width, frames, fps = vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_FRAME_WIDTH) , vid.get(cv2.CAP_PROP_FRAME_COUNT), vid.get(cv2.CAP_PROP_FPS)
        #                 # import pdb; pdb.set_trace; from IPython import embed; embed() 
        #                 target_fps = ceil(fps * sample_frames / frames ) 
        #                 long_size = ceil(short_size * max(height, width) / min(height, width) / 2) * 2
        #                 new_height, new_width = (long_size, short_size) if height > width else (short_size, long_size)
        #                 # h,w,x,y = short_size, short_size, int(new_width - short_size/2), int(new_height - short_size/2)
        #                 # full_str =  f' -vf scale={new_width}:{new_height},crop={h}:{w}:{x}:{y} '
        #                 full_str =  f'-vf fps={target_fps},scale={new_width}:{new_height}'
        #                 full_cmd =  f'ffmpeg -hide_banner -loglevel error -i {vid_path} {full_str} \
        #                                 -c:v libx{codec} -crf {crf} {out_path} -y'
        #                 cmd_list.append(full_cmd)
        #                 parent_dir = os.path.dirname(out_path)
        #                 if not os.path.exists(parent_dir):
        #                     os.makedirs(parent_dir)

                # print(full_cmd)
                # subprocess.run(full_cmd.split())

def work(vid_path):
    vid_name = vid_path.split('/')[-1].strip()
    out_path = os.path.join(out_dir.format(home=home, dataset=dataset, split=split, codec=codec, crf=crf, short_size=short_size, frames=sample_frames), vid_name, )
    if not os.path.exists(out_path) or not cv2.VideoCapture(out_path).isOpened():
        vid = cv2.VideoCapture(vid_path)
        height, width, frames, fps = vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_FRAME_WIDTH) , vid.get(cv2.CAP_PROP_FRAME_COUNT), vid.get(cv2.CAP_PROP_FPS)
        # import pdb; pdb.set_trace; from IPython import embed; embed() 
        target_fps = ceil(fps * sample_frames / frames ) 
        long_size = ceil(short_size * max(height, width) / min(height, width) / 2) * 2
        new_height, new_width = (long_size, short_size) if height > width else (short_size, long_size)
        # h,w,x,y = short_size, short_size, int(new_width - short_size/2), int(new_height - short_size/2)
        # full_str =  f' -vf scale={new_width}:{new_height},crop={h}:{w}:{x}:{y} '
        full_str =  f'-vf fps={target_fps},scale={new_width}:{new_height}'
        full_cmd =  f'ffmpeg -hide_banner -loglevel error -i {vid_path} {full_str} \
                        -c:v libx{codec} -crf {crf} {out_path} -y'
        # cmd_list.append(full_cmd)
        parent_dir = os.path.dirname(out_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)    
        subprocess.run(full_cmd.split())

count = cpu_count()
with Pool(processes=count) as p:
    p.map(work, path_list)    

# cmd_length = len(cmd_list)
# for i in tqdm(range(cmd_length//count)):
#     start_id, end_id = count * i, min(count * (i+1), cmd_length)
#     with Pool(processes=count) as p:
#         p.map(work, cmd_list[start_id:end_id])   


