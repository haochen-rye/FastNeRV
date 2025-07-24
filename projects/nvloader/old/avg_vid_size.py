import pandas as pd
import argparse
import os
from tqdm import tqdm
import time
import numpy as np
import glob
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', type=str, nargs='+', default='~/data/kinetics_vid/k400_train_h264_256_crf15_F8',)
parser.add_argument('--out_path', type=str, default='output/h264_size',
        help='out path',)
args = parser.parse_args()


def get_path_size(path):
    if os.path.isfile(path):
        file_size = os.path.getsize(path)
    else:
        file_size= sum([os.path.getsize(os.path.join(path,x)) for x in os.listdir(path)])
    return file_size / 1e6

dir_list = args.dir
if len(args.dir)==1:
    subdir_list = os.listdir(os.path.expanduser(args.dir[0]))
    # import pdb; pdb.set_trace; from IPython import embed; embed()       
    if '.mp4' not in subdir_list[0]:
        dir_list = [os.path.join(args.dir[0], x) for x in sorted(subdir_list)]

full_df_dict = {'dir': [], 'avg_size_MB': []}
for cur_dir in dir_list:
    root_dir = os.path.expanduser(cur_dir)
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    raw_dir = os.path.split(cur_dir)[-1]
    os.makedirs(args.out_path, exist_ok=True)
    out_file = f'{args.out_path}/stat_{time_str}_{raw_dir}.csv'
    if os.path.isdir(root_dir):
        vid_list = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
        if len(vid_list):
            count = cpu_count()
            with Pool(processes=count) as p:
                vid_size_list = p.map(get_path_size, tqdm(vid_list))    

            # vid_size_list = [get_path_size(x)/1e6 for x in vid_list]
            avg_vid_size = np.asarray(vid_size_list).mean().round(4)
            df = pd.DataFrame.from_dict({'size_list (MB)': vid_size_list, 'avg_size (MB)': avg_vid_size})
            df.to_csv(out_file)       
            full_df_dict['dir'].append(raw_dir)
            full_df_dict['avg_size_MB'].append(avg_vid_size)
    df = pd.DataFrame.from_dict(full_df_dict)
    print(df)
    df.to_csv(f'{args.out_path}/stat_{time_str}.csv')       
    # print(f'dumping to {out_file}')

