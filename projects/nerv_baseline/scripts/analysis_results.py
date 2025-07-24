#!/usr/bin/env python

import os
import glob
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default='~/nerv_projects/OneNeRV/projects/nerv_baseline/output', help='csv dir')
parser.add_argument('--dir_depth', type=str, default='2,5', help='dir depth')
parser.add_argument('--tag', '-s', type=str, default='', help='specific str for csv name')
parser.add_argument('--out_path', type=str, default='outputs/nerv_baseline',
        help='out path',)

args = parser.parse_args()

def convert_time(time_str):
    hours, minutes, seconds = time_str.split(':')
    numeric_value =  int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return numeric_value

def dump_results(out_dict, dump_csv=False):
    out_df = pd.DataFrame.from_dict(out_dict)
    print(out_df)
    if dump_csv:
        time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
        os.makedirs(args.out_path, exist_ok=True)
        out_path = f'{args.out_path}/result_{time_str}_{args.tag}.csv'
        out_df.to_csv(out_path)
        print(f'dumped results to {out_path}')

root_dir = os.path.expanduser(args.csv_path)
selected_metrics = ['Time',  'best_psnr', 'eval_psnr', 'eval_ssim', 'eval_lpips']
min_depth, max_depth = [int(x) for x in args.dir_depth.split(',')]
out_dict = {'method': [], 'vid_num': []}
for subdir in sorted(os.listdir(root_dir)):
    if len(args.tag)==0 or args.tag in subdir:
        print(subdir)
        cur_dir = os.path.join(root_dir, subdir)
        out_dict['method'].append(subdir)
        csv_files = []
        for dir_depth in range(min_depth, max_depth):
            # csv_files += glob.glob(cur_dir + '/*'*dir_depth + f'*.csv')
            csv_files += glob.glob(cur_dir + '/*'*dir_depth + f'*.csv')
        concat_df = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
        concat_df['Time'] = concat_df['Time'].map(convert_time)
        avg_values = concat_df[selected_metrics].mean()
        for metric, avg_v in zip(selected_metrics, avg_values):
            if metric not in out_dict:
                out_dict[metric] = []
            out_dict[metric].append(avg_v)
        out_dict['vid_num'].append(len(csv_files))
        dump_results(out_dict)
dump_results(out_dict, True)


# import pdb; pdb.set_trace; from IPython import embed; embed()    
