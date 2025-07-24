import os
import pandas as pd
import shutil
from math import ceil
import subprocess

code_dir = os.path.expanduser('~/nerv_projects/OneNeRV/projects/nerv_baseline')
script_dir = f'{code_dir}/scripts'

std_cmd = "cd {code_dir}\n\
python train_hypoconv.py -e {epoch} --vid_path '{vid_path}' --vid {vid} \
--crop_size {crop_size} --hid_dim {hid_dim} --strds {strds} --ks {ks} --pe_dim {pe_dim} \
--pe_sigma {pe_sigma} --act {act} --out_bias {out_bias} -b {batch_size} --lr {lr} \
--eval_freq {eval_epoch} {dump_str} --frames {frames} --outf {outf} "


prefix, gpu_num, partition, cpu_num = 'base0', 1, 'ai', 0
date, cls_vid_num, input_size = f'1020', '400_3', 256
root_dir = os.path.expanduser('~/nerv_projects/OneNeRV/data/dataset_meta/meta')
csv_list = ['k400_val_full.csv',  'sth-v2-validation-h264.csv',  'ucf101-test.csv']

cls_num, vid_num = [int(float(x)) for x in cls_vid_num.split('_')]
vid_list = []
for csv_file in csv_list:
    csv_data = pd.read_csv(os.path.join(root_dir, csv_file))
    sub_csv_data = csv_data.sort_values(['label', 'path']).groupby('label').head(vid_num)   
    # select the top CLS_NUM classes for the dataset
    sub_csv_data = pd.concat([group for _,group in sub_csv_data.groupby('label')][:cls_num])
    for id, vid_path in enumerate(sub_csv_data['path'].tolist()):
        dataset = csv_file.split('.')[0]
        vid = f'{dataset}_{id+1:03d}'
        vid_list.append((vid_path, vid, dataset))

crop_size, hid_dim, strds, ks, pe_dim = 256, 16, '4 4 4 4', '1 3', 16
pe_sigma, act, out_bias, batch_size, base_lr = 1024, 'gelu', 'tanh', 2, 5e-4
epoch, eval_epoch, dump_str, frames = 300, 50, '', 8

setup_list = []
epoch, frames = 600, 8
epoch_dict = {4: [600, 900, 1200, 1500], 8: [300, 600, 900],}
for frames in [4, 8]:
    for epoch in epoch_dict[frames]:
        setup_list.append((epoch, frames))    

cmd_list = []
for vid_path, vid, dataset in vid_list:
    for epoch, frames in setup_list:
        cur_cmd = std_cmd.format(code_dir=code_dir, epoch=epoch, vid_path=vid_path, vid=vid,
            crop_size=crop_size, hid_dim=hid_dim, strds=strds, ks=ks, pe_dim=pe_dim,
            pe_sigma=pe_sigma, act=act, out_bias=out_bias, batch_size=batch_size, lr=base_lr*batch_size,
            eval_epoch=eval_epoch, dump_str=dump_str, outf=f'{date}_{cls_vid_num}/{dataset}_e{epoch}_N{frames}', frames=frames)
        if cur_cmd not in cmd_list:
            cmd_list.append(cur_cmd)    

os.makedirs(script_dir, exist_ok=True)
os.chdir(script_dir)
if os.path.isdir(prefix):
    shutil.rmtree(prefix)
os.makedirs(prefix, exist_ok=True)

start_idx, cmd_num = 0, 675
script_num = ceil(len(cmd_list) / cmd_num )
print(f'Total cmd_num: {len(cmd_list)}, script_num: {script_num}')
submit_cmd_list = []
for i in range(script_num):
    end_idx = start_idx + cmd_num
    temp_file = os.path.join(prefix, f'{i+1:02d}')
    if i+1 >= 0:
        print(f'Creating {temp_file}...')
        cur_cmd_list = cmd_list[start_idx:end_idx]
        with open(temp_file, 'w') as f:
            f.write('\n'.join(cur_cmd_list))
        submit_command = f'submit {temp_file} --nog {gpu_num} --no_output -p {partition} --cpu {cpu_num}'
        submit_cmd_list.append(submit_command)
    start_idx = end_idx

for submit_cmd in submit_cmd_list:
    print(f'execute command: {submit_cmd}')
    subprocess.run(submit_cmd.split())                   