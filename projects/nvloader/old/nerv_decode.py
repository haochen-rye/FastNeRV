import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import write_video
from torchvision.utils import save_image     

def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'], quant_t['scale']
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t.float()

parser = argparse.ArgumentParser()
parser.add_argument('--vid_num', '-n', type=float, default=2e3,)
parser.add_argument('--epochs', '-e', type=int, default=1, help='total epochs',)
parser.add_argument('--dump_videos', '-v', type=int, default=100, help='video num to dump',)
parser.add_argument('--dump_path', type=str, default='~/data/kinetics_vid_evenly/nerv_928', help='dir path to dump videos',)
parser.add_argument('--out_path', type=str, default='output/nvloader_speed',
        help='out path',)

args = parser.parse_args()

base_dir = '/data/home/haochen1/nerv_projects/OneNeRV'
nerv_dir = os.path.join(base_dir, 'outputs/nerv/922_k400_train_29.9_g1_f8_s256_b32_e300_lr0.0001_S40_5/nerv_weights/modulate')
nerv_list = []
nerv_path_list = []
nerv_num = 0
for nerv_file in tqdm(os.listdir(nerv_dir)):
    if 'pth' in nerv_file and 'modulate_nerv' not in nerv_file:
        full_nerv_path = os.path.join(nerv_dir, nerv_file)
        nerv_path_list.append(full_nerv_path)
        nerv_weights = torch.load(full_nerv_path, map_location='cpu')
        dequant_nerv_weights = {k:dequant_tensor(v) for k,v in nerv_weights.items()}
        nerv_list.append(dequant_nerv_weights)
        nerv_num += 1
    if nerv_num >= int(args.vid_num):
        break

frames_list, device_list, load_batch_list, num_list, fps_list, epoch_list, avg_size_list = [[] for _ in range(7)]
def add_results(num_frames, device, batch_size,fps, epochs=args.epochs, vid_num=args.vid_num):
    frames_list.append(num_frames)
    device_list.append(device)
    load_batch_list.append(batch_size)
    fps_list.append(fps)
    num_list.append(vid_num)
    epoch_list.append(epochs)
    avg_size_list.append(avg_vid_size)

dump_vid_path, dump_vid_num = os.path.expanduser(args.dump_path), 0
os.makedirs(dump_vid_path, exist_ok=True)

vid_size_list = [os.path.getsize(x)/1e6 for x in nerv_path_list]
avg_vid_size = sum(vid_size_list) / len(vid_size_list)

for device in ['cpu', 'gpu']:
    if device == 'cpu':
        map_device = 'cpu'
        batch_list = [1]
    else:
        map_device = 'cuda'
        batch_list = [1, 32, 128, 512]
    device_nerv_list = [{k:v.to(map_device) for k,v in x.items()} for x in nerv_list]
    for idx, batch in enumerate(batch_list):
        if device != 'cpu' and batch != 1:
            stack_nerv_list = []
            start_idx, new_batch = 0, batch // batch_list[idx-1]
            for i in range(len(device_nerv_list) // new_batch):
                cur_batch = device_nerv_list[start_idx:start_idx+new_batch]
                cur_batch_nerv = {k:torch.cat([x[k] for x in cur_batch]) for k in nerv_list[0]}
                stack_nerv_list.append(cur_batch_nerv)
                start_idx += new_batch
            device_nerv_list = stack_nerv_list

        trace_model = os.path.join(nerv_dir, f'modulate_nerv_batch{batch}_{device}.pth')
        vid_decoder = torch.jit.load(trace_model, map_location=map_device)
        epoch_vps_list = []
        for epoch in range(args.epochs):
            start = time.time()
            pbar = tqdm(device_nerv_list)
            for i, nerv_weights in enumerate(pbar):                    
                x = vid_decoder(nerv_weights)
                if batch==1 and dump_vid_num <= args.dump_videos:
                    raw_vid_name = os.path.split(nerv_path_list[i])[-1].replace('.pth', '.mp4')
                    out_vid_path = os.path.join(dump_vid_path, raw_vid_name)
                    if not os.path.isfile(out_vid_path):
                        out_x = x.cpu()[:,0] 
                        write_video(out_vid_path, out_x.permute(0,2,3,1)*255, fps=2)        # t,h,w,c
                        save_image(out_x, out_vid_path.replace('.mp4', '.jpeg'))
                    dump_vid_num += 1
                vps = batch * (i+1) / (time.time() - start)
                pbar.set_description(f'NeRV w/ {device} batch {batch} e{epoch}: VPS={vps:.1f}')
            epoch_vps_list.append(vps)
        add_results(8, device, batch, sum(epoch_vps_list)/len(epoch_vps_list))

        # import pdb; pdb.set_trace; from IPython import embed; embed()    
        result_df = pd.DataFrame.from_dict({'frame': frames_list, 'vid_Num': num_list, 'epoch': epoch_list,
            'device':device_list, 'batch': load_batch_list, 'FPS': fps_list, 'Avg vid size (MB)': avg_size_list})    
        print(result_df)

time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
out_path = f'{args.out_path}/result_N{args.vid_num}_{time_str}.csv'
result_df.to_csv(out_path)
print(result_df, f'\n dumped to {out_path}')
