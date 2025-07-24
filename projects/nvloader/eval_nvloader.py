import torch
import os
from tqdm import tqdm
import argparse
import time
import pandas as pd
import glob
import pandas as pd
from torchvision.io import write_video
from torchvision.utils import save_image     
from dahuffman import HuffmanCodec
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--vid_num', '-n', type=float, default=1e4,)
parser.add_argument('--num_frames', type=int, default=8, help='video frame num',)
parser.add_argument('--epochs', '-e', type=int, default=1, help='total epochs',)
parser.add_argument('--dump_videos', '-v', type=int, default=0, help='video num to dump',)
parser.add_argument('--nerv_dir', type=str, default='/fsx/haochen1/data/compress_vid/k400_val_crop/nerv_b8', help='dir path for nerv weights',)
parser.add_argument('--dump_path', type=str, default='~/data/kinetics_vid_evenly/nerv_1020a', help='dir path to dump videos',)
parser.add_argument('--out_path', type=str, default='outputs/nvloader_results/nerv',
        help='out path',)

args = parser.parse_args()

def compute_total_bits(quant_ckt):
    quant_v_list, tmin_scale_len = [], 0
    for k, layer_wt in quant_ckt.items():
        quant_v_list.extend(layer_wt['quant'].flatten().tolist())
        tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

    # get the element name and its frequency
    unique, counts = np.unique(quant_v_list, return_counts=True)
    num_freq = dict(zip(unique, counts))

    # generating HuffmanCoding table
    codec = HuffmanCodec.from_data(quant_v_list)
    sym_bit_dict = {}
    for k, v in codec.get_code_table().items():
        sym_bit_dict[k] = v[0]

    # total bits for quantized embed + model weights
    total_bits = 0
    for num, freq in num_freq.items():
        total_bits += freq * sym_bit_dict[num]
    bits_per_param = total_bits / len(quant_v_list)
    
    # including the overhead for min and scale storage, 
    total_bits += tmin_scale_len * 16               #(16bits for float16)
    full_bits_per_param = total_bits / len(quant_v_list)

    out_dict = {'total_bits': total_bits, 'bits_per_param': bits_per_param, 'full_bits_per_param': full_bits_per_param}
    return out_dict

def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'], quant_t['scale']
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t.float()

nerv_list, nerv_path_list, nerv_num, total_bits = [], [], 0, 0
nerv_dir = os.path.expanduser(args.nerv_dir)
for nerv_file in tqdm(sorted(os.listdir(nerv_dir))):
    if 'pth' in nerv_file and 'modulate_nerv' not in nerv_file:
        full_nerv_path = os.path.join(nerv_dir, nerv_file)
        nerv_path_list.append(full_nerv_path)
        nerv_weights = torch.load(full_nerv_path, map_location='cpu')
        total_bits += compute_total_bits(nerv_weights)['total_bits']
        dequant_nerv_weights = {k:dequant_tensor(v) for k,v in nerv_weights.items()}
        nerv_list.append(dequant_nerv_weights)
        nerv_num += 1
    if nerv_num >= int(args.vid_num):
        break

modulate_nerv_list = glob.glob(os.path.join(nerv_dir, 'modulate_nerv*.pth'))
dump_vid_path, dump_vid_num = os.path.expanduser(args.dump_path), 0
os.makedirs(dump_vid_path, exist_ok=True)

vid_size_list = [os.path.getsize(x)/1e6 for x in nerv_path_list]
avg_vid_size = sum(vid_size_list) / len(vid_size_list)
actual_ckt_size = total_bits / 8 / 1e6 / len(vid_size_list)

frames_list, device_list, load_batch_list, num_list, fps_list, epoch_list, avg_size_list, ckt_size_list = [[] for _ in range(8)]
def add_results(num_frames, device, batch_size,fps, epochs=args.epochs, vid_num=args.vid_num):
    frames_list.append(num_frames)
    device_list.append(device)
    load_batch_list.append(batch_size)
    fps_list.append(fps)
    num_list.append(vid_num)
    epoch_list.append(epochs)
    avg_size_list.append(avg_vid_size)
    ckt_size_list.append(actual_ckt_size)


for modulate_nerv in sorted(modulate_nerv_list):
    batch_str, device_str = modulate_nerv.split('_')[-2:]
    batch, device = int(batch_str.replace('batch', '')), device_str.replace('.pth', '')
    if device == 'cpu' and batch != 1:
        continue
    map_device = 'cpu' if device == 'cpu' else 'cuda'
    device_nerv_list = [{k:v.to(map_device) for k,v in x.items()} for x in nerv_list]
    if batch != 1:
        stack_nerv_list, start_idx = [], 0
        for i in range(len(device_nerv_list) // batch):
            cur_batch = device_nerv_list[start_idx:start_idx+batch]
            cur_batch_nerv = {k:torch.cat([x[k] for x in cur_batch]) for k in nerv_list[0]}
            stack_nerv_list.append(cur_batch_nerv)
            start_idx += batch
        device_nerv_list = stack_nerv_list

    vid_decoder = torch.jit.load(modulate_nerv, map_location=map_device)
    epoch_vps_list = []
    for epoch in range(args.epochs):
        start = time.time()
        pbar = tqdm(device_nerv_list)
        for i, nerv_weights in enumerate(pbar):                    
            x = vid_decoder(nerv_weights)
            if batch==1 and dump_vid_num <= args.dump_videos:
                raw_vid_name = os.path.split(nerv_path_list[i])[-1].replace('.pth', '.mp4')
                out_vid_path = os.path.join(dump_vid_path, raw_vid_name)
                out_img_path = out_vid_path.replace('.mp4', '.jpeg')
                if not os.path.isfile(out_vid_path):
                    write_video(out_vid_path, x.cpu()[:,0].permute(0,2,3,1)*255, fps=2)        # t,h,w,c
                if not os.path.isfile(out_img_path):
                    save_image(x.cpu()[:,0], out_img_path)
                # import pdb; pdb.set_trace; from IPython import embed; embed()    
                dump_vid_num += 1
            vps = batch * (i+1) / (time.time() - start)
            pbar.set_description(f'NeRV w/ {device} batch {batch} e{epoch}: VPS={vps:.1f}')
        epoch_vps_list.append(vps)
    add_results(args.num_frames, device, batch, sum(epoch_vps_list)/len(epoch_vps_list))
    result_df = pd.DataFrame.from_dict({'frame': frames_list, 'vid_Num': num_list, 'epoch': epoch_list,
        'device':device_list, 'batch': load_batch_list, 'FPS': fps_list, 'Avg vid size (MB)': avg_size_list, 
        'Actual ckt size (MB)': actual_ckt_size})    
    print(result_df)


time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
exp_name = '_'.join(args.nerv_dir.split('/')[-2:])
out_path = f'{args.out_path}/result_N{args.vid_num}_{time_str}_{exp_name}.csv'
result_df.to_csv(out_path)
print(result_df, f'\n dumped to {out_path}')
