import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
from torch.utils.data import Dataset

def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'], quant_t['scale']
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t.float()

class VideoDataSet(Dataset):
    def __init__(self, csv_path='', num_frames=8, data_len=2000):
        # path_df=pd.read_csv(csv_path)
        # self.num_frames = num_frames
        # self.vid_path_list = path_df['vid'].tolist()[:data_len]
        nerv_dir = '/opt/hpcaas/.mounts/fs-0470ec59e2d597fd0/haochen1/one-nerv/eval/bit7_g1_f8_s256_b32_e300_lr0.0001_S400_1e4/nerv_weights/modulate'
        trace_model = os.path.join(nerv_dir, 'modulate_nerv_batch1.pth')
        self.vid_decoder = torch.jit.load(trace_model, map_location='cuda')
        self.nerv_list = []
        for nerv_file in os.listdir(nerv_dir):
            if 'pth' in nerv_file and 'modulate_nerv' not in nerv_file:
                nerv_weights = torch.load(os.path.join(nerv_dir, nerv_file), map_location='cpu')
                dequant_nerv_weights = {k:dequant_tensor(v).cuda() for k,v in nerv_weights.items()}
                self.nerv_list.append(dequant_nerv_weights)

    def dequant_nerv(self, vid_path):
        vid_name = vid_path.split('/')[-1].replace('.mp4', '')
        nerv_path = os.path.join(self.nerv_dir, f'{vid_name}.pth')
        assert os.path.exists(nerv_path)
        nerv_path = os.path.expanduser('~/data/nerv/k400_train_drop/---QUuC4vJs_000084_000094.pth')
        nerv_weights = torch.load(nerv_path, map_location='cpu')
        dequant_nerv_weights = {k:dequant_tensor(v).cuda() for k,v in nerv_weights.items()}
        return dequant_nerv_weights

    def __len__(self):
        return len(self.nerv_list)

    def __getitem__(self, idx):
        try: 
            x = self.vid_decoder(self.nerv_list[idx])
            return x.detach()
        except Exception as e:
            print(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=int, default=2000, help='dataset length',)
    parser.add_argument('--epochs', '-e', type=int, default=4, help='total epochs',)
    parser.add_argument('--frames', type=int, nargs='+', default=[8, 12, 16, 32, 64], help='dataset length',) #

    args = parser.parse_args()

    root_dir = '/fs/cfar-projects/hao-nerv/SlowFast/data'
    csv_list = ['k400_240_.35M_nvloader', 'k400_360_.4M_nvloader', 'sth_240_2K_.2M_nvloader', 
        'ucf_240_.2M_nvloader', 'ucf_240_.35M_nvloader'][:1]
    # out_tags = ['k240_.35M', 'K360_.4M', 'sth_.15M']

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device, batch_size, workers = 'cpu', 8, 8
    patch_pencentile_list = [1, 0.5, 0.25, 0.1][:1]

    out_path = '/fs/cfar-projects/hao-nerv/nvloader/output/nvloader_speed'
    for csv_file in csv_list:
        data_list, epoch_list = [], []
        frames_list, patch_list, time_list, fps_list, variation_list = [], [], [], [], []
        # csv_file = os.path.join(root_dir, csv_file, 'train.csv')
        for num_frames in args.frames:
                train_dataset = VideoDataSet()
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=True, sampler=None, drop_last=False)

                data_list.append(args.data)
                epoch_list.append(args.epochs)
                frames_list.append(num_frames)
                epoch_time_list, epoch_fps_list = [], []
                for epoch in range(args.epochs):
                    start = time.time()
                    for batch in tqdm(train_dataloader, desc=f'{csv_file} frame{num_frames} Epoch {epoch}:'):
                        input_vid = batch
                    total_time = time.time() - start
                    epoch_time_list.append(total_time/60)
                    epoch_fps_list.append(args.data/total_time)
                epoch_fps_list.remove(min(epoch_fps_list))
                fps_list.append(sum(epoch_fps_list)/len(epoch_fps_list))
                fps_variation = (max(epoch_fps_list) - min(epoch_fps_list)) / 2
                variation_list.append(fps_variation)
                time_list.append(sum(epoch_time_list))       
                # import pdb; pdb.set_trace; from IPython import embed; embed()  
                result_df = pd.DataFrame.from_dict({'frame': frames_list, 'data': data_list, 'epoch': epoch_list, 
                    'time (min)': time_list, 'VPS': fps_list, 'vps_variation': variation_list})    
                print(result_df)
        time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        result_df.to_csv(f'{out_path}/{csv_file}_{time_str}.csv')

if __name__ == '__main__':
    main()
