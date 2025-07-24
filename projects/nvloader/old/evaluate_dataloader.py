import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
from torch.utils.data import Dataset

class VideoDataSet(Dataset):
    def __init__(self, csv_path, num_frames, patch_pencentile, data_len=2000):
        path_df=pd.read_csv(csv_path)
        self.num_frames = num_frames
        self.vid_path_list = path_df['vid'].tolist()[:data_len]
        self.patch_pencentile = patch_pencentile

    def __len__(self):
        return len(self.vid_path_list)

    def __getitem__(self, idx):
        try: 
            device = 'cpu'
            vid_path = self.vid_path_list[idx]
            img_decoder_path, patch_decoder_path, ckt_path = 'img_decoder.pth', 'patch_decoder.pth', 'quant_vid.pth'
            quant_ckt = torch.load(os.path.join(vid_path, ckt_path), map_location='cpu')
            vid_embed = torch.dequantize(quant_ckt['embed']).to(device)
            dequant_ckt = {k:torch.dequantize(v).to(device) for k,v in quant_ckt['model'].items()}
            if self.patch_pencentile==1:
                img_decoder = torch.jit.load(os.path.join(vid_path, img_decoder_path), map_location='cpu').to(device)
                img_decoder.load_state_dict(dequant_ckt)
                fold_func = torch.nn.Fold(quant_ckt['shape'],quant_ckt['strd'],stride=quant_ckt['strd']).to(device)     
                y = img_decoder(vid_embed[:self.num_frames])
                img = fold_func(y.permute(0,2,1))
                return img.detach()
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
        csv_file = os.path.join(root_dir, csv_file, 'train.csv')
        for num_frames in args.frames:
            for patch_pencentile in patch_pencentile_list:
                train_dataset = VideoDataSet(csv_file, num_frames, patch_pencentile, args.data)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True, sampler=None, drop_last=False)

                data_list.append(args.data)
                epoch_list.append(args.epochs)
                frames_list.append(num_frames)
                patch_list.append(patch_pencentile)
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
                result_df = pd.DataFrame.from_dict({'frame': frames_list, 'data': data_list, 'epoch': epoch_list, 'patch': patch_list,
                    'time (min)': time_list, 'VPS': fps_list, 'vps_variation': variation_list})    
                print(result_df)
        time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        result_df.to_csv(f'{out_path}/{csv_file}_{time_str}.csv')

if __name__ == '__main__':
    main()
