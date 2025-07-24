import subprocess
import pandas as pd
from tqdm import tqdm
# vid='/data/home/haochen1/data/kinetics/400/val/8TwvqsNgQwY_000060_000070.mp4' 
csv_path = '/data/home/haochen1/nerv_projects/OneNeRV/data/dataset_meta/meta/k400_val.csv'
path_list = pd.read_csv(csv_path)['path'].tolist()

for vid in tqdm(path_list[:500]):
    cmd = f"ffmpeg -i  {vid}  -vf vfrdet -an -f null -"
    out = subprocess.run(cmd.split(), capture_output=True, text=True)        
    vfr = None
    for out_str in out.stderr.split('\n')[-2].split():
        if 'VFR' in out_str:
            vfr = float(out_str.replace('VFR:', ''))
            if vfr:
                print(vid, vfr)
    if vfr == None:
        print(vid, 'no vfr')