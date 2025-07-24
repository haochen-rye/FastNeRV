from decoder import decode
from torchvision.io import write_video
import torch

path_to_vid='/data/home/haochen1/data/kinetics/400/val/8TwvqsNgQwY_000060_000070.mp4' 
out_list = []
backend_list = ['torchvision', 'pyav', 'decord']
for backend in backend_list:
    frames = decode(path_to_vid, -1, 16, backend=backend) 
    # (Tensor[T, H, W, C]): tensor containing the individual frames, as a uint8 tensor in [T, H, W, C] format
    write_video(f'{backend}.mp4',frames.permute(0,2,3,1), fps=2)
    # import pdb; pdb.set_trace; from IPython import embed; embed()  
    out_list.append(frames)
    if frames is not None:
        print(f'Backend: {backend} frames: {frames.shape} {frames.dtype} min_max: {frames.min()}_{frames.max()}')

for idx in [1,2]:
    print(f'{backend_list[0]} & {backend_list[idx]}:')
    print(torch.equal(out_list[0], out_list[idx]))

# import nvidia.dali.fn as fn



# ffmpeg -benchmark -i /data/home/haochen1/data/kinetics/400/val/8TwvqsNgQwY_000060_000070.mp4 -f null -
