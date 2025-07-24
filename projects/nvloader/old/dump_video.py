from torchvision.io import write_video

svtav1-params fast-decode=3

for codec,crf,tune0, tune1 in [('libx264', 50, 'tune', 'fastdecode'), ('av1', 60, 'svtav1-params', 'fast-decode=3')]:
    for crf_gap in [0, 10]:
        cur_crf = crf - crf_gap
        write_video(f'try_{codec}_crf{cur_crf}.mp4',x,video_codec=codec,options={'crf': f'{cur_crf}', f'{tune0}': f'{tune1}'},fps=2)