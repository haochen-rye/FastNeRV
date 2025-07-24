import os
import numpy as np

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

# video loading
batch_size=2
sequence_length=8
initial_prefetch_size=16
video_directory = "/data/home/haochen1/data/kinetics/400/val"
video_files=[video_directory + '/' + f for f in os.listdir(video_directory)]
n_iter=6


@pipeline_def
def video_pipe(filenames):
    videos = fn.readers.video(device="gpu", filenames=filenames, sequence_length=sequence_length,
                              shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size)
    return videos


pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0, filenames=video_files, seed=123456)
pipe.build()
for i in range(n_iter):
    pipe_out = pipe.run()
    sequences_out = pipe_out[0].as_cpu().as_array()
    print(sequences_out.shape)

# img reading
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "data/images"
max_batch_size = 8

@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels

pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
pipe.build()
pipe_out = pipe.run()
