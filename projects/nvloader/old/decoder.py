#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import numpy as np
import random
import torch
import torchvision
import torchvision.io as io
import av
from decord import VideoReader
from random import randint

def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def temporal_sampling(video_duration, sampling_rate, num_frames):
    if sampling_rate == -1:
        random_starting_idx, end_idx = 0, video_duration-1
    else:
        clip_duration = sampling_rate * (num_frames - 1)
        random_starting_idx = randint(0, max(video_duration - clip_duration, 0))
        end_idx = min(random_starting_idx + clip_duration, video_duration-1)
    frame_idx_list = torch.linspace(random_starting_idx, end_idx, num_frames).round().int().tolist()
    return frame_idx_list


def decord_decode(
    path_to_vid,
    sampling_rate,
    num_frames,    
    ):
    decord_vr = VideoReader(path_to_vid)
    video_duration = len(decord_vr)
    frame_idx_list = temporal_sampling(video_duration, sampling_rate, num_frames)
    frames = decord_vr.get_batch(frame_idx_list).asnumpy()

    return torch.from_numpy(frames)


def torchvision_decode(
    # video_handle,
    path_to_vid,
    sampling_rate,
    num_frames,
    clip_idx,
    video_meta={},
    num_clips_uniform=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
    use_offset=False,
    min_delta=-math.inf,
    max_delta=math.inf,
):
    """
    If video_meta is not empty, perform temporal selective decoding to sample a
    clip from the video with TorchVision decoder. If video_meta is empty, decode
    the entire video and update the video_meta.
    Args:
        video_handle (bytes): raw bytes of the video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips_uniform clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the resolution of the spatial shorter
            edge size during decoding.
        min_delta (int): minimum distance between clips when sampling multiple.
        max_delta (int): max distance between clips when sampling multiple.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): if True, the entire video was decoded.
    """
    timestamp_list = torchvision.io.read_video_timestamps(path_to_vid)[0]
    frame_idx_list = temporal_sampling(len(timestamp_list), sampling_rate, num_frames)
    start_end_pts = [timestamp_list[frame_idx_list[x]] for x in [0, -1]]
    # import pdb; pdb.set_trace; from IPython import embed; embed()  
    v_frames, _, _ = io.read_video(path_to_vid, start_pts=start_end_pts[0], 
                end_pts=start_end_pts[-1], pts_unit='pts')  
    # print(f'torchvision frame_shape: {v_frames.shape}')
    return v_frames


def pyav_decode(
    # container,
    path_to_vid,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips_uniform=10,
    target_fps=30,
    use_offset=False,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    container = av.open(path_to_vid)
    container.streams.video[0].thread_type = "AUTO"
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None or sampling_rate==-1:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        clip_size = np.maximum(
            1.0, np.ceil(sampling_rate * (num_frames - 1) / target_fps * fps)
        )
        start_idx, end_idx, fraction = get_start_end_idx(
            frames_length,
            clip_size,
            clip_idx,
            num_clips_uniform,
            use_offset=use_offset,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    # import pdb; pdb.set_trace; from IPython import embed; embed()    
    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.from_numpy(np.stack(frames))

    return frames


def decode(
    # container,
    path_to_vid,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips_uniform=1,
    video_meta=None,
    target_fps=30,
    backend="pyav",
    max_spatial_scale=0,
    use_offset=False,
    time_diff_prob=0.0,
    gaussian_prob=0.0,
    min_delta=-math.inf,
    max_delta=math.inf,
    temporally_rnd_clips=True,
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (list of ints): frame sampling rate (interval between two sampled
            frames).
        num_frames (list of ints): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips_uniform clips, and select the
            clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    try:
        if backend == "pyav":
            assert (
                min_delta == -math.inf and max_delta == math.inf
            ), "delta sampling not supported in pyav"
            frames_decoded = pyav_decode(
                path_to_vid,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips_uniform,
                target_fps,
                use_offset=use_offset,
            )
        elif backend == "torchvision":
            frames_decoded = torchvision_decode(
                path_to_vid,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips_uniform,
                target_fps,
                ("visual",),
                max_spatial_scale,
                use_offset=use_offset,
                min_delta=min_delta,
                max_delta=max_delta,
            )
        elif backend == "decord":
            frames_decoded = decord_decode(
                path_to_vid,
                sampling_rate,
                num_frames,
            )            
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None
    t_frames = frames_decoded.size(0)
    # print(f'frame_length: {t_frames}')
    sample_idx = torch.linspace(0, t_frames-1, num_frames).round().to(dtype=torch.int64)
    frames = frames_decoded.index_select(0, sample_idx)

    return frames.permute(0,-1,1,2)


def get_start_end_idx(
    video_size, clip_size, clip_idx, num_clips_uniform, use_offset=False
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips_uniform == 1:
                # Take the center clip if num_clips_uniform is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(
                    delta / (num_clips_uniform - 1)
                )
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips_uniform
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx, start_idx / delta if delta != 0 else 0.0


def get_multiple_start_end_idx(
    video_size,
    clip_sizes,
    clip_idx,
    num_clips_uniform,
    min_delta=0,
    max_delta=math.inf,
    use_offset=False,
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_sizes (list): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """

    def sample_clips(
        video_size,
        clip_sizes,
        clip_idx,
        num_clips_uniform,
        min_delta=0,
        max_delta=math.inf,
        num_retries=100,
        use_offset=False,
    ):
        se_inds = np.empty((0, 2))
        dt = np.empty((0))
        for clip_size in clip_sizes:
            for i_try in range(num_retries):
                # clip_size = int(clip_size)
                max_start = max(video_size - clip_size, 0)
                if clip_idx == -1:
                    # Random temporal sampling.
                    start_idx = random.uniform(0, max_start)
                else:  # Uniformly sample the clip with the given index.
                    if use_offset:
                        if num_clips_uniform == 1:
                            # Take the center clip if num_clips is 1.
                            start_idx = math.floor(max_start / 2)
                        else:
                            start_idx = clip_idx * math.floor(
                                max_start / (num_clips_uniform - 1)
                            )
                    else:
                        start_idx = max_start * clip_idx / num_clips_uniform

                end_idx = start_idx + clip_size - 1

                se_inds_new = np.append(se_inds, [[start_idx, end_idx]], axis=0)
                if se_inds.shape[0] < 1:
                    se_inds = se_inds_new
                    break

                se_inds_new = np.sort(se_inds_new, 0)
                t_start, t_end = se_inds_new[:, 0], se_inds_new[:, 1]
                dt = t_start[1:] - t_end[:-1]
                if (
                    any(dt < min_delta) or any(dt > max_delta)
                ) and i_try < num_retries - 1:
                    continue  # there is overlap
                else:
                    se_inds = se_inds_new
                    break
        return se_inds, dt

    num_retries, goodness = 100, -math.inf
    for _ in range(num_retries):
        se_inds, dt = sample_clips(
            video_size,
            clip_sizes,
            clip_idx,
            num_clips_uniform,
            min_delta,
            max_delta,
            100,
            use_offset,
        )
        success = not (any(dt < min_delta) or any(dt > max_delta))
        if success or clip_idx != -1:
            se_final, dt_final = se_inds, dt
            break
        else:
            cur_goodness = np.r_[dt[dt < min_delta], -dt[dt > max_delta]].sum()
            if goodness < cur_goodness:
                se_final, dt_final = se_inds, dt
                goodness = cur_goodness

    delta_clips = np.concatenate((np.array([0]), dt_final))
    start_end_delta_time = np.c_[se_final, delta_clips]

    return start_end_delta_time
    