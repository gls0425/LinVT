#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/06/28 14:26:35
@email: galishuai0425@163.com
@description: Utilities for video loading
================================================
"""
import os.path as osp
from typing import Any

import pysubs2
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import default_collate


def video_collate_fn(batch: Any) -> Any:
    """this collate function address dict video inputs, support to process variable number of frames for different inputs

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(batch[0], dict) and 'video' in batch[0]:
        video = [b.pop('video') for b in batch]
        batch = default_collate(batch)
        batch['video'] = video
    else:
        batch = default_collate(batch)
    return batch


def uniform_indices(num_frames: int, total_frames: int) -> list[int]:
    """Get uniform indices 

    Args:
        num_frames (int): number of frames
        total_frames (int): total number of frames

    Returns:
        list[int]: Output frame indices
    """
    if num_frames < total_frames:
        splits = torch.linspace(0, total_frames, num_frames+1, dtype=int)
        indices = ((splits[:-1] + splits[1:]) // 2).tolist()
    else:
        indices = list(range(total_frames))

    return indices


def fps_indices(input_fps: float, total_frames: int, output_fps: float = None, max_num_frames: int = -1) -> list[int]:
    """Get indices according to the output_fps

    Args:
        input_fps (float): input fps
        total_frames (int): total number of frames
        output_fps (float, optional): output fps. Defaults to None, means output_fps==input_fps.
        max_num_frames (int, optional): max number of frames. Defaults to -1, means no limitation.

    Returns:
        list[int]: Output frame indices
    """
    delta = 1 if output_fps is None else input_fps / output_fps
    indices = torch.arange(0, total_frames, delta).round().to(int)
    indices = [e for e in indices if e < total_frames]
    if 0 < max_num_frames < len(indices):
        indices = indices[:max_num_frames]

    return indices


def get_avg_fps_gif(src_path: str, im: Image.Image = None, default_fps: int = 20) -> float:
    """Get average fps for gif

    Args:
        src_path (str): gif path
        im (Image.Image, optional): gif Image object from src_path. Defaults to None.
        default_fps (int, optional): default fps if no duration is read. Defaults to 20.

    Returns:
        float: average fps
    """
    count, duration = 0, 0

    # if Image object is not provided, load it from src_path
    if image_flag := im is None:
        im = Image.open(src_path)

    for i in range(im.n_frames):
        try:
            im.seek(i)
            duration += im.info['duration']     # some frame do not have duration
            count += 1
        except:
            pass
    if count == 0:
        print(f'No duration is got from {src_path}, use default value as {default_fps}')
        fps = default_fps
    else:
        fps = 1000 * count / duration

    # only close object is opened by this function
    if image_flag:
        im.close()

    return fps


def load_subtitle(sub_path: str, indices: list[int], fps: float) -> str:
    """Load subtitle related to given indices

    Args:
        sub_path (str): subtitle path
        indices (list[int]): frame indices
        fps (float): video average fps

    Returns:
        str: subtitle
    """
    subs = pysubs2.load(sub_path, encoding='utf-8')
    subtitles = []
    for idx in indices:
        sub_text = []
        cur_time = pysubs2.make_time(fps=fps, frames=idx)
        for sub in subs:
            if sub.end < cur_time:
                continue
            elif sub.start < cur_time:
                sub_text.append(sub.text.replace('\\N', ' '))
                break   # in accordance with the official Video-MME Benchmark
            else:
                break
        sub_text = ' '.join(sub_text)
        if sub_text.strip():
            subtitles.append(sub_text)
    subtitles = '\n'.join(subtitles)

    return subtitles
    

def load_decord(src_path: str, sample_type: str, sub_path: str = None, **kwargs) -> list[Image.Image]:
    """Load video using decord, optionally load subtitles

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        sub_path (str): subtitle path, .srt
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, subtitle str (optional)
    """
    vr = VideoReader(src_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        indices = uniform_indices(num_frames, total_frames)
    elif sample_type == 'fps':
        input_fps = float(vr.get_avg_fps())
        output_fps = kwargs.pop('output_fps', None)
        max_num_frames = kwargs.pop('max_num_frames', -1)
        indices = fps_indices(input_fps, total_frames, output_fps, max_num_frames)
    else:
        raise ValueError(f'Do not support {sample_type} sample type')
    frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frame) for frame in frames]

    if sub_path is None:
        return frames
    elif osp.exists(sub_path):
        subtitles = load_subtitle(sub_path, indices=indices, fps=float(vr.get_avg_fps()))
        return frames, subtitles
    else:
        return frames, ''
