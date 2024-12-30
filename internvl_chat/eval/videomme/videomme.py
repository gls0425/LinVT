#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import os.path as osp
from copy import deepcopy

import torch
from pandas import read_parquet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import load_decord, video_collate_fn


class VideoMMEDataset(Dataset):
    """Video-MME dataset. By default, videos are saved in `video/` and subtitles are saved in `subtitle/`
    """

    def __init__(self, dataset_path: str, sample_config: dict, use_subtitle: bool = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.sample_config = sample_config
        self.use_subtitle = use_subtitle

        data_dict = {}
        index_keys = ['video_id', 'duration', 'domain', 'sub_category', 'videoID']
        value_keys = ['question_id', 'task_type', 'question', 'options', 'answer']
        df = read_parquet(osp.join(dataset_path, 'test-00000-of-00001.parquet'))
        df['options'] = df['options'].apply(list)
        for _, data in df.iterrows():
            key = tuple(data[k] for k in index_keys)
            value = data[value_keys].to_dict()
            if key in data_dict:
                data_dict[key].append(value)
            else:
                data_dict[key] = [value]
        self.data_list = [dict(zip(index_keys + ['questions'], list(k) + [v])) for k, v in data_dict.items()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> dict:
        if self.use_subtitle:
            frames, subtitles = load_decord(
                src_path=osp.join(self.dataset_path, 'data', self.data_list[idx]['videoID'] + '.mp4'),
                sub_path=osp.join(self.dataset_path, 'subtitle', self.data_list[idx]['videoID'] + '.srt'),
                **self.sample_config
            )
            text = ['\n'.join([
                "This video's subtitles are listed below:",
                subtitles,
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.data_list[idx]['questions']]
        else:
            frames = load_decord(
                src_path=osp.join(self.dataset_path, 'data', self.data_list[idx]['videoID'] + '.mp4'),
                **self.sample_config
            )
            text = ['\n'.join([
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.data_list[idx]['questions']]
            subtitles = ''

        return dict(
            video=frames,
            text=text
        )


@torch.inference_mode
def evaluate(
    model,
    tokenizer,
    image_processor,
    dataset_path: str,
    output_path: str,
    sample_config: dict,
    batch_size: int = 4,
    question_prompt: str = "Answer with the option's letter from the given choices directly."
):
    if not osp.exists(output_path):
        os.makedirs(output_path)

    dataset = VideoMMEDataset(
        dataset_path=dataset_path,
        sample_config=sample_config
    )
    for use_subtitle in (False, True):
        dataset.use_subtitle = use_subtitle
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=video_collate_fn
        )
        results = []
        for data in tqdm(dataloader):
            response = []
            images = data['video']
            for i in range(len(data['text'])):
                messages = [[{
                    'role': 'user',
                    'content': f'<video>\n{question}\n{question_prompt}'
                }] for question in data['text'][i]]
                response.append(model.chat(messages, images, tokenizer, image_processor, max_new_tokens=100, do_sample=False,
                                           eos_token_id=2, pad_token_id=2))
            response = [[response[i][j] for i in range(len(response))] for j in range(len(response[0]))]
            results.extend(response)

        outputs = []
        for data, responses in zip(dataset.data_list, results):
            data = deepcopy(data)
            data.pop('videoID')
            for question, response in zip(data['questions'], responses):
                question['response'] = response
            outputs.append(data)

        suffix = 'w_sub' if use_subtitle else 'wo_sub'
        with open(osp.join(output_path, f'output_{suffix}.json'), 'w') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
