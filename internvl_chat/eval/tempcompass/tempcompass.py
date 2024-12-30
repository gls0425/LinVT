#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import os.path as osp
from copy import deepcopy
import argparse
import torch
from pandas import read_parquet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from utils_tempcompass import load_decord

ANNO_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/gaolishuai/video_dataset/TempCompass/questions'
VIDEO_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/gaolishuai/video_dataset/TempCompass/videos'
TASK_TYPES=('multi-choice' 'captioning' 'yes_no' 'caption_matching')
output_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/gaolishuai/results/leaderboard/tempcompass/'
answer_prompt = {
    # "multi-choice": "\nBest Option:",     # The old version
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    # "caption_matching": "\nBest Option:",     #The old version
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""  # The answer "Generated Caption:" is already contained in the question
}

def evaluation(model, tokenizer, sample_config: dict, batch_size: int = 4,):
    for task_type in TASK_TYPES:
        # Loading questions
        question_path = osp.join(ANNO_DIR, task_type+'.json')
        with open(question_path, 'r') as f:
            input_datas = json.load(f)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pred_file = osp.join(output_path, task_type+'.json')
        # Loading existing predictions
        if os.path.isfile(pred_file):
            with open(osp.join(output_path, task_type+'.json'), 'r') as f:
                predictions = json.load(f)
        else:
            predictions = {}

        for vid, data in tqdm(input_datas.items()):
            if vid not in predictions:
                predictions[vid] = {}
                video_path = os.path.join(VIDEO_DIR, f'{vid}.mp4')
                for dim, questions in data.items():
                    predictions[vid][dim] = []
                    for question in questions:
                        if task_type == 'caption_matching':
                            # question example: "Which description is a more suitable match for the video?\nOption 1: The man is dribbling a basketball.\nOption 2: A man is dunking a basketball."
                            options = question['question'].split('\n')[1:]
                            options = [o.split(':')[0] for o in options]
                            inp = question['question'] + answer_prompt[task_type].replace(':', f" ({' or '.join(options)}):")
                        else:
                            inp = question['question'] + answer_prompt[task_type]
                        video_llm_pred = inference_single_video(video_path, inp, model, tokenizer, sample_config)
                        # while not any(prefix in video_llm_pred for prefix in ['A', 'B', 'C', 'D']):
                        #     video_llm_pred = inference_single_video(video_path, inp, chat, args)
                        predictions[vid][dim].append(
                            {'question': question['question'], 'answer': question['answer'], 'prediction': video_llm_pred})
                with open(pred_file, 'w') as f:
                    json.dump(predictions, f, indent=4)

def inference_single_video(video_path, prompt, model, tokenizer, sample_config):
    images = load_decord(video_path, **sample_config)
    num_patches_list = []
    pixel_values = []
    for image in images:
        if args.dynamic:
            patches = dynamic_preprocess(image, image_size=image_size, use_thumbnail=use_thumbnail,
                                         max_num=args.max_num)
        else:
            patches = [image]
        num_patches_list.append(len(patches))
        pixel_values.extend([transform(patch) for patch in patches])

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    llm_message = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        num_patches_list=num_patches_list,
        num_video_query_token=args.num_video_query_token if args.use_ffc else -1,
        question=prompt,
        generation_config=generation_config,
        verbose=True
    )
    print(llm_message)
    return llm_message

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=64)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--num_segments', type=int, default=16)
    parser.add_argument('--use_ffc', type=bool, default=False)
    parser.add_argument('--num_video_query_token', type=int, default=64)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    transform = build_transform(is_train=False, input_size=image_size)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')

    generation_config = dict(
        num_beams=args.num_beams,
        max_new_tokens=1000,
        min_new_tokens=1,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
    )
    evaluation(
        model,
        tokenizer,
        dict(
            sample_type='uniform',
            num_frames=96,
        ),
        1
    )