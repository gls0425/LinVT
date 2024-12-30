#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import os.path as osp
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils_videovista import load_decord, video_collate_fn
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess


class VideoVistaDataset(Dataset):

    understanding = [
        "Objects Existence", "Objects Count", "Action Count", "Detailed Description", 'Brief Description', 'Event Description', 'Event Sequence', 'Optical Character Recognition',
        'Action Recognition',  'Action Sequence', 'Action Location', 'Event Location', 'Objects Temporal Location', 'Objects Temporal Relation',
        'Objects Spatial Location', 'Objects Spatial Relation', 'Objects Spatial Tracking', 'Human Activity Analysis', 'Anomaly Detection'
    ]

    reasoning = [
        'Relation Reasoning-Image', 'Relation Reasoning-Video', 'Event Prediction', 'Action Prediction', 'Causal Reasoning', 'Counterfactual Reasoning', 'Commonsense Reasoning', 'Logic Reasoning'
    ]

    def __init__(self, dataset_path: str, input_size: int, dynamic_image_size: bool, use_thumbnail: bool, max_num: bool, sample_config: dict) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.sample_config = sample_config

        with open(osp.join(dataset_path, 'VideoVista.json'), 'r') as f:
            self.data_list = json.load(f)
        for item in tqdm(self.data_list):
            duration_group = item['video_name'].split('.')[1]
            if osp.exists(osp.join(self.dataset_path, 'merged', item['category'], duration_group, item['video_name'])):
                item['video'] = osp.join(self.dataset_path, 'merged', item['category'], duration_group, item['video_name'])
            else:
                for i in os.listdir(osp.join(self.dataset_path, 'merged')):
                    item['video'] = osp.join(self.dataset_path, 'merged', i, duration_group, item['video_name'])
                    if osp.exists(item['video']):
                        break
                else:
                    print(item['video_name'], ' not exist')
                    raise ValueError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index) -> dict:
        item = self.data_list[index]

        # find actual video path and load video
        images = load_decord(item['video'], **self.sample_config)
        num_patches_list = []
        pixel_values = []
        for image in images:
            if self.dynamic_image_size:
                patches = dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=self.use_thumbnail,
                                             max_num=self.max_num)
            else:
                patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)

        # wrap choices into question
        question = item['Question'] + '\nOptions:\n'
        for idx, option in enumerate(item['Answer_Choices']):
            option = f"({chr(ord('A') + idx)}) {option}\n"
            question += option
            if idx == item['Answer']:
                answer = option

        return dict(
            pixel_values=pixel_values,
            question=question,
            answer=answer,
            num_patches_list=num_patches_list,
            task_type=item['Type']
        )

    @staticmethod
    def check_answer(predict: str, answer: str) -> bool:
        for k in ('Answer:', 'The answer is', 'Answer is'):
            predict = predict.removeprefix(k).strip()
        predict_option = predict.split(' ')[0].lower().replace('.', '')
        answer_option = answer.split(' ')[0].lower()
        return len(predict_option) > 0 and (predict_option in answer_option or answer_option in predict_option)


@torch.inference_mode
def evaluate(
    model,
    tokenizer,
    dataset_path: str,
    output_path: str,
    sample_config: dict,
    batch_size: int = 4,
):
    if not osp.exists(output_path):
        os.makedirs(output_path)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    dataset = VideoVistaDataset(
        dataset_path=dataset_path,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
        sample_config=sample_config
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=video_collate_fn
    )

    # Get raw results
    results = {k: [] for k in dataset.understanding + dataset.reasoning}
    for _, (video_names, pixel_values, questions, answers, num_patches_lists, task_types) in enumerate(tqdm(dataloader)):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=1000,
            min_new_tokens=1,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
        )
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_lists[0],
            num_video_query_token = args.num_video_query_token if args.use_ffc else -1,
            question=questions[0],
            generation_config=generation_config,
            verbose=True
        )
        for answer, task_type, predict in zip(answers, task_types, pred):
            results[task_type].append(dict(
                predict=predict,
                answer=answer,
                correct=dataset.check_answer(predict, answer)
            ))
    with open(osp.join(output_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    accuracy, correct, total = {}, {}, {}
    for k, v in results.items():
        correct[k] = len(list(filter(lambda x: x['correct'], v)))
        total[k] = len(v)
        accuracy[k] = round(correct[k] / total[k] * 100, 2)

    understanding_correct = sum(correct[k] for k in VideoVistaDataset.understanding)
    understanding_total = sum(total[k] for k in VideoVistaDataset.understanding)
    accuracy['understanding'] = round(understanding_correct / understanding_total * 100, 2)

    reasoning_correct = sum(correct[k] for k in VideoVistaDataset.reasoning)
    reasoning_total = sum(total[k] for k in VideoVistaDataset.reasoning)
    accuracy['reasoning'] = round(reasoning_correct / reasoning_total * 100, 2)

    accuracy['avg'] = round((understanding_correct + reasoning_correct) / (understanding_total + reasoning_total) * 100, 2)
    print(f'Understanding: {accuracy["understanding"]}%, Reasoning: {accuracy["reasoning"]}%, Avg: {accuracy["avg"]}%')

    with open(osp.join(output_path, 'upload_leaderboard.json'), 'w') as f:
        json.dump(accuracy, f, indent=4, ensure_ascii=False)

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

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')

    dataset_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vision-data/yuanyitian/MVLU/MLVU'
    output_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/gaolishuai/results/leaderboard/mlvu/'
    evaluate(
        model,
        tokenizer,
        dataset_path,
        output_path,
        dict(
            sample_type='uniform',
            num_frames=96
        ),
        1
    )