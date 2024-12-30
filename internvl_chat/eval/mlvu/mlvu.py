#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from .utils_mlvu import load_decord, video_collate_fn
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess

class MLVUDataset(Dataset):

    _raw_data = dict(
        M={
            'count': ('4_count.json', 'video/count', 'video'),
            'ego': ('3_ego.json', 'video/ego', 'video'),
            'needle': ('2_needle.json', 'video/needle', 'video'),
            'order': ('5_order.json', 'video/order', 'video'),
            'plotQA': ('1_plotQA.json', 'video/plotQA', 'video'),
            'anomaly_reco': ('6_anomaly_reco.json', 'video/anomaly_reco', 'video'),
            'topic_reasoning': ('7_topic_reasoning.json', 'video/topic_reasoning', 'video')
        },
        G={
            'subPlot': ('8_sub_scene.json', 'video/subPlot', 'video'),
            'summary': ('9_summary.json', 'video/summary', 'video')
        }
    )

    def __init__(self, dataset_path: str, task_name: str,
                 input_size: int, dynamic_image_size: bool, use_thumbnail: bool, max_num: bool,
                 sample_config: dict):
        """
        Args:
            dataset_path (str): _description_
            task_name (str): ['M', 'G'], 'multiple choice', 'generation'
            sample_config (dict): _description_
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.sample_config = sample_config

        self.data_list = []
        for k, (json_path, video_path, data_type) in self._raw_data[self.task_name].items():
            prefix = osp.join(dataset_path, 'video', osp.splitext(json_path)[0])
            with open(osp.join(dataset_path, 'json', json_path), 'r') as f:
                for data in json.load(f):
                    self.data_list.append({
                        'task_type': k,
                        'prefix': prefix,
                        'data_type': data_type,
                        'data': data
                    })

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])

        correct = 0
        total = 0
        res = f'There are {len(self.data_list)} videos as follow:\n'
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f'{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n'
            correct = correct + 1 / option_list[k]
        res += f'Total random accuracy: {correct/total*100:.2f}%'
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def qa_template(self, data):
        if self.task_name == 'M':
            question = f"Question: {data['question']}\n"
            question += 'Options:\n'
            answer = data['answer']
            answer_idx = -1
            for idx, c in enumerate(data['candidates']):
                question += f"({chr(ord('A') + idx)}) {c}\n"
                if c == answer:
                    answer_idx = idx
            question = question.rstrip()
            answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        else:
            question = data['question']
            answer = data['answer']
        return question, answer

    def __getitem__(self, idx):
        video_path = osp.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        images = load_decord(video_path, **self.sample_config)
        num_patches_list = []
        pixel_values = []
        for image in images:
            if self.dynamic_image_size:
                patches = dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=self.use_thumbnail, max_num=self.max_num)
            else:
                patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)
        question, answer = self.qa_template(self.data_list[idx]['data'])

        return {
            'video_name': osp.split(self.data_list[idx]['data']['video'])[1],
            'pixel_values': pixel_values,
            'question': question,
            'answer': answer,
            'num_patches_list': num_patches_list,
            'task_type': self.data_list[idx]['task_type']
        }

    @staticmethod
    def check_answer(predict: str, answer: str) -> bool:
        for k in ('Answer:', 'The answer is'):
            predict = predict.removeprefix(k).strip()
        predict_option = predict.split(' ')[0].lower().replace('.', '')
        answer_option = answer.split(' ')[0].lower()
        return len(predict_option) > 0 and (predict_option in answer_option or answer_option in predict_option)


@torch.inference_mode
def evaluate(
    model,
    tokenizer,
    dataset_path: str,
    output_dir: str,
    sample_config: dict,
    batch_size: int = 4
):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    # Multiple choice
    dataset = MLVUDataset(
        dataset_path=dataset_path,
        task_name='M',
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
    results = {k: [] for k in dataset._raw_data['M']}
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

    with open(osp.join(output_dir, 'choice_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    # Get accuracy
    accuracy, correct_count, total_count = {}, 0, 0
    for k, v in results.items():
        correct = len(list(filter(lambda x: x['correct'], v)))
        total = len(v)
        accuracy[k] = round(correct / total * 100, 2)
        correct_count += correct
        total_count += total
    accuracy['Avg'] = round(correct_count / total_count * 100 + 1e-5, 2)    # correct rounding 55.125 -> 55.13
    print(f'Total accuracy: {accuracy["Avg"]}%')
    with open(osp.join(output_dir, 'choice_leaderboard.json'), 'w') as f:
        json.dump(accuracy, f, indent=4, ensure_ascii=False)

    # Generation
    dataset = MLVUDataset(
        dataset_path=dataset_path,
        task_name='G',
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
    results = {k: [] for k in dataset._raw_data['G']}

    for _, (video_naems, pixel_values, questions, answers, num_patches_lists, task_types) in enumerate(tqdm(dataloader)):
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
            num_video_query_token=args.num_video_query_token if args.use_ffc else -1,
            question=questions[0],
            generation_config=generation_config,
            verbose=True
        )
        for video_name, question, answer, task_type, predict in zip(video_names, questions,answers, task_types, pred):
            results[task_type].append(dict(
                video_name=video_name,
                Q=question,
                A=answer,
                pred=predict
            ))
    for task_type, results in results.items():
        with open(osp.join(output_dir, f'generation_{task_type}_results.json'), 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

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