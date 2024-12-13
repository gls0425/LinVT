# LinVT: Empower Your Image-level Large Language Model to Understand Videos

## News
[2024/12/09] 🔥 Our paper is coming! We release our paper on [Arxiv](https://arxiv.org/abs/2412.05185). Please refer to the paper for more details. 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linvt-empower-your-image-level-large-language/video-question-answering-on-mvbench)](https://paperswithcode.com/sota/video-question-answering-on-mvbench?p=linvt-empower-your-image-level-large-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linvt-empower-your-image-level-large-language/zero-shot-video-question-answer-on-egoschema-1)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-egoschema-1?p=linvt-empower-your-image-level-large-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linvt-empower-your-image-level-large-language/video-question-answering-on-next-qa)](https://paperswithcode.com/sota/video-question-answering-on-next-qa?p=linvt-empower-your-image-level-large-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linvt-empower-your-image-level-large-language/zeroshot-video-question-answer-on-activitynet)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-activitynet?p=linvt-empower-your-image-level-large-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linvt-empower-your-image-level-large-language/zeroshot-video-question-answer-on-msvd-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msvd-qa?p=linvt-empower-your-image-level-large-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linvt-empower-your-image-level-large-language/zeroshot-video-question-answer-on-msrvtt-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msrvtt-qa?p=linvt-empower-your-image-level-large-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/linvt-empower-your-image-level-large-language/zeroshot-video-question-answer-on-tgif-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-tgif-qa?p=linvt-empower-your-image-level-large-language)


The other ranking lists are coming soon.

<p align="center">
 <img src="docs/network.jpg" width="100%">
</p>

## 📖 Abstract
Large Language Models (LLMs) have been widely used in various tasks, motivating us to develop an LLM-based assistant for videos. Instead of training from scratch, we propose a module to transform arbitrary well-trained image-based LLMs into video-LLMs (after being trained on video data). To better adapt image-LLMs for processing videos, we introduce two design principles: linear transformation to preserve the original visual-language alignment and representative information condensation from redundant video content. Guided by these principles, we propose the Linear Video Tokenizer (LinVT), which enables existing image-LLMs to understand videos. We benchmark LinVT with six recent visual LLMs: Blip-3, Molmo, Mipha, InternVL2, Qwen2-VL and Aquila, show-casing the high compatibility of LinVT. Extensive experiments illustrate the effectiveness of LinVT in multi-modal video understanding while preserving the original image-comprehension capabilities.

## Installation
Install required packages. 

```bash
conda create -n LinVT python=3.10.13
conda activate LinVT
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -c conda-forge -y
pip install -r requirements.txt
```

## Model weights
comming soon.

## Inference
comming soon.

## Training
comming soon.
