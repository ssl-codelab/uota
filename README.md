# UOTA: Improving Self-supervised Learning with Automated Unsupervised Outlier Arbitration
This repository is the official [PyTorch](http://pytorch.org) implementation of **UOTA** (**U**nsupervised **O**u**T**lier **A**rbitration).

## 1 Citation
Yu Wang, Jingyang Lin, Jingjing Zou, Yingwei Pan, Ting Yao, Tao Mei. *UOTA: Improving Self-supervised Learning with Automated Unsupervised Outlier Arbitration*. In *NeurIPS*, 2021.
```
@InProceedings{wang2021NeurIPS,
  title={Improving Self-supervised Learning with Automated Unsupervised Outlier Arbitration},
  author={Wang, Yu and Lin, Jingyang and Zou, Jingjing and Pan, Yingwei and Yao, Ting and Mei, Tao},
  booktitle={NeurIPS},
  year={2021},
}
```

## 2 Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.6.0
- torchvision install = 0.7.0
- CUDA 10.1
- [Apex](https://github.com/NVIDIA/apex) with CUDA extension
- Other dependencies: opencv-python, scipy, pandas, numpy

## 3 Pretraining
We release a demo for the SwAV+UOTA self-supervised learning approach. The model is based on ResNet50 architecture, pretrained for 200 epochs.

### 3.1 SwAV+UOTA pretrain

To train SwAV+UOTA on a single node with 4 gpus for 200 epochs, run:
```
DATASET_PATH="path/to/ImageNet1K/train"
EXPERIMENT_PATH="path/to/experiment"

python -m torch.distributed.launch --nproc_per_node=4 main_uota.py \
--data_path ${DATASET_PATH} \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--use_pil_blur true \
--epochs 200 \
--warmup_epochs 0 \
--batch_size 64 \
--base_lr 0.6 \
--final_lr 0.0006 \
--uota_tau 350. \
--epoch_uota_starts 100 \
--wd 0.000001 \
--use_fp16 true \
--dist_url "tcp://localhost:40000" \
--arch uota_r50 \
--sync_bn pytorch \
--dump_path ${EXPERIMENT_PATH}
```

## 4 Linear Evaluation
To train a linear classifier on frozen features out of deep network pretrained via various self-supervised pretraining methods, run:
```
DATASET_PATH="path/to/ImageNet1K"
EXPERIMENT_PATH="path/to/experiment"
LINCLS_PATH="path/to/lincls"

python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py \
--data_path ${DATASET_PATH} \
--arch resnet50 \
--lr 1.2 \
--dump_path ${LINCLS_PATH} \
--pretrained ${EXPERIMENT_PATH}/swav_uota_r50_e200_pretrained.pth \
--batch_size 64 \
--num_classes 100 \
```

## 5 Results
To compare with SwAV fairly, we provide a SwAV+UOTA model with ResNet-50 architecture pretrained on ImageNet1K for 200 epochs, and release the pretrained model and the linear classier.

| method | epochs | batch-size | multi-crop | ImageNet1K top-1 acc. | pretrained model | linear classifier |
|-------------------|-------------------|---------------------|--------------------|--------------------|--------------------|--------------------|
| SwAV | 200 | 256 | 2x224 + 6x96 | 72.7 | / | / |
| SwAV + UOTA | 200 | 256 | 2x224 + 6x96 | 73.5 | [pretrained](https://github.com/ssl-codelab/uota/releases/download/v1.0.0/swav_uota_r50_e200_pretrained.pth.tar) | [linear](https://github.com/ssl-codelab/uota/releases/download/v1.0.0/swav_uota_r50_e200_lincls.pth.tar) |
