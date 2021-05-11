#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=8 \
        tools/train.py ./configs/PSMNet/scene_flow.py --launcher pytorch --validate --gpus 8 --resume_from /home/user/data1/StereoMatching/exps/PSMNet/scene_flow/epoch_4.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=7 \
        tools/train.py ./configs/PSMNet/scene_flow.py --launcher pytorch --validate --gpus 7


python tools/train.py ./configs/PSMNet/scene_flow.py --gpus 1 --launcher none --validate

python tools/train.py ./configs/PSMNet/kitti_2015.py --gpus 1 --launcher none --validate


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=4 \
        tools/train.py ./configs/PSMNet/kitti_2015.py --launcher pytorch --validate --gpus 4

