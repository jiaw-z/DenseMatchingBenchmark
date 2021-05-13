#!/usr/bin/env bash


NGPUS=$1
CFG_PATH=$2
PORT=$3
SHOW=$4

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$NGPUS \
        test.py $CFG_PATH --launcher pytorch --validate --gpus $NGPUS --show $SHOW
<<<<<<< HEAD

# sceneflow-test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=7 \
        tools/test.py ./configs/PSMNet/scene_flow.py --launcher pytorch --validate --gpus 8 --show True \
        --checkpoint /data1/StereoMatching/exps/PSMNet/scene_flow/epoch_10.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=7 \
        tools/test.py ./configs/PSMNet/scene_flow_enc.py --launcher pytorch --validate --gpus 7 --show True \
        --checkpoint /data1/StereoMatching/exps/PSMNet/scene_flow/encoder-sep/epoch_16.pth \
        --out_dir /data1/StereoMatching/exps/PSMNet/scene_flow/encoder-sep/epoch_16


# kitti
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=7 \
        tools/test.py ./configs/PSMNet/kitti_2015.py --launcher pytorch --validate --gpus 7 --show True \
        --checkpoint /data1/StereoMatching/exps/PSMNet/kitti_2015/epoch_1000_full.pth


# sceneflow-trained   kitti-test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=7 \
        tools/test.py ./configs/PSMNet/kitti_2015.py --launcher pytorch --validate --gpus 7 --show True \
        --checkpoint /data1/StereoMatching/exps/PSMNet/scene_flow/epoch_10_4gpus.pth \
        --out_dir /data1/StereoMatching/exps/PSMNet/kitti_2015/epoch_10_sceneflow

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=8 \
        tools/test.py ./configs/PSMNet/kitti_2015.py --launcher pytorch --validate --gpus 8 --show True \
        --checkpoint /data1/StereoMatching/exps/PSMNet/scene_flow/epoch_10.pth \
        --out_dir /data1/StereoMatching/exps/PSMNet/kitti_2015/epoch_10_8gpus_sceneflow

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=7 \
        tools/test.py ./configs/PSMNet/kitti_2015_enc.py --launcher pytorch --validate --gpus 7 --show True \
        --checkpoint /data1/StereoMatching/exps/PSMNet/scene_flow/encoder-sep/epoch_10.pth \
        --out_dir /data1/StereoMatching/exps/PSMNet/kitti_2015/encoder-sep/scene_flow_10


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=7 \
        tools/test.py ./configs/AcfNet/kitti_2015_uniform.py --launcher pytorch --validate --gpus 7 --show True \
        --checkpoint /data1/StereoMatching/exps/pretrained/AcfNet-SceneFlow-Uniform.pth \
        --out_dir /data1/StereoMatching/exps/AcfNet/scene_flow_uniform/valid_sceneflow_pretrained
=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
