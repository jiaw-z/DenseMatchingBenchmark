#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python tools/demo.py \
    --config-path ./configs/PSMNet/kitti_2015.py \
    --checkpoint-path /data1/StereoMatching/exps/PSMNet/scene_flow/epoch_10.pth \
    --data-root tools/demo_data_kitti/ \
    --device cuda:0 \
    --log-dir /data1/StereoMatching/exps/PSMNet/kitti_2015/output_sf_10 \
    --pad-to-shape 384 1248 \
    --scale-factor 1.0 \
    --disp-div-factor 1.0


CUDA_VISIBLE_DEVICES=7 python tools/demo.py \
    --config-path ./configs/PSMNet/scene_flow_enc.py \
    --checkpoint-path /data1/StereoMatching/exps/PSMNet/scene_flow/epoch_16_enc.pth \
    --data-root tools/demo_data/ \
    --device cuda:0 \
    --log-dir /data1/StereoMatching/exps/PSMNet/scene_flow/output_sf_enc_16 \
    --pad-to-shape 544 960 \
    --scale-factor 1.0 \
    --disp-div-factor 1.0



--checkpoint-path /data1/StereoMatching/exps/PSMNet/scene_flow/epoch_10.pth \
    --checkpoint-path /data1/StereoMatching/exps/PSMNet/kitti_2015/PSMNet-KITTI2015.pth \
