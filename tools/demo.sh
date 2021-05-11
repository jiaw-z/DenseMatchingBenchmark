#!/bin/bash
python tools/demo.py \
    --config-path ./configs/PSMNet/kitti_2015.py \
    --checkpoint-path /data1/StereoMatching/exps/PSMNet/kitti_2015/epoch_300.pth \
    --data-root tools/demo_data_kitti/ \
    --device cuda:0 \
    --log-dir /data1/StereoMatching/exps/PSMNet/kitti_2015/output \
    --pad-to-shape 384 1248 \
    --scale-factor 1.0 \
    --disp-div-factor 1.0 \
