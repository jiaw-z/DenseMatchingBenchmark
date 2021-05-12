#!/bin/bash
<<<<<<< HEAD
python tools/demo.py \
    --config-path ./configs/PSMNet/kitti_2015.py \
    --checkpoint-path /data1/StereoMatching/exps/PSMNet/kitti_2015/epoch_300.pth \
    --data-root tools/demo_data_kitti/ \
    --device cuda:0 \
    --log-dir /data1/StereoMatching/exps/PSMNet/kitti_2015/output \
    --pad-to-shape 384 1248 \
=======
python demo.py \
    --config-path ../configs/AcfNet/scene_flow_adaptive.py \
    --checkpoint-path /data/exps/AcfNet/scene_flow_adaptive/epoch_20.pth \
    --data-root ./demo_data/ \
    --device cuda:0 \
    --log-dir /data/exps/AcfNet/scene_flow_adaptive/output/ \
    --pad-to-shape 544 960 \
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
    --scale-factor 1.0 \
    --disp-div-factor 1.0 \
