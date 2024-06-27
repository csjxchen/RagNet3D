#!/usr/bin/env bash

# set -x
# NGPUS=$1
# PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=4 test.py --launcher pytorch \
--cfg_file cfgs/kitti_models/pv_rcnn_adaptive4_formal/pv_rcnn_adaptive4_avg_test_h10.yaml \
--ckpt /opt/data/private/research_exps/detection3d/OpenPCDet/output/kitti_models/pv_rcnn_adaptive4_formal/pv_rcnn_adaptive4_avg_test_h10/default/ckpt/checkpoint_epoch_80.pth \
--save_to_file \
--infer_time \
--workers 4
