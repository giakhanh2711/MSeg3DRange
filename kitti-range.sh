#!/bin/bash

#source ~/.bashrc
source ./.bashrc.local


# 1 gpu: non-distributed train0
CUDA_VISIBLE_DEVICES=4 python ./tools/train.py configs/semantickitti/MSeg3D/semkitti_range_48_e36.py \
	--validate
    # --resume_from work_dirs/semkitti_range_48_e36/latest.pth \
