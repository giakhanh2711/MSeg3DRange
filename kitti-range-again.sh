#!/bin/bash

#source ~/.bashrc
source ./.bashrc.local


# 1 gpu: non-distributed train0
CUDA_VISIBLE_DEVICES=5 python ./tools/train.py configs/semantickitti/MSeg3D/semkitti_range_48_e50_again.py \
	--resume_from work_dirs/semkitti_range_48_e50_again/latest.pth \
	--validate
