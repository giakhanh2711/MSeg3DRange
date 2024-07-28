#!/bin/bash

#source ~/.bashrc
source ./.bashrc.local


# 1 gpu: non-distributed train0
CUDA_VISIBLE_DEVICES=3 python ./tools/dist_test.py \
	configs/semanticnusc/MSeg3D/semnusc_range_48_e50_test.py \
	--work_dir work_dirs/semnusc_range_48_e50 \
	--checkpoint work_dirs/semnusc_range_48_e50/best.pth \
	--testset
    # --miou1
