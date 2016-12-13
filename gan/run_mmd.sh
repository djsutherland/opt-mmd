#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_mmd.py --use_kernel --is_train=True \
    --name=test_mmd_lr2_init0.1_b500_iter50000_one_kernel \
    --max_iteration=50000 --init=0.1 --learning_rate=2 --batch_size=500
