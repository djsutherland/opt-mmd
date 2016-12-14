#!/usr/bin/env bash

if [[ $1 == 'sample' ]]; then
    is_train=False
    echo "Sampling"
else
    is_train=True
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main_mmd_fm.py \
    --max_iteration=50000 \
    --use_kernel --is_train=$is_train \
    --name=mmd_fm_lr0.5_init0.2_dlr0.02_concat_b500_iter50000 \
    --batch_size=500 --kernel_d_learning_rate=0.02  --init=0.2 \
    --learning_rate=0.5 --use_gan
