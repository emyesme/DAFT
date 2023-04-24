#!/bin/sh

DATA_DIR="/home/ecarvajal /Desktop/DAFT"  # <- Your Data Directory

python train.py \
    --experiment_name "2group_mask_flair_t1" \
    --train_data "${DATA_DIR}/2grouptp_train.h5" \
    --val_data "${DATA_DIR}/2grouptp_val.h5" \
    --test_data "${DATA_DIR}/2grouptp_val.h5" \
    --input_channels 3 \
    --discriminator_net "daft" \
    --optimizer "AdamW" \
    --activation "tanh" \
    --learning_rate "0.00013" \
    --decay_rate "0.001" \
    --bottleneck_factor "7.0" \
    --n_basefilters "4" \
    --num_classes "2" \
    --dataset "longitudinal" \
    --normalize_tabular \
    --normalize_image "minmax" \
    --task "clf" \
    --tensorboard \
    --batchsize 16 \
    --epoch 100 \
    --workers 2

