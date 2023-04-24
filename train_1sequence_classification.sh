#!/bin/sh

DATA_DIR="/home/ecarvajal /Desktop/DAFT"  # <- Your Data Directory

python train.py \
    --train_data "${DATA_DIR}/4labels_train.h5" \
    --val_data "${DATA_DIR}/4labels_val.h5" \
    --test_data "${DATA_DIR}/4labels_test.h5" \
    --input_channels 1 \
    --discriminator_net "daftselu" \
    --optimizer "AdamW" \
    --activation "linear" \
    --learning_rate "0.01" \
    --decay_rate "0.1" \
    --film_location "3" \
    --bottleneck_factor "7.0" \
    --n_basefilters "4" \
    --num_classes "4" \
    --dataset "longitudinal" \
    --normalize_image "standardize" \
    --normalize_tabular \
    --task "clf" \
    --batchsize 32 \
    --epoch 100 \
    --workers 4

# --experiment_name "reproduction_03-01_23-02_good" \