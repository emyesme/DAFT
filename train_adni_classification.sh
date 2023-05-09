#!/bin/sh

DATA_DIR="/home/ecarvajal /Desktop/MyCloneDAFT/DAFT/data_dir"  # <- Your Data Directory

python train.py \
    --experiment_name "mask_flair_t1_folds_manual2_aug2" \
    --train_data "${DATA_DIR}/manualtry2_train_fold1.h5" \
    --val_data "${DATA_DIR}/manualtry2_val_fold1.h5" \
    --test_data "${DATA_DIR}/manualtry2_test_fold1.h5" \
    --input_channels 3 \
    --discriminator_net "daft_v2" \
    --optimizer "AdamW" \
    --activation "tanh" \
    --learning_rate "0.00013" \
    --decay_rate "0.001" \
    --bottleneck_factor "7.0" \
    --n_basefilters "4" \
    --num_classes "4" \
    --dataset "longitudinal" \
    --normalize_image "minmax" \
    --task "clf" \
    --tensorboard \
    --batchsize 16 \
    --epoch 200 \
    --workers 2

