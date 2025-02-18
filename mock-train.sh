#!/bin/bash

export CUDA_VISIBLE_DEVICES=""  # Force CPU-only execution

python m2m.py train \
    --train-in-patterns "train/lr_scalar-*.npy" \
    --train-tgt-patterns "train/hr_scalar-*.npy" \
    --val-in-patterns "val/lr_scalar-*.npy" \
    --val-tgt-patterns "val/hr_scalar-*.npy" \
    --model srsgan.G \
    --adv-model srsgan.D \
    --adv-criterion WDistLoss \
    --adv-r1-reg-interval 16 \
    --instance-noise 0.1 \
    --lr 1e-4 \
    --batch-size 2 \
    --epochs 10 \
    --seed 42 \
    --loader-workers 0 \
    --log-interval 1 \
    --scale-factor 2 \
    --misc-kwargs '{
        "chan_base": 64,
        "chan_min": 32,
        "chan_max": 256,
        "cat_noise": true
    }' \
    --optimizer-args '{"betas": [0.5, 0.9]}'
