#!/bin/bash

#SBATCH --job-name=R2D2
#SBATCH --output=%x-%j.out
#SBATCH --partition=cpu_partition
#SBATCH --nodes=1
#SBATCH --exclusive
##SBATCH --partition=gpu_partition
##SBATCH --gres=gpu:1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00

hostname; pwd; date

python m2m.py test \
    --test-in-patterns "test/lr_scalar-*.npy" \
    --test-tgt-patterns "test/hr_scalar-*.npy" \
    --model srsgan.G \
    --batch-size 2 \
    --load-state "checkpoint.pt" \
    --misc-kwargs '{
        "chan_base": 64,
        "chan_min": 32,
        "chan_max": 256,
        "cat_noise": true
    }' \
    --scale-factor 2 \

date

# Wait for a moment to ensure files are written
sleep 1

