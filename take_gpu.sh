#!/bin/sh

#srun --account=cvcs_2023_group17 --partition=students-prod --gres=gpu:1 python train.py --pretrained-weights darknet53.conv.74


srun --account=cvcs_2023_group17 --exclude=aimagelab-srv-10  --partition=students-prod --gres=gpu:1 --pty bash



# srun --account=cvcs_2023_group17 --partition=students-dev --gres=gpu:2 python train.py --pretrained-weights darknet53.conv.74 --n_cpu 2
