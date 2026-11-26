#!/bin/bash
# Task 1: bigger move-LM pretrains (hold-out 800 players, 8 epochs), launched 2026-11-16 07:55 JST on gsm-gpu2
# then fine-tune: train_idft.py <tag> --init lm4|lm5 --epochs 4 --lr 1e-4 --s 20 --m 0.1 --seed 0   (hold-out check vs b1), then --allp 1 --epochs 10
source "$(dirname "$0")/env.sh" 2>/dev/null
cd ${GO_ROOT:-/home/jtan/solardefect/go_20261113}/src
CUDA_VISIBLE_DEVICES=1 python train_lm.py lm4 --epochs 8 --d 1024 --layers 16 --heads 16 --lr 0.0003 --seed 4 > ../logs/train_lm4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_lm.py lm5 --epochs 8 --d 896 --layers 14 --heads 14 --lr 0.0003 --seed 5 > ../logs/train_lm5.log 2>&1 &
wait
# lm6 d1280 L20 h20 seed6 (1117 06:40)
# CUDA_VISIBLE_DEVICES=0 python train_lm.py lm6 --epochs 8 --d 1280 --layers 20 --heads 20 --lr 0.0003 --seed 6
