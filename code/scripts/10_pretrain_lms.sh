#!/bin/bash
# Rank-conditioned causal move LMs (all start from scratch on the 1M training games, hold-out 800 players inside train_lm.py).
set -e; source "$(dirname "$0")/env.sh"; export CUDA_VISIBLE_DEVICES=$GPU
$PY -u src/train_lm.py lm1 --epochs 4                                                    > logs/train_lm1.log 2>&1   # d384 x 8L (defaults)
$PY -u src/train_lm.py lm2 --epochs 8 --d 512 --layers 10 --heads 8 --lr 3e-4            > logs/train_lm2.log 2>&1
$PY -u src/train_lm.py lm3 --epochs 8 --d 768 --layers 12 --heads 12 --lr 2.5e-4 --bs 96 > logs/train_lm3.log 2>&1
# outputs results/lm_{lm1,lm2,lm3}.pt  (checkpoint dict has 'args' and the 'hold' player list used everywhere else)
