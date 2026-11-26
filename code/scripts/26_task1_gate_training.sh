#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" src/train_lm.py lm6 --epochs 8 --d 1280 --layers 20 --heads 20 --bs 128 --lr .0003 --seed 6
"$PY" src/train_ft.py ft6_sord8 --init lm4 --loss sord --tau 1.0 --epochs 8 --bs 128 --lr .00008 --seed 0 --pool both
"$PY" src/train_ft.py ft7_sord8 --init lm5 --loss sord --tau 1.5 --epochs 8 --bs 128 --lr .00008 --seed 1 --pool both
"$PY" src/train_ft.py ft8_sord8 --init lm6 --loss sord --tau 1.0 --epochs 8 --bs 128 --lr .00008 --seed 2 --pool both
"$PY" src/train_ft.py ft9_sord8 --init lm5 --loss sord --tau 1.0 --epochs 8 --bs 128 --lr .00008 --seed 3 --pool both
