#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" src/train_idft.py c33 --init lm4 --epochs 10 --lr .0001 --s 20 --m .05 --seed 121 --allp 1
"$PY" src/train_idft.py c34 --init lm4 --epochs 10 --lr .00015 --s 20 --m .1 --seed 122 --allp 1
# Inference, fusion, and verifier fitting are performed by reproduce_task1_1117.py.
