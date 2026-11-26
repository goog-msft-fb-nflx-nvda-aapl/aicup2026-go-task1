#!/usr/bin/env bash
set -euo pipefail
: "${GO_ROOT:?Set GO_ROOT to the directory containing official data and baseline results}"
export GO_ROOT PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$(dirname "$0")/../src"
python train_idft.py cx35_lm6 --init lm6 --epochs 8 --lr .0001 --s 20 --m .05 --seed 1818 --allp 1
