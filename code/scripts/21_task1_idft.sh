#!/bin/bash
# Task 1 main model: end-to-end ArcFace fine-tune of the pretrained LM (train_idft.py), 16 models; then pool scoring + ensemble + final csv (=id_v2_idftI, website 0.8528; idftH 0.8518 is the same with 14 models)
set -e; source "$(dirname "$0")/env.sh"; export CUDA_VISIBLE_DEVICES=$GPU
T="$PY -u src/train_idft.py"
$T a2  --init lm3 --epochs 2  --s 20 --m 0.05 --lr 5e-5
$T b1  --init lm3 --epochs 4  --s 20 --m 0.1  --lr 1e-4
$T b2  --init lm2 --epochs 4  --s 20 --m 0.05 --lr 5e-5 --seed 1
$T b3  --init lm1 --epochs 4  --s 20 --m 0.05 --lr 5e-5 --seed 2
# all-player runs (--allp 1: the 800 hold-out players are also used for training; no local validation possible)
$T c1  --init lm3 --epochs 6  --s 20 --m 0.1  --lr 1e-4   --allp 1 --seed 5
$T c2  --init lm2 --epochs 6  --s 20 --m 0.05 --lr 5e-5   --allp 1 --seed 6
$T c3  --init lm3 --epochs 6  --s 20 --m 0.05 --lr 5e-5   --allp 1 --seed 7
$T c4  --init lm3 --epochs 10 --s 20 --m 0.1  --lr 1e-4   --allp 1 --seed 21
$T c5  --init lm2 --epochs 8  --s 20 --m 0.1  --lr 1e-4   --allp 1 --seed 22
$T c6  --init lm3 --epochs 8  --s 24 --m 0.1  --lr 1e-4   --allp 1 --seed 31
$T c7  --init lm1 --epochs 8  --s 20 --m 0.1  --lr 1e-4   --allp 1 --seed 41
$T c8  --init lm2 --epochs 10 --s 20 --m 0.1  --lr 1.5e-4 --allp 1 --seed 42
$T c9  --init lm2 --epochs 12 --s 20 --m 0.1  --lr 2e-4   --allp 1 --seed 61
$T c10 --init lm3 --epochs 10 --s 20 --m 0.1  --lr 1.5e-4 --allp 1 --seed 62
$T c11 --init lm2 --epochs 10 --s 20 --m 0.1  --lr 1.5e-4 --allp 1 --seed 91
$T c12 --init lm3 --epochs 10 --s 20 --m 0.05 --lr 1e-4   --allp 1 --seed 92
for t in a2 b1 b2 b3 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12; do $PY src/idft_pools.py $t; done     # results/Sidft_<tag>_<pool>.npy
$PY src/ens_sidft.py ens7 a2 b1 b1 c1 c1 c3 c3 c2 b2 b3 c4 c4 c5 c5 c6 c6 c7 c8 c8 c8 c9 c9 c10 c10 c10 c11 c11 c12   # weights = repeats (ens6 = same without c11/c12 -> idftH 0.8518)
# gate/sinkhorn/csls fusion on top of the 5-view matrices:  args: base lam_new lam_old sinkhornT qT ens w_new w_old csls
$PY src/mk_gate4.py sub/base.csv sub/id_v2_idftI.csv 2 3 1.4 1.0 ens7 3 1 0.6          # = idftI (0.8528)
# per-query embeddings (for the query-graph step) and the final file = idftJ (0.8543): graph beta 0.5, cosine threshold 0.5, k=3 on the new pools
for t in a2 b1 b2 b3 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12; do $PY src/idft_qemb.py $t; done
$PY src/mk_gate5.py sub/base.csv sub/id_v2_idftJ.csv 2 3 1.4 1.0 ens7 3 1 0.6 0.5 0.5 a2,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 0
