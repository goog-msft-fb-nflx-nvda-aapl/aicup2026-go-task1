#!/bin/bash
# Task 1 base: hashed-token SVD + frozen-LM-feature MLP views, fused 5-view score matrices S_*.npy, and the rank-consistency gate inputs.
set -e; source "$(dirname "$0")/env.sh"; export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=24
$PY -u src/build_f6_cache.py
$PY -u src/svd_only.py 1000 > logs/svd1000.log 2>&1                                       # results/rank_svd1000.joblib, rank_svd1000_Z.npy
for s in 11 12 13 14 15 16; do ZNC=1000 $PY -u src/id_mlp.py 6 2048 0.2 0.3 0.1 $s > logs/id_mlp1000_s$s.log 2>&1; done   # args: EP H MARGIN DROP NOISE SEED
# frozen LM features -> MLPs  (ZNC = 2*d)
for lm in lm1 lm2 lm3; do $PY -u src/extract_lm.py $lm > logs/extract_$lm.log 2>&1; done
for cfg in "8 1024 0.2 0.2 0.05 0" "8 2048 0.3 0.3 0.1 0" "12 1024 0.2 0.3 0.1 1"; do ZNC=1024 $PY -u src/id_mlp.py $cfg > /dev/null 2>&1; done   # lm2
for cfg in "8 1024 0.2 0.2 0.05 0" "12 1024 0.2 0.3 0.1 1" "12 2048 0.2 0.3 0.1 2"; do ZNC=768  $PY -u src/id_mlp.py $cfg > /dev/null 2>&1; done   # lm1
for cfg in "12 1024 0.2 0.3 0.1 1" "12 2048 0.2 0.3 0.1 2" "8 1024 0.2 0.2 0.05 0"; do ZNC=1536 $PY -u src/id_mlp.py $cfg > /dev/null 2>&1; done   # lm3
# fine-tuned rank models used by the rank-consistency gate and by Task 2 (see 30_task2_ft.sh) must exist before this step
$PY src/rank_lp_pool.py     > logs/rank_lp_pool.log 2>&1       # results/LPc_*,LPq_* for the new pools
$PY src/rank_lp_pool_old.py > logs/rank_lp_pool_old.log 2>&1   # same for the original pools
# 5-view score matrices S_{pub,priv,pub2,priv2}.npy (+_pids): args = svd glob, -, -, base csv (row order only), out csv, w_svd, w_lm, w_max, old-flag, sinkhorn T
cp data/tests1v2/tests/sample_submission_player_v2.csv sub/base.csv
$PY src/id_final6.py "id_mlp1000_6_2048_0.2_0.3_0.1_1*.pt" x x sub/base.csv sub/id_v2_fuse6.csv 1 0 0.25 1 0.7 > logs/id_final6.log 2>&1
