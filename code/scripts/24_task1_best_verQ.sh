#!/bin/bash
# Task 1 best submission id_v2_verQ.csv (website 0.8588). Run AFTER: 00_setup.sh, 10_pretrain_lms.sh, 30_task2_ft.sh, 20_task1_base_views.sh, 21_task1_idft.sh (members a2,b1,b2,b3,c1..c12), 22_lm4_lm5_pretrain.sh.
# Every command is a single-GPU job; the jobs below are independent and can be run in parallel on different GPUs (set GPU per job). Full descriptions: REPRODUCE_TASK1_BEST.md.
set -e; source "$(dirname "$0")/env.sh"; export CUDA_VISIBLE_DEVICES=$GPU
T="$PY -u src/train_idft.py"
# hold-out models used to train the verifier (no --allp)
$T h4 --init lm4 --epochs 4 --lr 1e-4 --s 20 --m 0.1 --seed 0
$T h5 --init lm5 --epochs 4 --lr 1e-4 --s 20 --m 0.1 --seed 0
# all-player ensemble members, 10 epochs, s=20
for cfg in "c13 lm3 1.5e-4 0.1 101" "c14 lm2 1.5e-4 0.1 102" "c15 lm3 2e-4 0.1 103" "c16 lm3 1e-4 0.05 104" "c17 lm3 1.5e-4 0.1 105" "c18 lm3 1e-4 0.05 106" \
           "c19 lm5 1.5e-4 0.1 107" "c20 lm4 1.5e-4 0.1 108" "c21 lm4 1.5e-4 0.1 109" "c22 lm4 1e-4 0.05 110" "c23 lm4 1.5e-4 0.1 111" "c24 lm4 1e-4 0.05 112" \
           "c25 lm4 1.5e-4 0.1 113" "c26 lm5 1e-4 0.05 114"; do set -- $cfg
  $T $1 --init $2 --epochs 10 --lr $3 --s 20 --m $4 --seed $5 --allp 1; done
ALL="a2 b1 b2 b3 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 c21 c22 c23 c24 c25 c26"
for t in $ALL; do $PY src/idft_pools.py $t; $PY src/idft_qemb.py $t; $PY src/idft_games.py $t; done     # Sidft_/Qemb_/G_ files per member
$PY src/ens_sidft.py ensQ a2 b1 b1 c1 c1 c3 c3 c2 b2 b3 c4 c4 c5 c5 c6 c6 c7 c8 c8 c8 c9 c9 c10 c10 c10 c11 c11 c12 c13 c13 c14 c14 c15 c15 c16 c16 c17 c17 c18 c18 c19 c19 c19 c20 c20 c20 c21 c21 c21 c22 c22 c22 c23 c23 c23 c24 c24 c24 c25 c25 c25 c26 c26 c26
LIST=$(echo $ALL | tr ' ' ',')
SFINAL=Q $PY src/mk_gate5.py sub/base.csv sub/id_v2_idftQ.csv 2 3 1.4 1.0 ensQ 3 1 0.6 0.5 0.5 $LIST 0      # sub/base.csv = sample submission, used only as row template
$PY src/holdout_embeds.py a2 b1 b2 b3 h4 h5                                                                  # hold-out embeddings for the verifier training
$PY src/mk_verifier.py sub/id_v2_idftQ.csv sub/id_v2_verQ.csv Q a2,b1,b2,b3,h4,h5 $LIST 1.0                  # FINAL: sub/id_v2_verQ.csv
