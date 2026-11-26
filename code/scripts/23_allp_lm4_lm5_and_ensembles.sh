#!/bin/bash
# Task 1: all-player (--allp 1) fine-tunes from the bigger LMs, post-processing chain, ensemble idftP, verifier. Run from go_20261113/src on gsm-gpu2.
# PY = python of codex_20261028/venv ; GPU index via CUDA_VISIBLE_DEVICES (smoke-test with a real tensor op first).
PY=/home/jtan/solardefect/codex_20261028/venv/bin/python
# member:  tag init lr margin seed      (all: --epochs 10 --s 20 --allp 1)
#  c20 lm4 1.5e-4 .1 s108 | c21 lm4 1.5e-4 .1 s109 | c22 lm4 1e-4 .05 s110 | c23 lm4 1.5e-4 .1 s111 | c24 lm4 1e-4 .05 s112
#  c25 lm4 1.5e-4 .1 s113 | c26 lm5 1e-4 .05 s114   (c19 = lm5 1.5e-4 .1 s107)
train_post(){ c=$1; init=$2; lr=$3; m=$4; seed=$5   # run with CUDA_VISIBLE_DEVICES set
  $PY train_idft.py $c --init $init --epochs 10 --lr $lr --s 20 --m $m --seed $seed --allp 1 > ../logs/idft_$c.log 2>&1
  post $c; }
post(){ c=$1
  $PY idft_pools.py $c > ../logs/pools_$c.log 2>&1; $PY idft_qemb.py $c > ../logs/qemb_$c.log 2>&1
  $PY idft_games.py $c > ../logs/games_$c.log 2>&1; touch ../logs/${c}_post.done; }
# ensemble idftP (weights = repetition count; lm4/lm5 members x3); add c23..c26 when post is done
E="a2 b1 b1 c1 c1 c3 c3 c2 b2 b3 c4 c4 c5 c5 c6 c6 c7 c8 c8 c8 c9 c9 c10 c10 c10 c11 c11 c12 c13 c13 c14 c14 c15 c15 c16 c16 c17 c17 c18 c18 c19 c19 c19 c20 c20 c20 c21 c21 c21 c22 c22 c22"
build_P(){ $PY ens_sidft.py ensP $E
  SFINAL=P $PY mk_gate5.py ../sub/id_v2_idftJ.csv ../sub/id_v2_idftP.csv 2 3 1.4 1.0 ensP 3 1 0.6 0.5 0.5 a2,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22 0; }
# verifier (LightGBM lambdarank over top-30; trained on hold-out embeddings of hold-out models; needs G_<tag>_* from idft_games.py)
#  $PY mk_verifier.py ../sub/id_v2_idftP.csv ../sub/id_v2_verP.csv P a2,b1,b2,b3,h4,h5 a2,b1,b2,b3,c1,...,c22 1.0
# dry run on idftJ: $PY mk_verifier.py ../sub/id_v2_idftJ.csv ../sub/id_v2_verJ2.csv J a2,b1,b2,b3,h4,h5 a2,b1,b2,b3,c1,...,c19 1.0   (top-1 changed 12.3% pub2 / 14.0% priv2)
#  c27 lm4 1.5e-4 .1 s115 | c28 lm5 1.5e-4 .1 s116 | c29 lm4 1.5e-4 .1 s117 | c30 lm4 1e-4 .05 s118
# idftQ: E = idftP list + "c23 c23 c23 c24 c24 c24 c25 c25 c25 c26 c26 c26", tag ensQ, SFINAL=Q; verQ: mk_verifier.py ../sub/id_v2_idftQ.csv ../sub/id_v2_verQ.csv Q a2,b1,b2,b3,h4,h5 <a2,b1,b2,b3,c1..c26> 1.0   (website 0.8588)
