#!/bin/bash
# Data + external weights. Run once. Needs ~2 GB disk for data, 0.5 GB for KataGo.
set -e
SD="$(cd "$(dirname "$0")" && pwd)"; source "$SD/env.sh"
cp "$SD"/../src/*.py $GO_ROOT/src/
# 1) official training data (AILAB-NDHU/aicup2026-go-dataset release v1.1): 1,000,000 games, 10 ranks x 100k
cd $GO_ROOT/data
[ -f aicup2026_go_training_v1.1.tar.xz ] || curl -L -o aicup2026_go_training_v1.1.tar.xz https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/download/v1.1/aicup2026_go_training_v1.1.tar.xz
tar xJf aicup2026_go_training_v1.1.tar.xz      # csv files must end up as $GO_ROOT/data/training/train_*.csv (move them if the tar uses another folder)
# 2) test sets: download the competition files from the AIdea pages (login required) and unzip so that
#    data/tests1v2/tests/player_identification_{candidates,candidates2,test,test2}_{public,private}.csv and
#    data/tests2/tests/rank_prediction_test_{public,private}.csv exist (Task 1 v2 package and Task 2 package).
# 3) KataGo human-SL network (EXTERNAL WEIGHTS, declared): b18c384nbt-humanv0 (release v1.15.0 asset)
cd $GO_ROOT/katago
[ -f humanv0.ckpt ] || curl -L -o humanv0.ckpt https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.ckpt
# KataGo python sources must have the python/katago/{game,train} package layout (NOT the flat v1.15.0 tag layout). The exact source used
# (files python/katago/game/sgfmetadata.py and train/load_model.py verified identical by md5) is master commit d91ea855110dae533f0aada947b2b7d78cc8a4e1:
if [ ! -d kgsrc ]; then
  curl -L -o kg.zip https://github.com/lightvector/KataGo/archive/d91ea855110dae533f0aada947b2b7d78cc8a4e1.zip && unzip -q kg.zip && mv KataGo-d91ea855110dae533f0aada947b2b7d78cc8a4e1 kgsrc
fi
# 4) CSV -> arrays
cd $GO_ROOT && $PY src/prep.py            # writes data/train_arr.npz
