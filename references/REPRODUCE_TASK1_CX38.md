# Reproduce Task 1 public score 0.8621245

The verified November 19 upload is go19_cx38.csv. SHA-256 is `0b34e38881dc40ba224d0e188f678d8d2f68b6bf3d76565ed3a083dfbe9ff5ca`. It adds one orientation-preserving held-out lm4 model at repetition weight three to the previous 0.8605620 ensemble. Its selected checkpoint is epoch index 2 of cx38_lm4_aug0_bi0, trained for four epochs with lm4 initialization, lr 0.0001, scale 20, margin 0.1, seed 0, allp 0 and aug 0. Candidate selection used local validation. This public score is independently verified by the website receipt in reproduction/20261119.

## Environment, official data and parent assets

Use the Python3.11, PyTorch2.6 CUDA environment, official dataset filenames and private baseline asset downloads in REPRODUCE_TASK1_CX35.md. Clone source tag go-best-cx38-20261119. The original dataset-to-prediction baseline check is still running. Its cached-input reproduction has passed byte-for-byte.

Download the review experiment archives and checkpoint.

```bash
mkdir -p review_assets restore
gh release download go-review-pipeline-20261118 \
  --repo goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect \
  --pattern cx38_training_complete.tar.gz --pattern cx38_inference_complete.tar.gz \
  --pattern idft_cx38_selected.pt --dir review_assets
tar -xzf review_assets/cx38_training_complete.tar.gz -C restore
tar -xzf review_assets/cx38_inference_complete.tar.gz -C restore
export GO_ROOT="$PWD/restore/reproduction/cx38_candidate_pipeline1118"
mkdir -p "$GO_ROOT/results" "$GO_ROOT/data/tests1v2"
cp review_assets/idft_cx38_selected.pt "$GO_ROOT/results/"
cp -R src/. "$GO_ROOT/src/"
```

Place the official Task 1 v2 test CSVs under `$GO_ROOT/data/tests1v2/tests`. Restore the original baseline trained weights and score archives, gate delta and cx35 score delta into `$GO_ROOT`, following REPRODUCE_TASK1_CX35.md. Those commands create the required `results/` files and frozen verifier. The inference archive above supplies the eight new selected-member score/embedding arrays. The training archive includes the accepted parent CSV and exact configuration.

The selected checkpoint SHA-256 is `8614b9c681ab8fb5986f7de79224c6ff7dc37f805800c9b7147438d6702d8e08`. Training archive SHA-256 is `f53cfd06ec2168003ff4c9df439d92c8927f1f64d2aa25e92a2e879847fe357b`. Inference archive SHA-256 is `e95d63b2665a7ddf79802f21a37b014b7a1755640983edd53f6dd85ad7e5ba98`. Check downloaded files against these hashes and GitHub release asset digests.

## Execute

Regenerate the added member from official tests, then run the accepted fusion and frozen verifier against the restored parent score artifacts.

```bash
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0
python src/codex_build_review_candidate_1118.py --root "$GO_ROOT" --mode inference
sha256sum "$GO_ROOT/sub/go19_cx38.csv"
```

For the exact cached-input replay, use `--mode artifacts`. Output must match the SHA-256 at the top of this guide. The runtime also validates question IDs, 1600 rows, unique answers and preservation of original-family predictions. This command reproduces the added-member inference plus production fusion using the saved parent artifacts. The full raw-data regeneration of every parent component remains a separate verification stage.

For retraining, follow the parent training recipes, then run the matching held-out ablation command below. Select its saved ep2 checkpoint as idft_cx38_selected.pt. Original weights are provided because retraining can vary across kernels.

```bash
python src/train_idft_ablation_1118.py cx38_lm4_aug0_bi0 \
  --init lm4 --epochs 4 --lr .0001 --s 20 --m .1 --seed 0 --allp 0 --aug 0
```

Training requires official v1.1 CSVs in data/training and `python src/prep.py` to construct data/train_arr.npz. Inference requires no hidden labels. External software and generative-AI assistance are recorded in the report draft and review experiment notes. November 24 final competition upload is reserved for the user's manual action.
