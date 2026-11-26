# Reproduce Task 1 score 0.8598258

The accepted file is `go18_gate.csv`, evaluated by AIdea at 2026-11-17 16:25:40 UTC, which is November 18 at 01:25:40 JST. Its SHA-256 is `ae7cf8b835527a9058fcd6f219aea8cc433de3f69eb4e1f6b24f9e727cc61d3d`. It contains 1,600 questions and five ranked candidate IDs per question.

Fetch the frozen source with `git clone --branch go-best-20261118 https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect.git`, then enter its `go/` directory.

## Environment and dataset

Run from the repository's `go/` folder on Linux with Python 3.11 and a CUDA GPU. The measured runtime is in `reproduction/20261118/environment.json`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export GO_ROOT=/absolute/path/to/go_verification
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
```

Place the organizer's Task 1 v2 CSV files in this structure. Keep each filename intact.

```text
go_verification/
└── data/
    └── tests1v2/
        └── tests/
            ├── player_identification_candidates_public.csv
            ├── player_identification_candidates_private.csv
            ├── player_identification_candidates2_public.csv
            ├── player_identification_candidates2_private.csv
            ├── player_identification_test_public.csv
            ├── player_identification_test_private.csv
            ├── player_identification_test2_public.csv
            ├── player_identification_test2_private.csv
            └── sample_submission_player_v2.csv
```

For retraining, place the ten official v1.1 training files `train_D.csv` through `train_6D.csv` under `$GO_ROOT/data/training`. The official release checksum is recorded in the dataset audit.

## Trained weights

The private repository is `goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect`. The organizer needs authorized repository access or delivery of the same source and asset files through the organizer's submission channel.

Download the original 57 trained model files from release `task1-reproduction-20261117` and the five additional checkpoints from release `go-best-20261118`. Verify the downloaded files against each release's `release_assets.json` before extracting. The additional files are lm6 and ft6_sord8 through ft9_sord8, with exact training arguments in `reproduction/20261118/checkpoint_configs.json`.

```bash
mkdir -p baseline_assets gate_assets "$GO_ROOT/results"
gh release download task1-reproduction-20261117 \
  --repo goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect --dir baseline_assets
gh release download go-best-20261118 \
  --repo goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect --dir gate_assets
python src/verify_task1_assets.py baseline_assets
python src/verify_task1_assets.py gate_assets
cat baseline_assets/task1-weights-1117.tar.gz.part* | tar -xzf - -C "$GO_ROOT"
cat gate_assets/go18_gate_weights.tar.part* | tar -xf - -C "$GO_ROOT"
tar -xzf gate_assets/go18_gate_score_delta.tar.gz -C "$GO_ROOT"
```

The delta archive supplies the frozen LambdaRank verifier at `$GO_ROOT/results/verifier_gate1118.txt`. All trained model and SVD files appear under `$GO_ROOT/results` after extraction. The new rank-gate source remains the same in the inference and cached-input procedures.

## Dataset-to-prediction command

```bash
python src/reproduce_task1_1117.py \
  --root "$GO_ROOT" \
  --config reproduction/task1_1118_config.json \
  --mode inference
```

This regenerates the base views, the original and updated rank-gate inputs, candidate/query embeddings and ensemble scores from the official test CSVs and trained weights. It then applies the frozen verifier and writes `$GO_ROOT/sub/go18_gate.csv`.

The program checks the complete CSV SHA-256 automatically. A successful run writes `$GO_ROOT/results/reproduction_1118_verification.json`. The expected score is 0.8598258 for the original hidden evaluation used on November 18. The code uses the same 32 production player-ID models as the previous accepted candidate. The new-family gate averages eight-symmetry log probabilities from ft6_sord8, ft7_sord8, ft8_sord8 and ft9_sord8. Original-family predictions are regenerated from the original verQ pipeline.

## Cached-input verification

```bash
tar -xzf baseline_assets/task1-score-artifacts-1117.tar.gz -C "$GO_ROOT"
tar -xzf gate_assets/go18_gate_score_delta.tar.gz -C "$GO_ROOT"
python src/reproduce_task1_1117.py \
  --root "$GO_ROOT" \
  --config reproduction/task1_1118_config.json \
  --mode artifacts
```

A fresh-directory cached-input replay passed on November 18 and reproduced every output byte. The record is `reproduction/20261118/replay_verification.json`. Each of the five additional archived checkpoints matched its original file hash, as recorded in `reproduction/20261118/weight_verification.json`. A complete clean dataset-to-prediction run remains a separate verification stage.

## Training and attribution

Run the baseline training recipe in `REPRODUCE_TASK1_1117.md`, followed by `scripts/26_task1_gate_training.sh`. The new script contains the exact lm6 and rank-model commands. Full training runs can vary numerically across GPU kernels, so the supplied original checkpoints and hashes preserve the accepted weights.

The accepted Task 1 model uses the official competition data and project networks trained from scratch. Software and generative-AI assistance are disclosed in `FINAL_REPORT_DRAFT.md`. The rejected HumanSL and h6 variants remain in the experiment records. The final November 24 candidate requires its own frozen config and verified output hash if it supersedes this version.
