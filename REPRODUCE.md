# Reproduction instructions for Task 1

## 1 Official dataset placement

The organizer supplies the official data; those CSVs are not redistributed here. Choose an absolute workspace:

```text
PROJECT_ROOT/
  src/                           copy code/src/*.py here
  scripts/                       copy code/scripts/*.sh here if using historical stage scripts
  data/training/train_A.csv ... train_D.csv
  data/training/train_1D.csv ... train_6D.csv
  data/tests1v2/tests/sample_submission_player_v2.csv
  data/tests1v2/tests/player_identification_candidates_public.csv
  data/tests1v2/tests/player_identification_candidates_private.csv
  data/tests1v2/tests/player_identification_test_public.csv
  data/tests1v2/tests/player_identification_test_private.csv
  data/tests1v2/tests/player_identification_candidates2_public.csv
  data/tests1v2/tests/player_identification_candidates2_private.csv
  data/tests1v2/tests/player_identification_test2_public.csv
  data/tests1v2/tests/player_identification_test2_private.csv
  results/                       restored trained parent models / declared intermediates
  reproduction/                  preserve archive-relative paths
  katago/humanv0.ckpt             external HumanSL network for raw extraction
  katago/kgsrc/                   external source pinned to d91ea855...
```

Check official CSVs with the supplied inventory (it records names/checksums, not dataset bytes):

```bash
python tools/check_official_data.py --root "$PROJECT_ROOT" \
  --inventory config/official_input_hashes.json
```

For inference, add `--tests-only` to the checker; the ten training CSVs are required only for training. No retraining is required to use the published weights.

## Public weight download (publication in progress)

```bash
python tools/download_weights.py --directory /absolute/path/to/downloads
```

This uses anonymous HTTPS only, verifies all expected SHA256 values, and fails if any required asset is not yet published. See `config/public_weights.json` for archive grouping and restore paths. Do not run training to compensate for a missing published checkpoint. The original-data inference acceptance test is not yet passing.

## 2 Exact saved-model replay already verified

Existing assets are still in the private backing repository; the canceled migration is not resumed:
https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect/releases/tag/go-organizer-package-20261124

Provide authorized reviewer access or hand over `organizer_saved_model_replay1124.tar.gz` directly. Expected archive SHA256:

```
b0b6f144a3c0961566472f680f72112e59df236f273a69aba60af57007f404c8
```

Install Python3.11.17 and the pinned recorded replay stack. The tested PyTorch build executes on CPU even though its distribution includes CUDA libraries:

```bash
python3.11 -m venv .replay-venv
. .replay-venv/bin/activate
python -m pip install -r requirements/requirements-replay.txt
python tools/reproduce.py \
  --archive /absolute/path/to/organizer_saved_model_replay1124.tar.gz \
  --workspace /absolute/path/to/EMPTY-workspace
```

The runner verifies22 input files, regenerates MLP embeddings and Gaussian matching, and checks the full1,600-row prediction. It uses declared frozen coordinate-fusion scores and policy features; original-family rows are inherited from an explicitly archived parent prediction. It does not silently treat these intermediates as raw-data inference.

Output: `EMPTY-workspace/reproduction/best1123/g23_greplace.csv` and `organizer_replay_receipt.json`.
Expected prediction SHA256:

```
eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26
```

`evidence/accepted_prediction.csv` has these same bytes. Existing private release `go-best-20261123` contains the accepted result and evaluation records.

## 3 Raw preprocessing and policy training commands

These are recorded reconstruction instructions; they are not a passing raw-data-only acceptance test. Use a separate restored training workspace, not the frozen replay directory. The fit/extraction scripts protect existing output directories.

```bash
export GO_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONHASHSEED=0
mkdir -p "$PROJECT_ROOT/src" "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/sub"
cp code/src/*.py "$PROJECT_ROOT/src/"
(cd "$PROJECT_ROOT/src" && SUBSAMPLE=1 python prep.py)
```

Restore the pinned HumanSL source/network under `katago/`, preserving its notices. Then:

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_test_features_1119.py" --root "$PROJECT_ROOT" --pool pub2
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_test_features_1119.py" --root "$PROJECT_ROOT" --pool priv2
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_residual_pilot_1119.py" --root "$PROJECT_ROOT" --training --players 1000
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_residual_pilot_1119.py" --root "$PROJECT_ROOT" --players 600
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_expand1122.py" --root "$PROJECT_ROOT" --training --players 2500 --shard 0
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_expand1122.py" --root "$PROJECT_ROOT" --training --players 2500 --shard 1
GO_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/codex_policy_metric1119.py"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python "$PROJECT_ROOT/src/codex_policy_expand_fit1122.py" --root "$PROJECT_ROOT"
```

Selection also requires aligned `results/HO_meta.npz` and `HO_{a2,b1,b2,b3,h4,h5}.npy`, plus the parent/control scores. These are derived model outputs, not hidden labels. They must be restored from the documented parent model pipeline or regenerated.

Coordinate pretraining, identity fine-tuning, rank-gate and ensemble commands are preserved in `code/scripts/` and the frozen guides under `references/`. Exact member repetitions and source/checkpoint settings are in `config/coordinate_fusion.json`, `parent_fusion_config.json`, `parent_checkpoint_configs.json` and `gate_checkpoint_configs.json`. Historical `00_setup.sh` also mentions Task2 setup; do not mistake that historical script for the Task1-only data layout above. Never run an upload helper as part of reproduction.

Accepted cx40 fine-tuning command, after LM4 and official training arrays are restored:

```bash
GO_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/src/train_idft_crosscolour_1119.py" cx40_lm4_noaug_all \
  --init lm4 --epochs 4 --lr .0001 --s 20 --m .1 --seed 0 --allp 1 --aug 0
```

The predefined selected checkpoint is epoch index2; archived weights ensure exact saved-model inference without rerunning stochastic training. Subsequent accepted score uses the MLP weight.25 and expanded Gaussian weight.10, with no older Gaussian or expanded cosine contribution.

## 4 Known verification limits

- Complete raw-data regeneration of the inherited legacy base view changes lower-ranked predictions. The accepted best is not certified as raw-CSV-only reproducible.
- The frozen policy extractor rejects HD>1 and uses `GameState.RULES_CHINESE`; the official FAQ describes Japanese-rule records. This implementation choice is disclosed, not changed after competition closing.
- Local audit identities were reused and the all-player LM4 increment is excluded from the local proxy. Local scores are not independent hidden-test estimates.
- A clean-directory replay was tested in the recorded installed environment. A fresh dependency installation and complete retraining were not independently verified by this preparation step.
