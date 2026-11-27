# Task 1 saved-weight inference instructions

Version: `reproduction-20261127`. The report and this guide share `config/reproduction_contract.json`.

## Current status

All 27 model/archive/license downloads are public. No retraining is required. This includes the fine-tuned identity models, parent language models (LM1–LM6), fitted SVD/IDF and MLP transforms, rank models, HumanSL, policy MLP and fitted LDA.

**A complete original-CSV → accepted-CSV run has not passed.** The original test data are currently unavailable locally; historical parent orchestration also needs completion and the documented legacy-view prediction discrepancy needs resolution. The component commands below do not certify the complete prediction. The previously verified saved-feature/score replay is a different test.

## 1 Code and environment

```bash
git clone --branch reproduction-20261127 https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1.git
cd aicup2026-go-task1
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/requirements-training-recorded.txt
export GO_ROOT=/absolute/path/to/new-go-inference
export DOWNLOADS=/absolute/path/to/weight-downloads
export PYTHONHASHSEED=0
```

The environment file records the original versions; the name does not mean training must run. Fresh installation of this exact full inference environment has not been independently verified. Use a CUDA GPU and run a real tensor-operation smoke test before model inference.

## 2 Anonymous weight download and restoration

```bash
python tools/download_weights.py --check-public
python tools/download_weights.py --directory "$DOWNLOADS"
python tools/restore_weights.py --directory "$DOWNLOADS" --root "$GO_ROOT"
```

The downloader verifies all 27 original SHA256 values. The restorer verifies every downloaded file again, concatenates parent archive parts in numeric order as gzip tar and gate parts as uncompressed tar, and restores their model files into `$GO_ROOT/results`. Standalone files use the `restore_path` values in `config/public_weights.json`. A fresh model workspace is required; existing models are never silently overwritten.

The selected cx40 checkpoint is **epoch index 2**, SHA256 `6d44cc780541c286720e715645088b9695bde485ca8861516ea4ab6096040749`. Other epochs must not replace it. HumanSL SHA256 is `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`. Parent/fitted weights are restored, not retrained.

## 3 Official test inputs

The organizer supplies these files. Training CSVs are not needed for saved-weight inference and are not redistributed. Place all nine files under `$GO_ROOT/data/tests1v2/tests/`:

```text
sample_submission_player_v2.csv
player_identification_candidates_public.csv
player_identification_candidates_private.csv
player_identification_test_public.csv
player_identification_test_private.csv
player_identification_candidates2_public.csv
player_identification_candidates2_private.csv
player_identification_test2_public.csv
player_identification_test2_private.csv
```

```bash
python tools/check_official_data.py --root "$GO_ROOT" --inventory config/official_input_hashes.json --tests-only
```

If any official input is missing or different, stop. No cached score matrix, parent prediction CSV or saved feature file substitutes for original-data inference.

## 4 Source and HumanSL setup

```bash
mkdir -p "$GO_ROOT/src" "$GO_ROOT/sub" "$GO_ROOT/logs" "$GO_ROOT/reproduction"
cp code/src/*.py "$GO_ROOT/src/"
git clone https://github.com/lightvector/KataGo.git "$GO_ROOT/katago/kgsrc"
git -C "$GO_ROOT/katago/kgsrc" checkout d91ea855110dae533f0aada947b2b7d78cc8a4e1
export PYTHONPATH="$GO_ROOT/src:$GO_ROOT/katago/kgsrc/python"
```

HumanSL network and its license are restored under `$GO_ROOT/katago/` in step 2. Identity component scripts also load parent LM checkpoints for architecture settings, even though inference uses fine-tuned identity weights. Keep those parent models.

## 5 Component inference commands

These commands extract the cx40 component and the two HumanSL test-feature pools from original CSVs. They are **partial inference**, not a complete accepted-submission runner. Run in a clean workspace. The training/fitting scripts are not part of these steps.

```bash
CUDA_VISIBLE_DEVICES=0 python "$GO_ROOT/src/codex_infer_ablation_1118.py" --root "$GO_ROOT" --tag cx40_lm4_noaug_all
CUDA_VISIBLE_DEVICES=0 python "$GO_ROOT/src/codex_policy_test_features_1119.py" --root "$GO_ROOT" --pool pub2
CUDA_VISIBLE_DEVICES=0 python "$GO_ROOT/src/codex_policy_test_features_1119.py" --root "$GO_ROOT" --pool priv2
```

The accepted updated-pool fusion is `z(cx40 coordinate fusion) + 0.25*z(policy MLP cosine) + 0.10*z(expanded Gaussian)`. Original-pool predictions follow the frozen parent pipeline. The 34 parent members, 76 weighted entries and cx40 increment of weight 3 are recorded in `config/coordinate_fusion.json`; model arguments are in `config/parent_checkpoint_configs.json` and `config/gate_checkpoint_configs.json`.

## 6 Acceptance criteria and unresolved work

The final prediction must contain 1,600 questions and five unique candidate IDs per question, and its bytes must match SHA256:

```text
eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26
```

There is currently no certified single command for original-data inference to this exact file. `code/src/codex_reproduce_best1123.py` requires saved coordinate scores, policy features and a parent CSV; it is a replay/fusion runner. Historical parent wrappers also reference configurations and verifier inputs not covered by this restoration procedure. These scripts must not be described as a complete clean inference pipeline. Raw test data alone will not resolve those orchestration and numerical differences automatically.

Historical training/replay commands are preserved separately in `references/HISTORICAL_REPRODUCE_20261126.md`. They describe provenance and previous tests, not an alternative promise of exact raw inference.
