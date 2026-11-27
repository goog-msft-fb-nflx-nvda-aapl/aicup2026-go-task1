# Task 1 saved-weight inference instructions

Version: `inference-20261127`. The report and this guide share `config/reproduction_contract.json`.

## Current status

All 27 model/archive/license downloads are public. No retraining is required. This includes the fine-tuned identity models, parent language models (LM1–LM6), fitted SVD/IDF and MLP transforms, rank models, HumanSL, policy MLP and fitted LDA.

**The complete inference command is implemented; official-data exactness is not yet validated.** It runs 41 stages using 66 restored model files, starting from the nine official test CSVs. No training, historical prediction CSV, saved feature cache or archived score matrix is an inference input. The runner generates all features/scores within its fresh workspace and accepts success only when the final CSV matches the accepted SHA256. The local official test files are unavailable; an earlier legacy reconstruction difference remains to be checked when they are restored.

## 1 Code and environment

```bash
git clone --branch inference-20261127 https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1.git
cd aicup2026-go-task1
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/requirements-inference.txt
export GO_ROOT=/absolute/path/to/new-go-inference
export DOWNLOADS=/absolute/path/to/weight-downloads
export PYTHONHASHSEED=0
```

The inference requirements omit training-only and verifier-only dependencies. Training must not run. Fresh installation of this exact full inference environment has not been independently verified. Use a CUDA GPU and run a real tensor-operation smoke test before model inference.

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

## 5 Complete inference

From the cloned repository, after steps 1–4:

```bash
python tools/infer.py --root "$GO_ROOT" --gpu 0 --preflight-only
python tools/infer.py --root "$GO_ROOT" --gpu 0
```

`--gpu 0` chooses physical GPU 0; change it to one available GPU. Do not run on a busy shared GPU without allocation. The preflight checks all nine official test files, all 66 model paths, standalone model hashes, the pinned KataGo source and a real CUDA tensor operation. It records input/model/source hashes and environment details. A fresh model workspace must contain no previously generated `.npy` scores, policy feature cache or prediction CSV.

The 41 stages generate:

1. The four legacy base-score matrices from SGFs, frozen SVD/IDF, base LMs and MLPs.
2. Original-pool rank signals from ft4/ft5 and updated-pool rank signals from ft6–ft9.
3. The 34 parent identity members plus selected cx40; original graph query embeddings are generated in the same pass. Legacy members use four symmetries; cx38/cx40 respect checkpoint augmentation settings, including cx40's one orientation.
4. HumanSL residual features for both updated pools from actual board replay.
5. Original-pool fusion from the 64 original weighted entries; updated fusion from the 76 parent entries plus cx40 three times, with rank gating, candidate correction and query graph. The final updated score adds `.25*z(policy MLP cosine) + .10*z(expanded Gaussian)` to the standardized coordinate score. No LambdaRank verifier or verifier fitting is needed.
6. All 1,600 top-five rows in the official sample order, without inheriting prediction rows from an earlier CSV.

Output:

```text
$GO_ROOT/sub/g23_greplace.csv
$GO_ROOT/sub/prediction_verification.json
$GO_ROOT/logs/<stage>.log
$GO_ROOT/inference/<stage>.receipt.json
```

A nonmatching CSV is retained for diagnosis, but the command exits with failure; it never reports success merely because a file exists. Success prints `EXACT_ACCEPTED_CSV_REPRODUCED` only after re-reading and hashing the actual output file.

For a disconnected/interrupted controller with unchanged code, data, models, GPU and output path:

```bash
python tools/infer.py --root "$GO_ROOT" --gpu 0 --resume
```

The runner monitors confirmed live stages, excludes duplicate workers with locks, and reuses only recorded outputs whose hashes still match. It does not blindly retry failed/uncertain stages. Inspect their logs and receipts first; a changed implementation requires a fresh run workspace.

To inspect the entire command plan without data, models, Torch or a GPU:

```bash
python tools/infer.py --root "$GO_ROOT" --plan
```

## 6 Acceptance criteria and unresolved work

The final prediction must contain 1,600 questions and five unique candidate IDs per question, and its bytes must match SHA256:

```text
eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26
```

The complete command above is implemented, but the official-data checksum run has not been executed because the local test CSVs are missing. An earlier legacy base-view reconstruction differed; implementation completeness must not be described as proof of exact output. `code/src/codex_reproduce_best1123.py` remains a separate historical replay/fusion runner and is not called by the new pipeline.

Checks already performed on the new implementation:

- Pure coordinate fusion is bit-identical to historical gate scripts on synthetic original/updated pools.
- Against genuine retained intermediate arrays, updated coordinate matrices are bit-identical and all 800 updated-pool top-five predictions match the accepted submission. This is an intermediate regression, not raw-data inference.
- Selected epoch-2 cx40 and frozen HumanSL each passed real-GPU forward checks on explicitly synthetic legal SGFs. HumanSL produced 432 channels; its smoke mode writes no production feature file.
- Persistent worker tests verified duplicate exclusion, completion receipts, output hashes and failure when an expected output is missing.

Optional data-free regression checks after installing the inference dependencies:

```bash
python tests/check_fusion_against_reference.py
python tests/check_stage_controller.py
```

Historical training/replay commands are preserved separately in `references/HISTORICAL_REPRODUCE_20261126.md`. They describe provenance and previous tests, not an alternative promise of exact raw inference.
