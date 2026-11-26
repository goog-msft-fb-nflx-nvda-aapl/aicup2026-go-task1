# Task 1 reproduction for November 17

The submitted file `go_c3334_1117.csv` scored **0.8590127** on AIdea at **2026-11-17 02:45:37 UTC**, rank 11/99 when checked. Its SHA-256 is `20fe47661017bd33b61cc076ae261ea371a3180ebc37e7ad8b32902844b808be`. The file contains 1,600 predictions, five candidate player IDs per question.

For the organizer-facing dataset layout and inference command, start with the README section **Reproduce Task 1**.

## Access and files

Use the private repository `goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect`, folder `go`, and release `task1-reproduction-20261117`. The team must grant the organizer access or deliver the same source and release files through the organizer's submission channel. The release remains private during the competition.

The release contains `task1-score-artifacts-1117.tar.gz`, split trained-weight archive files `task1-weights-1117.tar.gz.part*`, `inputs_manifest.json`, `checkpoint_configs.json` and `release_assets.json`. Individual file hashes, data hashes and checkpoint settings bind the artifacts to this submission. Official test and training CSVs must be supplied from the organizer's data packages. Account credentials are unnecessary for local inference.

## Environment

The measured runtime uses Python 3.11.17, PyTorch 2.6.0+cu124, NumPy 2.3.5, SciPy 1.17.1, scikit-learn 1.9.0, LightGBM 4.7.0 and joblib 1.6.0. The dependency file is `requirements.txt`; CUDA PyTorch wheels are installed from the CUDA 12.4 wheel index. Hardware details and measured versions are recorded in `reproduction/environment_1117.json`. Inference uses one CUDA GPU at a time. The base SVD object alone is approximately 10 GB, so allocate sufficient CPU RAM and disk for the full checkpoint archive and extracted files.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export PYTHONHASHSEED=0
export GO_ROOT=/absolute/path/to/go_verification
mkdir -p "$GO_ROOT"/data/tests1v2/tests
```

Place the official Task 1 v2 test CSV files, including `sample_submission_player_v2.csv`, under `$GO_ROOT/data/tests1v2/tests`. Full inference also needs official training CSVs under `$GO_ROOT/data/training`, because the existing preprocessing and held-out embedding stages use them.

## Download and verify the private release

Run these commands from the repository's `go` folder after authenticating GitHub as a user authorized for the private repository.

```bash
mkdir -p release_1117
gh release download task1-reproduction-20261117 \
  --repo goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect \
  --dir release_1117
python src/verify_task1_assets.py release_1117
tar -xzf release_1117/task1-score-artifacts-1117.tar.gz -C "$GO_ROOT"
```

## Regenerate the exact CSV from archived score inputs

```bash
python src/reproduce_task1_1117.py --root "$GO_ROOT" --mode artifacts
```

This recomputes the original verQ ensemble and its original-family predictions, the c33/c34-augmented ensemble, rank gate, query graph and LambdaRank verifier. The verifier is fitted again from saved held-out embeddings. The command checks the complete generated CSV against the submitted SHA-256 and writes `$GO_ROOT/results/reproduction_1117_verification.json`. It reads the official sample CSV for question order. A previously generated prediction CSV is unnecessary as an input.

This check exercises final scoring from intermediates. Trained-weight inference and from-scratch training have their own procedures below.

## Recompute the intermediates from the archived trained weights

```bash
cat release_1117/task1-weights-1117.tar.gz.part* | tar -xzf - -C "$GO_ROOT"
export CUDA_VISIBLE_DEVICES=0
python src/reproduce_task1_1117.py --root "$GO_ROOT" --mode inference
```

This regenerates the base SVD/MLP views, rank-consistency gate inputs, per-model pool scores and query/per-game embeddings, then fits the verifier and checks the final CSV hash. The held-out LM/ID models are frozen. GPU numerical differences can change close rankings. If a hash differs, retain the generated file and logs for investigation, and compare intermediate hashes and predicted IDs with the accepted archive.

## Train the models from the competition data

The older verQ training recipe remains in `REPRODUCE_TASK1_BEST.md` and `scripts/24_task1_best_verQ.sh`. Run the numbered setup, LM, rank-gate, base-view and ID-training scripts described in the README, followed by `scripts/25_task1_c3334_training.sh`. The base-view script now explicitly builds `results/F6_csr.npz` with `build_f6_cache.py` before SVD fitting. All six token flags are enabled, the cache stores binary indices, and `PYTHONHASHSEED=0` is required. Four cached training rows were independently reconstructed with seed zero and matched their original token sets.

Exact checkpoint dictionaries and original hyperparameters are in `checkpoint_configs.json`. c33 uses lm4, 10 epochs, learning rate .0001, scale 20, margin .05, seed 121 and all-player training. c34 uses lm4, 10 epochs, learning rate .00015, scale 20, margin .1, seed 122 and all-player training. Their logged NEW-like training diagnostics include training players and must be interpreted accordingly. The verifier uses a2/b1/b2/b3/h4/h5 embeddings from models excluding the fixed 800-player hold-out.

The accepted ensemble adds three repetitions each of c33 and c34 to the original verQ weighted member list. The query graph retains the original 30 members. The original-family rows are regenerated using the original verQ list. Machine-readable lists are in `reproduction/task1_1117_config.json`.

From-scratch training can produce different weights through GPU numerical nondeterminism. The archived original trained weights and their hashes preserve the accepted experiment.

## Sources and final report

Task 1's accepted model uses the official competition games, AIdea test packages and project LMs trained from scratch. The official training source is `AILAB-NDHU/aicup2026-go-dataset` release v1.1. KataGo HumanSL was used in a rejected Task 1 variant and Task 2, with attribution retained in the existing report. Claude Code and OpenAI Codex contributed code and experiment assistance and are disclosed in the report draft. Source libraries and their installed versions are listed in the environment and dependency files.

`FINAL_REPORT_DRAFT.md` includes the updated Task 1 algorithm and score. Team members and advisor fields require team completion. The final model is selected on November 24, so update the final report, exact submission hash, release and verification record if the accepted model changes.

## Verification record

The original submitted candidate passed CSV integrity checks, original-family preservation checks and actual AIdea evaluation. A fresh-directory score-artifact replay passed on November 17 using the archived inputs and portable source, with all 1,600 rows and the full CSV SHA-256 identical. The exact record is `reproduction/replay_verification_1117.json`. The official test CSV directory was reused as an input; generated scores and predictions were rebuilt in the separate verification directory. The runtime used the recorded server environment. Full checkpoint-to-CSV inference and full from-scratch training require separate clean runs before their verification can be claimed.

The checkpoint archive was streamed and every one of its 57 files matched the original file size and SHA-256, totaling 28,411,174,587 uncompressed bytes. The result is `reproduction/weight_archive_verification_1117.json`. Release-asset hashes are also committed in `reproduction/release_assets_1117.json`.

A CPU FP32 smoke check loaded the c33 checkpoint and regenerated ten production reference-game embeddings. Minimum cosine agreement with the cached CUDA BF16 embeddings was 0.9999921, with maximum absolute coordinate difference 0.0007111. This checkpoint check is recorded in `reproduction/checkpoint_smoke_1117.json`. The complete checkpoint-to-CSV inference path still requires its separate clean run.
