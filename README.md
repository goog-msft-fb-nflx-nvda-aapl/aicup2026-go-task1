# AI CUP 2026 Go Task 1

Competition source and configuration for TEAM_10990. This repository contains no author contact table, portal credentials, organizer raw datasets. Trained checkpoint downloads are being published in the public release linked below.

## Result

Accepted prediction: public score **0.8721950**; private score **0.8855415**, rank **21/136** in the 25 November 2026 snapshot. These are discounted top-five ranking utilities, not accuracies. Accepted CSV SHA256: `eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26`.

## Source map

- `code/src/prep.py`, `gocommon.py`: official CSV/SGF processing.
- `code/src/train_lm.py`, `train_idft_crosscolour_1119.py`: move Transformer pretraining and identity fine-tuning; the latter concatenates own-move and all-valid-move pooled states.
- `code/src/codex_policy_residual_pilot_1119.py`, `codex_policy_test_features_1119.py`: frozen HumanSL residual extraction.
- `code/src/codex_policy_metric1119.py`: learned identity embedding.
- `code/src/codex_policy_expand1122.py`, `codex_policy_expand_fit1122.py`: expanded feature extraction and LDA/Gaussian fitting.
- `code/src/codex_reproduce_best1123.py`: accepted saved-intermediate replay.
- `config/`: exact selected hyperparameters, checkpoint argument records and input/code hashes.
- `REPRODUCE.md`: dataset layout, commands, expected outputs and verification scope.
- `requirements/`: recorded environment stacks; `licenses/` and `EXTERNAL_RESOURCES.md`: attribution.

## Verification scope

The archived compact replay was verified byte-for-byte from frozen coordinate scores and extracted policy features, with inherited original-family predictions. **This is not a passing full raw-CSV reconstruction.** The inherited legacy base view still changes some lower-ranked candidates when regenerated; a separately evaluated reconstruction scored 0.8670060. The documentation preserves this unresolved difference. Training from random seeds is not claimed to reproduce checkpoint bytes.

## Public model weights

Download without logging in: [trained-weights-20261127](https://github.com/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1/releases/tag/trained-weights-20261127). Publication is in progress. `config/public_weights.json` lists every expected asset and checksum; `tools/download_weights.py` refuses to report success if any are missing. Training is not needed when using these fitted weights. The full original-test-data inference acceptance test remains unresolved; public availability alone is not proof of a passing prediction test.

## Running

Place the organizer-provided files as shown in `REPRODUCE.md`, verify them with `python tools/check_official_data.py --root /absolute/project --inventory config/official_input_hashes.json`, and use the documented training/extraction commands. To verify the archived compact replay, install `requirements/requirements-replay.txt`, obtain the authorized archive and run `python tools/reproduce.py --archive /path/to/archive.tar.gz --workspace /path/to/empty-workspace`.

## Attribution and assistance

The system adapts Transformer and angular-margin identity learning, and uses an externally pretrained KataGo HumanSL model. HumanSL is frozen; task-specific extraction, residual averaging, identity adaptation, LDA fitting, variance shrinkage and fusion are documented in the source. Codex and Claude Code assisted implementation, experiment operations and writing; ChatGPT/Gemini supported research synthesis. The participant retains responsibility for the final claims and materials.
