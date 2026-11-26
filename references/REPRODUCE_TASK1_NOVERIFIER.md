# Reproduce Task 1 public score 0.8630397

The accepted candidate is **go19_nover.csv**, evaluated November 19, 2026. It preserves the original-family predictions from the cx38 parent and uses its pre-verifier fusion scores for the two new-family pools. Parent public score is 0.8621245. All embeddings, ensemble weights, rank gates and query graph settings are preserved. The final LambdaRank verifier is removed from the output path.

Expected CSV SHA-256 is `abf65aebd37460b1279dc03177e5755660ff05063099bcae9f778ad21d5a1fa7`. All five actual November 19 results are recorded in reproduction/20261119/submission_results.json.

## Restore environment, official data and checkpoints

Clone frozen source tag `go-best-noverifier-20261119`. Follow REPRODUCE_TASK1_CX38.md to install the recorded Python3.11/PyTorch2.6 CUDA environment, download original parent weights and score archives plus the exact selected checkpoint, and restore the cx38 working directory. Set GO_ROOT to that working directory, containing src/, results/, sub/, reproduction/review_candidate_config.json and official CSVs under data/tests1v2/tests. The guide specifies the exact filenames and checkpoint hashes. After extracting the training archive, overlay this frozen repository's go/src files onto `$GO_ROOT/src` as the parent guide describes.

The original trained weights and rank-gate checkpoints are in releases task1-reproduction-20261117 and go-best-20261118. The cx35 checkpoint and score delta are in go-cx35-experiment-20261118. The selected orientation-preserving checkpoint, training source/settings/logs and inference arrays are in go-review-pipeline-20261118. The pre-submission verifier-removal source, predictions and exact fusion score arrays are in go-experiments-20261119/go19_noverifier_sources.tar.gz. All assets are private and require authorized access.

## Run

```bash
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0
python src/codex_noverifier_1119.py --root "$GO_ROOT" --mode inference
sha256sum "$GO_ROOT/sub/go19_nover.csv"
```

The command regenerates the selected member's embeddings from official tests, recomputes parent fusion from restored baseline artifacts, validates the exact parent, removes the verifier from the final predictions and checks the complete best CSV hash. For a cached-input replay use `--mode artifacts`. Both routes perform the same final fusion. This is the added-member inference plus production-fusion path using the archived parent inputs. Complete raw-data regeneration of all parent models remains a separate running verification job.

The selected checkpoint's exact training command and selected ep2 weights are documented in REPRODUCE_TASK1_CX38.md. The parent training recipes, environment and attribution remain unchanged. The source and settings were backed up before website evaluation. The portable hash-checked entrypoint was then replay-tested against that exact evaluated artifact. Actual public feedback is separate from local validation and smoke tests.

November 24 final competition upload is reserved for the user's manual action. This file is the strongest verified artifact as of the five November 19 evaluations. Earlier accepted versions remain available.
