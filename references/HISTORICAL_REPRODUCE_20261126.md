# Historical training and replay records

These records are retained for provenance, not as current saved-weight inference instructions. Use [REPRODUCE.md](../REPRODUCE.md), version inference-20261127, for public model downloads, parent-model restoration, dataset placement, commands and verification limits.

## Training history

Training scripts remain under code/scripts and code/src. They describe how the submitted models were developed. Retraining is not required to restore the fitted public models. Historical training from random seeds is not certified to reproduce checkpoint bytes.

## Saved-intermediate replay

The earlier clean-directory replay regenerated the accepted CSV from frozen coordinate scores, extracted policy features and fitted policy models, with original-family rows inherited from a parent prediction. It was not a clean original-data inference test. The current guide does not use that replay as evidence of end-to-end inference.

## Earlier experiment-specific notes

The other REPRODUCE_TASK1 documents in this directory are dated experiment records. Their old access arrangements and stage-specific commands are superseded by the current versioned guide. They must not be used as the organizer's current download procedure.
