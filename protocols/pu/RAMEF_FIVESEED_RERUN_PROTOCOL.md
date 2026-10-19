# RAMEF independent five-seed rerun protocol

Freeze date: 2026-09-21

## Purpose

Run a new five-seed PU optimisation-sensitivity audit with a seed set fixed before this rerun opens target-query data. The experiment estimates reproducibility across the five declared initialisations; it does not select a seed or subset by target performance.

## Evidence boundary

- The broader project has historically accessed the fixed PU query, so this rerun remains a development-cohort reproducibility audit rather than an untouched external test.
- Stage A uses source data only and must finish with checkpoint hashes written to `source_freeze.lock` before Stage B can open target-query data.
- Stage B evaluates every frozen seed. No seed may be removed, replaced, ranked, or weighted using target metrics.
- The reported five-seed mean and sample standard deviation are calculated over all five declared seeds.

## Fixed seeds

`20260911, 20260912, 20260913, 20260914, 20260915`

## Fixed training protocol

- Source domains: PU D0, DM, and DF at 1500 rpm.
- Source split: 816 training and 204 validation measurements, identical to the previous PU multi-seed audit.
- Encoder: U2-Hz with the same frozen model configuration.
- Training: 12 epochs, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 12, gradient-norm clipping at 5.
- Checkpoint selection: highest source-validation Macro-F1; ties retain the earliest epoch.
- Target representation: eight registered non-overlapping blocks per measurement.
- Classifier: `StandardScaler` plus multinomial logistic regression, `C=0.1`, source weight 1, target-support weight 10, `max_iter=4000`.
- Target support/query identities: unchanged locked independent-measurement split, 15 supports and 240 queries.
- RAMEF-F physical descriptor: unchanged registered 63-dimensional descriptor.

## Required artifacts

- `protocol_frozen.md`
- `seed_registry.json` and `seed_registry.lock`
- `training_history.csv`
- `source_metrics.csv`
- `source_freeze.json` and `source_freeze.lock`
- five frozen checkpoints
- `target_seed_metrics.csv`
- `target_predictions.csv`
- `result.json`
- `target_evaluation.lock`
- `artifact_sha256.csv`

