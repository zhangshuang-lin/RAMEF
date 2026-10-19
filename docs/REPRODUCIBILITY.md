# Reproduction levels

1. `python scripts/verify_bundle.py` checks SHA-256 and static import completeness without importing historical scripts.
2. `python scripts/replay_paper_results.py --output-dir runs/replay01` recomputes class metrics and seed summaries from class predictions, with no waveforms or model files.
3. Optional paired PU bootstrap: copy `reference/pu_paired_bootstrap/` into a new sandbox and execute its `replay.py` there (requires NumPy). Never replay in the frozen reference directory.
4. Fresh MCC5 computation: obtain the official data, install the recorded modelling dependencies plus an appropriate PyTorch build, and use `scripts/mcc5_pipeline.py` in prepare/train/evaluate order with a fresh work directory.
5. Raw PU/CWRU/SQ historical runs and post-target controls require their documented stage caches and additional model assets. The source closure is included, but cross-machine end-to-end execution has not been validated.

No target model was re-evaluated while assembling this package. Historical result JSON/locks are reference evidence, not permissions to rerun an original target event. A fresh computational reproduction uses a new workspace and does not create an independent cohort.

## Model configuration

The original M2 model-initialisation fields are retained in `configs/pu_model_initialization.yaml`. Historical `make_u2` explicitly overrides adapters, physics gates, late fusion and front-end choice; active RAMEF settings are listed separately in `configs/active_parameters.json`. Do not run an old M2 training configuration as if it were the paper recipe.

## Paths and identities

The PU split and historical protocols retain original paths as immutable evidence. Re-locate physical files using their record identifiers when adapting raw-data execution; do not change measurement/run identities, label roles or split membership. The portable MCC5 loader normalises only the disk address, not the string used by Random63 hashing.
