# RAMEF: code and prediction replay

This repository accompanies **Rotation-aware measurement evidence fusion for cross-speed five-shot bearing fault diagnosis**. RAMEF combines distributed non-overlapping observation, a U2-Hz carrier representation, a 63-coordinate soft-order descriptor, and a target-support-calibrated linear classifier.

This is a lightweight **code + prediction/reference** package prepared for author upload. It does not contain third-party vibration recordings, feature caches, model checkpoints, fitted classifier files, or manuscript/reference PDFs. It is not itself evidence of a completed public deposit.

## Start here

From the repository root, use Python 3.10 or later for the standard-library-only checks:

```text
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python scripts/replay_paper_results.py --output-dir runs/replay01
```

The replay command reads the packaged class predictions, recomputes metrics, and compares them with the supplied expected results. It does **not** load waveforms, train a model, select a checkpoint, or run new target inference. Results are written to a new working directory rather than replacing the archived evidence.

## Repository guide

| Directory | Contents |
|---|---|
| `scripts/` | Bundle verification, prediction replay, and the portable MCC5 pipeline entry point |
| `code/` | Historical experiment scripts and their local Python dependencies; file-location adaptations are documented |
| `evidence/pu/` | PU prediction ledgers and the reference results used for replay |
| `evidence/mcc5/` | MCC5 prediction ledgers and reference metadata |
| `figures/` | Selected figure source scripts and public numerical source data |
| `docs/` | Script index, manuscript mapping, execution requirements, and provenance |
| `tests/` | Standard-library-only bundle/replay tests |

See `docs/SCRIPT_INDEX.md` for the role and prerequisites of each script and `docs/MANUSCRIPT_MAP.md` for the manuscript-to-artifact mapping. A script being included does not mean that every historical workflow can run from this lightweight checkout alone.

## What the prediction replay checks

### PU development cohort

The primary unit is one vote per distinct measurement: **171 measurements**, not 240 independent records. The 240-entry ledger is retained as historical provenance where included. The locked centre-crop, U2-Hz-DM8, and RAMEF-F results are:

| Method | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Centre-crop U2-Hz | 88.89 | 88.16 |
| U2-Hz-DM8 | 97.66 | 97.55 |
| RAMEF-F | 98.83 | 98.78 |

The separate new-five-seed RAMEF-F summary is **94.04% ± 1.91% Accuracy** and **93.76% ± 2.00% Macro-F1**. The standard deviations describe encoder-initialisation variability on the same measurement cohort. The PU query had been accessed during project development; these are locked, historically accessed **development-cohort** results. Measurement isolation is not bearing-specimen isolation, and blocks do not count as shots.

### MCC5 run-level stress test

The original locally prespecified experiment used 42 source-training runs, 42 separate source-validation runs, 15 target-support runs, and **27 fixed target-query runs** (NRF/IR/OR = 13/7/7). Five seeds share those same 27 query runs; they are not five independent test cohorts. The original 63-coordinate RAMEF result is **96.30% ± 2.62% Accuracy** and **96.23% ± 2.81% Macro-F1**.

NRF means no inner- or outer-race fault; it need not mean a globally healthy motor. The task is run-level race localisation in the presence of structured motor-fault interference, not unseen-bearing-specimen confirmation. The local protocol freeze is not third-party preregistration. Later capacity/dimension and encoder-by-physics controls are post-target analyses; their results, including the compact-fusion 99.26% control, do not replace the original 96.30% result. Query-free support audits concern the registered 15-support neighbourhood.

## Training or fresh inference on MCC5

Obtain MCC5-THU Motor data separately from its official provider and follow its terms. Use a fresh work directory and supply the directory expected by the data loader explicitly:

```text
python scripts/mcc5_pipeline.py prepare --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh
python scripts/mcc5_pipeline.py train --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh
python scripts/mcc5_pipeline.py evaluate --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh
```

Preparation and training use source/source-validation/support roles. Evaluation is a separate, locked operation after checkpoint selection. Do not remove an evaluation lock to rerun into the same work directory. A new training run is a computational reproduction, not a new independent target cohort.

The modelling workflow additionally requires NumPy, SciPy, PyTorch, scikit-learn, joblib, PyYAML, and related dependencies specified by the packaged environment/dependency documentation. Plotting additionally requires matplotlib and pandas. The original deep-learning runtime used PyTorch 2.12.0+cu126, NumPy 2.4.4, SciPy 1.17.1, and scikit-learn 1.8.0. Matching these versions does not guarantee bitwise-identical GPU training. No fresh-machine, raw-data-to-results training validation is claimed by this script-organising release.

## Historical PU scripts and figures

The historical PU entry points are retained with their dependencies for inspection and reuse. Some require stage-specific caches and frozen artifacts that are not distributed. A new portable raw-PU end-to-end training entry point has **not** been implemented in this bundle.

Figure scripts are archived source, not a promise that all manuscript figures rebuild with one command. Some expect the original staged-result layout; the real waveform asset used in Fig. 1 is not redistributed. Public numerical inputs included here can be inspected without the raw datasets. The figure index records current manuscript bindings and historical aliases.

## Integrity and publication status

The bundled manifest records file SHA-256 values. Verification detects changed or missing bundle files; it does not certify the chronology of an external preregistration service or establish scientific validity by itself. Original measurement/run identities are retained; portable file-location changes do not redefine the split.

Author identity, repository URL, archival DOI, and software/data licensing require author confirmation before publication. See `LICENSE_STATUS.md`, `THIRD_PARTY_NOTICES.md`, and `UPLOAD_CHECKLIST.md`. No software licence or third-party redistribution permission is implied by this preparation package.
