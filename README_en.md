# RAMEF

**Rotation-aware measurement evidence fusion for cross-speed five-shot bearing fault diagnosis**

[中文说明](README_zh.md) · [Code and evidence map](docs/MANUSCRIPT_MAP.md) · [Reproducibility guide](docs/REPRODUCIBILITY.md) · [Citation](CITATION.md)

RAMEF combines distributed observation, learned resonance-carrier features,
geometry-conditioned physical evidence and target-support calibration for
cross-speed bearing fault diagnosis. It makes **one decision per complete
physical measurement record**, using five labelled target records per class.

This repository contains model and experiment source code, fixed split manifests,
archived class predictions, metric-replay tools and manuscript artwork.

## Method

The workflow has three stages:

1. **Observe.** Extract eight distributed, non-overlapping blocks from one
   record. These blocks increase within-record coverage; they are not additional
   labelled examples or shots.
2. **Form and fuse evidence.** Apply the frozen U2-Hz encoder to the standardised
   Hz-domain view and average the block embeddings. In parallel, compute
   geometry-conditioned soft BPFO/BPFI evidence and filtered-waveform statistics,
   then aggregate them into a 63-dimensional physical descriptor. Concatenate
   the carrier and physical representations at the measurement level.
3. **Calibrate and diagnose.** Fit a standardiser and a regularised multinomial
   logistic-regression head using source features and five labelled target
   supports per class. Freeze this calibration package before query inference.

U2-Hz is trained on source data, selected using source validation and then frozen.
PU source training uses centre crops; MCC5 source training uses distributed
blocks. The physical path has no trainable parameters. Query records contribute
no fitting statistics, parameter updates or model selection during the reported
calibration/inference procedure.

![RAMEF observation, dual-path evidence fusion and support-calibrated diagnosis](figures/current/Figure_01.svg)

**Naming:** RAMEF is the current method name. `RAMEF-F` in historical scripts,
prediction fields or logs refers to the same full 63-dimensional implementation;
those records retain their original names and hashes.

## Quick start: verify and replay

Run the following commands from the repository root with **Python 3.10 or later**.
These steps use only the Python standard library: no GPU, raw recordings or
model weights are required.

```bash
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python scripts/replay_paper_results.py --output-dir runs/replay01
```

Use an output directory that does not already exist. Prediction replay recomputes
scores from the archived class predictions; it does not train a network or
perform new model inference.

| Output | Contents |
|---|---|
| `pu_metrics.csv` | PU metrics and confusion matrices, with one vote per distinct measurement |
| `mcc5_seed_metrics.csv` | MCC5 run-level metrics for each method and seed |
| `seed_summary.csv` | Seed means and sample standard deviations |
| `summary.json` | Reference-value checks, input hashes and scoring definitions |

The local distribution checks cover **41 unit tests**, complete file-manifest
verification, prediction replay and a Windows Git line-ending round trip. See
[the recorded checks](docs/LOCAL_PACKAGE_CHECKS.json) for their scope and runtime.

## Tasks and reference results

### PU: cross-speed real-damage development task

The three source conditions D0, DM and DF operate at 1500 rpm; the target condition
DS operates at 900 rpm. The task classifies normal (N), inner-race fault (IR) and
outer-race fault (OR) states. Source training and validation contain 816 and 204
measurements. After encoder selection, all 1020 source measurements and 15 target
supports form the final calibration cohort.

Each of the **171 distinct query measurements** is counted once: 55 N, 62 IR and
54 OR. Support and query measurement identities are disjoint.

| Locked method | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Centre-crop U2-Hz | 88.89 | 88.16 |
| U2-Hz-DM8 | 97.66 | 97.55 |
| RAMEF | 98.83 | 98.78 |

A separate five-seed RAMEF rerun, with seeds fixed before that rerun, gives
**94.04% ± 1.91% Accuracy** and **93.76% ± 2.00% Macro-F1**. It is reported
separately from the locked single-model result.

The historical 240-entry ledger contains repeated measurement identities; replay
checks duplicate-prediction consistency before collapsing them. PU query records
were accessed during earlier project development, so these results describe a
**locked, historically accessed development cohort**, not untouched external
confirmation or unseen-bearing-specimen generalisation.

### MCC5: run-level race localisation under motor-fault interference

The locally prespecified MCC5 task uses 42 source-training runs tagged 3000 rpm,
42 source-validation runs tagged 2000 rpm, and a 1000-rpm-tagged target set of
15 supports plus **27 query runs**. The query contains 13 no-race-fault (NRF),
7 IR and 7 OR runs. Final calibration uses the 42 source-training runs and
15 supports, excluding the source-validation runs.

NRF means no inner- or outer-race fault, not necessarily a globally healthy motor.
IR and OR include pure race faults and compound motor/race faults. Ball faults
and simultaneous inner-/outer-race faults are excluded.

| Locally prespecified comparison | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| U2-Hz-LM8: local eight-block carrier | 67.41 ± 4.06 | 60.25 ± 7.18 |
| U2-Hz-DM8: distributed eight-block carrier | 79.26 ± 6.20 | 75.17 ± 7.11 |
| RAMEF | 96.30 ± 2.62 | 96.23 ± 2.81 |

Values are **mean ± sample SD across five encoder seeds on the same 27 runs**,
not five independent cohorts or population confidence intervals. MCC5 supplies
a locally prespecified **run-level mechanistic stress test**, not third-party
preregistration or specimen-level confirmation. Query-free support audits concern
the registered 15-support neighbourhood. Subsequent capacity, dimension and
same-physics comparisons remain frozen post-target analyses and do not replace
the original 96.30% result. CWRU and SQ are supplementary ceiling controls.

## Running MCC5 from raw data

Obtain MCC5-THU Motor recordings separately from the dataset provider under its
terms. Install the modelling dependencies and a suitable PyTorch build:

```bash
python -m pip install -r requirements-model.txt
```

`requirements-model.txt` does **not** install PyTorch. The recorded experiment
used Python 3.13.13, PyTorch 2.12.0+cu126, NumPy 2.4.4, SciPy 1.17.1 and
scikit-learn 1.8.0; complete version metadata are in
[`configs/recorded_runtime.json`](configs/recorded_runtime.json). These are
recorded versions, not a guarantee of wheel availability or bitwise-identical
training on other hardware. The standard-library replay workflow above is separate.

Replace `PATH_TO_MCC5_DATA` with the parent directory containing
`MCC5-THU Motor_speed_circulation`. Use a fresh work directory:

```bash
# Inspect the preparation command without reading waveforms.
python scripts/mcc5_pipeline.py prepare --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh --dry-run

# Prepare source, source-validation and support data.
python scripts/mcc5_pipeline.py prepare --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh

# Train on source data and select checkpoints on source validation.
python scripts/mcc5_pipeline.py train --work-dir runs/mcc5_fresh

# Run the separate locked evaluation stage after selection.
python scripts/mcc5_pipeline.py evaluate --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh
```

Keep original evidence directories unchanged. Do not delete evaluation locks to
repeat an evaluation in the same workspace. Fresh training is a computational
reproduction on the existing task, not an independent target cohort. The local
package tests validate launchers and stored-prediction replay, not a complete
fresh-machine raw-data training run.

## Repository guide

| Location | Contents |
|---|---|
| `code/` | Model, physical-descriptor and historical experiment implementations |
| `scripts/` | Integrity verification, prediction replay and portable MCC5 launcher |
| `configs/` | Active parameters and recorded software environment |
| `protocols/` | Dataset splits, role manifests and historical protocols |
| `evidence/` | Class predictions used for metric replay |
| `reference/` | Archived statistics and numerical reference inputs |
| `figures/current/` | Exact final manuscript SVG artwork and its manifest |
| `figures/source/fig1/` | Current Figure 1 source chain and provenance metadata |
| `docs/` | Script index, manuscript mapping, model-asset inventory and provenance |
| `tests/` | Standard-library launcher, metric and distribution tests |

Useful entry points:

- [Script index](docs/SCRIPT_INDEX.md): executable entry points and prerequisites.
- [Manuscript mapping](docs/MANUSCRIPT_MAP.md): sections, tables and code/evidence.
- [Figure guide](figures/README.md): current artwork, source scripts and external inputs.
- [Version binding](docs/MANUSCRIPT_VERSION.json): exact main/supplement identities.
- [Model-asset inventory](docs/EXTERNAL_MODEL_ASSETS.csv): original filenames and hashes.

Historical PU/CWRU/SQ runners and some figure scripts require stage-specific
caches or separately obtained assets. Model checkpoints, fitted classifiers,
raw recordings, feature caches and manuscript PDFs are **not included**. Figure 1's
source-waveform CSVs are external prerequisites. Figure 7 reuses frozen
coordinates rather than refitting t-SNE.

`.gitattributes` preserves exact file bytes across platforms. If an old Windows
checkout has already converted line endings, obtain a fresh checkout rather than
rewriting historical hashes. Current distribution changes and instructions are
recorded in [CHANGELOG.md](CHANGELOG.md) and
[UPLOAD_INSTRUCTIONS.md](UPLOAD_INSTRUCTIONS.md).

## Citation and rights

The associated manuscript is:

> Shuanglin Zhang, Hao Zhang, Liang Zhang and Yuanhang Sun. *Rotation-aware
> measurement evidence fusion for cross-speed five-shot bearing fault diagnosis*.
> Unpublished manuscript, 2026.

See [CITATION.md](CITATION.md) for version-aware citation information. No
publication or archival DOI is assigned in this repository package.

Software licensing and third-party rights are documented in
[LICENSE_STATUS.md](LICENSE_STATUS.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A public repository does not
itself grant an open-source licence; this package does not assign one or relicense
the original datasets.
