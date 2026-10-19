# PU paired-bootstrap replay package

This package reproduces the measurement-level paired statistics reported for the **historically accessed PU development cohort**. It starts from frozen class predictions. It does not read vibration waveforms, train a model, choose a checkpoint, change a decision threshold, or provide an independent target confirmation.

## Contents

| File | Purpose |
|---|---|
| `frozen_240_entry_predictions.csv` | Minimal frozen query-prediction ledger: 240 registered entries, measurement identity, bearing code, true class, centre-crop prediction and locked RAMEF-F prediction. Class indices are 0=N, 1=IR, 2=OR. |
| `unique_171_measurement_predictions.csv` | Deterministic one-vote-per-measurement collapse, sorted by measurement identity, with paired error-transition labels. |
| `measurement_bootstrap_replicates.csv` | All 10,000 class-stratified measurement-resampling gains in percentage points. |
| `code_cluster_bootstrap_replicates.csv` | All 10,000 class-stratified bearing-code-cluster resampling gains in percentage points. |
| `code_contributions.csv` | Corrected and introduced errors for each bearing code. |
| `leave_one_code_out.csv` | Accuracy and OR-recall gains after omitting one bearing code. |
| `summary.json` | Observed scores, paired error counts, exact paired-binomial result, bootstrap means and percentile intervals. |
| `summary_table.csv` | Eight measurement- and code-cluster-bootstrap metric rows suitable for checking the reported intervals. |
| `historical_expected.json` | Pre-existing archived statistics used as strict replay assertions. |
| `source_manifest.json` | SHA-256 provenance of the frozen source ledgers and packaged input. |
| `MANIFEST.sha256` | SHA-256 checksums for package files other than the manifest itself. |
| `replay.py` | Self-contained statistical replay script. |

## Replay

Use Python with NumPy installed. This package was replayed and checked with Python 3.12.14 and NumPy 2.3.5 in the current workspace runtime. From this directory:

```text
python replay.py
```

The script validates the 240-entry ledger, collapses repeated measurement identities, performs the fixed resampling sequences, checks the historical results to a strict numerical tolerance, and rewrites the derived CSV/JSON outputs and checksum manifest. It does not require the original PU raw signal files.

To verify the packaged files before replay, compare each entry in `MANIFEST.sha256` with the corresponding file. On Windows PowerShell, `Get-FileHash -Algorithm SHA256` can be used for individual files.

## Scope and access

The prediction ledger is derived from existing target-query evaluations. The query had been accessed earlier in the broader project. The resulting intervals quantify sensitivity **inside this fixed development cohort**; 171 measurement identities do not represent 171 independent bearing damage morphologies. The OR-only bearing-code sign-flip analysis has limited non-zero clusters. Do not use these outputs for model selection or describe them as an untouched test.

The package contains predictions and statistics, not third-party vibration waveforms. A public repository identifier, licence and DOI have not yet been assigned; this local package is not itself a public deposit.
