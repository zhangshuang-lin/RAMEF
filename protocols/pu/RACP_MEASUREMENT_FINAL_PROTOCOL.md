# RACP_MEASUREMENT_FINAL protocol

Status: performance-priority candidate selected and frozen without DS query.

## Task

- PU D0/DM/DF at 1500 rpm to DS at 900 rpm.
- Three classes: N, IR and OR.
- Five labelled DS support measurements per class.
- Support and query measurement IDs are strictly disjoint according to
  `split_independent_measurement.json`.

## Observation unit

The classification unit is one physical measurement, not one isolated crop.
Eight uniformly placed, mutually non-overlapping 16,384-sample blocks are read
from each four-second measurement.  This does not increase the number of
labelled target measurements.  It changes the observation budget and must be
reported separately from the centre-crop A1 result.

## Frozen methods

1. `A1_SINGLE`: exact centre-crop frozen A1 embedding.
2. `RACP_MEASUREMENT`: mean of eight frozen A1 embeddings concatenated with the
   fixed 63-D RACP descriptor.

For both, StandardScaler and LogisticRegression(C=0.1) are fitted once using
all 1020 source measurements (weight 1) and 15 DS supports (weight 10).
No safety gate, target query statistic, threshold, tuning or model selection is
permitted after candidate freeze.

## One-time target evaluation

The 240 fixed DS query measurements are evaluated once.  The output directory
contains `target_evaluation.lock`; an existing lock aborts reruns.

Performance-priority success requires all:

- Accuracy > 90.00%.
- Macro-F1 >= 89.50%.
- N recall >= 95.00%.
- IR recall >= 83.00%.
- OR recall >= 82.00%.
- RACP accuracy exceeds the same-cohort centre-crop A1 accuracy.

The epistemic status is a newly frozen candidate evaluated after extensive
historical DS-query exposure elsewhere in the project.  It is not described as
a globally untouched test set, but this candidate was selected using support
only and is not changed after its evaluation.
