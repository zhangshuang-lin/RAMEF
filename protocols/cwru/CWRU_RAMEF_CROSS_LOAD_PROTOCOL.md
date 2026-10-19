# CWRU_RAMEF_CROSS_LOAD protocol

Status: frozen before bag creation and evaluation.

## Task

- Native CWRU ten-class classification: N, IR007/014/021, B007/014/021 and
  OR007/014/021.
- Cross-load: each of L0-L3 is target in turn and the other three loads are
  source.
- Five target support bags per class and three target query bags per class.
- Ten deterministic support/query episodes per target load and five already
  trained native-A1 backbone seeds: 200 evaluations.

## Observation and isolation

One bag contains eight distinct 1,536-sample raw windows.  Within an episode,
no target raw window appears in both support and query, and no two bags share a
window.  CWRU supplies only one record for most class-load combinations, so
support and query bags necessarily share a parent record.  This is a segment-
bag cross-load protocol and is explicitly not claimed to have PU-style
measurement isolation.

Each source class-load record contributes five non-overlapping bags.  Each
target class record contributes five support and three query bags.

## Geometry-conditioned physics

Drive-end bearing orders are fixed as BPFO=3.5848, BPFI=5.4152 and BSF=2.3570.
Raw sample rate is 12 kHz.  Registered resonance bands are 500-1800,
1800-3500 and 3500-5500 Hz.  For each order and band, harmonic-ratio evidence
is aggregated across the eight blocks by mean and median only.  Kurtosis and
log-RMS mean/median are retained.  Descriptor dimension is 36.

## Fixed comparisons

1. `A1_MEAN8`.
2. `RAMEF`: A1 mean8 + 36-D geometry-conditioned static physics.
3. `A1_MEAN8_RANDOM36`: deterministic equal-dimensional Gaussian control.

Every head uses StandardScaler and LogisticRegression(C=0.1), source/support
weights 1/10.  Query data cannot select a method or parameter.

## Confirmation gates

RAMEF passes cross-dataset confirmation if mean ten-class accuracy and
Macro-F1 are no lower than A1 mean8 by more than 0.1 pp, mean accuracy is at
least 99%, every target-load mean is at least 98%, and random36 does not exceed
RAMEF.  Ceiling ties count as safe transfer but not as proof of a positive CWRU
accuracy gain.
