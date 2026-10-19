# Fixed statistical replay protocol

This document records the deterministic replay of an **existing post-hoc analysis**, not a prospective experiment. Its input predictions, scoring correction, random seeds and reported intervals already existed before this local release package was assembled. No new target query is loaded for model development.

## Inputs and decision unit

- Source: archived `outputs/staged/RACP_FAIR_ABLATION/predictions.csv`, SHA-256 `f51d74badfbccaede3b4442c75186a0e7cd560d73104be60db45005786110007`.
- Independent consistency ledgers: `SUPPORT_CALIBRATION_DIRECT_ABLATION/predictions.csv`, SHA-256 `f8a21ca0f09c8147dc86645e59ed9286d1e453ce35559ac34da3f1e15179bf09`, and `RACP_MEASUREMENT_FINAL/predictions.csv`, SHA-256 `d0216382dc28a72fd10b30e6f497be93aa79b285a5baad7a61901425903c5e64`.
- Frozen methods: centre-crop U2-Hz (`B0_A1_SINGLE`) and locked full-descriptor RAMEF-F (`B5_A1_MEAN8_RACP`).
- Input consists of 240 registered query entries from 171 distinct physical measurement identities. Every repeated identity must have the same true label, bearing code and method predictions across entries.
- Collapse rule: sort distinct measurement IDs lexicographically and retain one identical prediction tuple per ID. The 171 scores comprise 55 N, 62 IR and 54 OR measurements. No window or block is treated as an extra vote or shot.

## Paired calculations

Confusion matrices use true class as row and prediction as column. Accuracy, class recall and macro-F1 are computed separately for centre crop and RAMEF-F, with zero F1 if a class denominator is zero. A correction occurs when centre crop is wrong and RAMEF-F is correct; an introduced error is the reverse. The exact two-sided paired-binomial probability is computed from the 17 discordant corrections and zero introductions.

1. **Measurement bootstrap:** 10,000 replicates, NumPy `default_rng(20260924)`. In each replicate, sample measurement indices with replacement separately within N, IR and OR, preserving class denominators 55/62/54. Retain paired predictions for every sampled identity. Record RAMEF-F minus centre-crop Accuracy, Macro-F1 and OR-recall gains in percentage points.
2. **Bearing-code-cluster bootstrap:** 10,000 replicates, Python `random.Random(20260925)`. Sample bearing codes with replacement within each macro class, retaining every measurement belonging to each sampled code. Cluster sample size may vary. Record paired Accuracy, Macro-F1 and class-recall gains.
3. **Intervals:** compute the arithmetic mean and the 2.5th and 97.5th percentiles (`numpy.quantile`, default linear interpolation) over each fixed sequence of 10,000 gains.
4. **Code sensitivity:** enumerate all sign assignments for non-zero bearing-code net-correction counts, once across classes and once for OR codes only. Also recompute paired gains after omitting each bearing code in turn.

The original implementation and archived summary are `paper/audit/pu_unique_measurement_reanalysis.py` and `outputs/staged/PU_UNIQUE_MEASUREMENT_REANALYSIS/results.json`. This package's `replay.py` is self-contained and verifies the reproduced values against `historical_expected.json`.

## Interpretation boundary

All analyses are descriptive of a fixed, historically accessed PU development cohort. The intervals do not estimate uncertainty for a newly collected bearing population or unseen bearing individuals. No prediction, parameter, support set, model or threshold is changed by this replay.
