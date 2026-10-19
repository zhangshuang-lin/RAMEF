# SQ_RAMEF_STRICT_V1 protocol

Status: preregistered before feature extraction, model training or query scoring.

## 1. Scientific objective

SQ is used to test whether RAMEF can provide a second record-independent positive
result by recovering fault evidence under continuous speed variation. The
positive conclusion is not assumed in advance. CWRU remains a separate
ceiling-level geometry-portability/non-inferiority experiment.

The primary SQ hypothesis is:

> Under equal sample and label budgets, blocks distributed across independent
> rotational states provide more stable low-speed/transient evidence than the
> same number of locally adjacent blocks.

## 2. Dataset contract

- Root: `D:/SQ-variable-speed-bearing`.
- Sampling rate: 25.6 kHz.
- Measurement unit: one `RECxxxx` acquisition.
- Vibration channel: `RECxxxx_ch2.txt`.
- Keyphasor/pulse channel: `RECxxxx_ch3.txt`.
- Each channel pair must share the same REC identifier, block size and time axis.
- One record contains a continuous 0 -> 3000 rpm -> steady -> 0 process.
- Classes:
  - `NC` -> N;
  - `IF_1`, `IF_2`, `IF_3` -> IR;
  - `OF_1`, `OF_2`, `OF_3` -> OR.
- Damage level is retained as a subtype covariate for stratified analysis and is
  never supplied to the three-class classifier.

Audited independent-record counts:

| State | REC count |
|---|---:|
| NC | 9 |
| IF_1 / IF_2 / IF_3 | 10 / 9 / 6 |
| OF_1 / OF_2 / OF_3 | 6 / 6 / 6 |
| Macro N / IR / OR | 9 / 25 / 18 |

## 3. Signal and speed audit (P0)

Before any model training:

1. Parse the 17-line exported-text header and verify sample rate/block size.
2. Detect falling pulse edges from ch3 and compute instantaneous rpm and
   cumulative shaft angle.
3. Reject or flag records with unmatched channels, non-monotone timestamps,
   insufficient pulses, implausible rpm, or fewer than eight eligible blocks.
4. Freeze per-record valid speed intervals and their SHA-256 ledger.
5. Do not read class labels while selecting pulse thresholds beyond the fixed
   signal-quality check.

Speed regions are preregistered as:

- low: 300–1000 rpm;
- middle: 1000–2000 rpm;
- high transient: 2000–2800 rpm;
- plateau/source: >= 2400 rpm with absolute speed slope below the source-only
  threshold frozen during P0.

If fewer than 90% of records support a region, P0 may merge adjacent regions
using only pulse-coverage statistics. No classification metric may guide this
decision.

## 4. Record-independent outer folds

Use nine deterministic outer folds, indexed by the nine NC records.

Within every fold and macro class:

- query: at least one REC, with the NC query fixed by the fold index;
- target support: exactly five distinct REC identifiers;
- source training: every remaining eligible REC;
- source, support and query REC sets are mutually disjoint;
- all speed regions from one REC inherit the same role.

IR and OR query/support selections rotate across damage levels so that each
severity contributes to query evaluation. The selection schedule is generated
from REC identifiers and seed before loading waveforms. Aggregate statistics
use REC as the resampling cluster.

The five-shot count refers to five physical REC acquisitions per macro class,
not five windows or five speed-bin bags.

## 5. Cross-speed training and target task

- Source encoder training uses only plateau/source regions of source-role REC.
- Target support calibration uses transient regions of support-role REC.
- Target query evaluation uses the same registered transient regions from
  query-role REC.
- No query feature, speed statistic, prediction or label is used for epoch,
  seed, speed-bin, block-position or hyperparameter selection.
- A query firewall records every materialised REC identifier.

## 6. Equal-budget observation interventions

All compared representations use the same raw sample count per decision.

1. `Single`: one registered local block.
2. `Local-M8`: eight adjacent non-overlapping blocks from one eligible local
   interval.
3. `Distributed-M8`: eight non-overlapping blocks distributed across all
   eligible acceleration/deceleration intervals in the speed region.
4. `Distributed-M8-physical`: physical descriptor only.
5. `RAMEF-F`: Distributed-M8 U2-Hz carrier plus geometry-conditioned physical
   descriptor.
6. `Random-control`: Distributed-M8 carrier plus equal-dimensional deterministic
   random features.

Raw block duration is initially 0.256 s (6554 samples at 25.6 kHz, rounded by
the frozen implementation) and is resampled to 16,384 points at 64 kHz for the
U2-Hz carrier encoder. Native-rate samples are retained for physical evidence.

If P0 shows that eight 0.256-s blocks cannot fit a preregistered speed region,
all methods must use the same shorter source-only-selected block duration; the
number of blocks, total samples and method comparison remain equal.

## 7. Fixed-time versus fixed-rotation mechanism test

The ch3 keyphasor enables a direct mechanism test:

- fixed-time blocks: registered duration in seconds;
- fixed-rotation blocks: four shaft rotations per block, angle-resampled to the
  common input length.

The expected falsifiable pattern is that the low-speed deficit and the
Distributed-M8 advantage should shrink after fixing rotations per block. The
result is reported even if the pattern is absent.

## 8. Model and selection contract

- Train an SQ-native U2-Hz encoder; do not reuse the PU-trained checkpoint as
  the main SQ model.
- Encoder checkpoint and hyperparameters are selected only by source-role
  record validation.
- Final target calibration uses the registered linear head, source weight 1,
  support weight 10 and C=0.1 unless a source-only audit rejects numerical
  stability before any query inference.
- At least five encoder seeds are trained.
- All methods receive identical REC roles, block positions and support labels.

The NSK6203 soft BPFO/BPFI anchors are either taken from verified geometry or
estimated once from source-role plateau records within preregistered theoretical
ranges. Query spectra cannot tune the anchors.

## 9. Outcomes

Report per fold, seed, speed region and damage subtype:

- Accuracy and Macro-F1;
- N, IR and OR recall;
- confusion matrix;
- worst subtype accuracy;
- Local-M8 vs Distributed-M8 paired changes;
- U2-Hz-M8 vs RAMEF-F paired changes;
- support perturbation stability;
- observation and computation latency.

## 10. Positive-conclusion gates

A second strict positive RAMEF conclusion requires all of the following:

1. Distributed-M8 exceeds Local-M8 by at least 2 percentage points in mean
   Macro-F1 under equal sample count.
2. The improvement is positive in at least 7/9 outer folds and at least 4/5
   encoder seeds.
3. Low-speed Macro-F1 gain is at least as large as high-speed gain.
4. Worst IR/OR damage-subtype accuracy does not decrease.
5. Random features do not reproduce the improvement.
6. REC-cluster paired confidence interval for the primary comparison excludes
   zero, or the exact paired test passes the preregistered alpha.
7. No source/support/query REC leakage is detected.

The physical-fusion claim remains bounded. RAMEF-F is not required to exceed
carrier-only by 2 points if the carrier observation effect passes; any physical
increment must instead show non-inferiority and improved support stability.

If these gates fail, SQ is reported as a falsification or boundary result and
is not described as a second positive validation.

## 11. CWRU role

CWRU retains its existing ten-class cross-load protocol solely to verify that
geometry parameterisation does not degrade an already saturated native
representation. It is not used to estimate RAMEF gain, rotation-budget benefit
or record-independent generalisation.

