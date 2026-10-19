# MCC5 encoder × physics factorial: frozen supplementary experiment

Registered 2026-10-07 before executing this new target-scoring event. The 27
MCC5 query runs have already been used in earlier, separately locked events.
Accordingly, this is a **post-target explanatory experiment**, not an
independent confirmation or a replacement for the original MCC5 result.

## Question and fixed design

Does the same 63-dimensional, parameter-free BPFO/BPFI physical descriptor
improve two frozen carrier encoders equally, or does its effect depend on the
carrier representation? The complete 2 × 2 grid is:

| Encoder | Carrier only | Carrier + physical |
|---|---|---|
| U2-Hz | first unique 128-D carrier token | same 128-D token + same 63-D descriptor |
| Width-matched WDCNN | 128-D embedding | same 128-D embedding + same 63-D descriptor |

All cells use the same eight distributed 3,072-point blocks, equal block
aggregation, five frozen seeds 20261101–20261105, 42 source training runs,
42 source-validation runs, 15 target-support runs and 27 fixed query runs.
The source/support role assignment, run identities, labels, physical descriptor,
StandardScaler, inverse-class-frequency-balanced source weights, target-support
weight 10 and multinomial LogisticRegression(C=0.1, solver=lbfgs,
max_iter=4000) are identical across cells. No encoder is retrained. The
development-only phase fits the four heads for each seed and freezes their
hashes before this event reads query features. No query statistic is used to
fit a scaler, head or threshold.

The original full RAMEF uses a 1,024-D representation containing eight
identical copies of a 128-D carrier token, plus 63 physical coordinates.
This duplication changes effective regularisation. The compact 128+63 grid is
therefore a *controlled factorial* and not a numeric replay of full RAMEF.
The earlier full-fusion 96.30% result is reported separately as a historical
anchor, never inserted as a fifth factorial cell. Fixed C=0.1 may penalise
the compact U2 view differently; no C is retuned after seeing query outcomes.

## Read order and contrasts

The preparation phase may read frozen development embeddings, checkpoints,
source labels and target-support labels. Source-validation and support-LOO
numbers are diagnostic only; they cannot select a cell or revise its head.
An exclusive target-evaluation lock is created before query features are read.
The locked execution scores every seed × cell exactly once and retains all
20 result rows and all 540 seed–cell–run predictions.

Primary descriptive contrasts, in percentage points, are: physical effect
within U2-Hz; physical effect within WDCNN; difference of those two effects
(interaction); and fused U2-Hz versus fused WDCNN. Report Accuracy,
Macro-F1, class recalls, paired corrections/new errors, and five-seed mean
with sample SD. A 10,000-replicate seed-and-class-stratified-run bootstrap
using fixed RNG 20261007 may describe uncertainty *inside the same fixed
cohort*. It must not be interpreted as independent-cohort inference.

The result can support only a representation-dependent, fixed-cohort effect.
It cannot establish that U2-Hz has universally superior architecture, that
the physical descriptor is optimal, or that the gain generalises to unseen
bearing individuals or new MCC5 runs. The frozen primary event and previous
post-target fairness event must remain byte-for-byte unchanged.
