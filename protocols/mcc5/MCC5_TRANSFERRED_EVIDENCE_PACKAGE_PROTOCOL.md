# MCC5_TRANSFERRED_EVIDENCE_PACKAGE protocol

## Status and evidence boundary

This package is frozen after the original preregistered MCC5 RAMEF target
evaluation.  It cannot alter the original PASS decision or modify RAMEF.

- T1 is query-free and uses source, source-validation and the 15 target-support
  runs only.
- T2 is a post-hoc frozen fair-comparator benchmark.  Baseline names, seeds,
  input budget, training epochs, checkpoint selection and adaptation rules are
  fixed here before baseline query scoring.
- T3–T5 use frozen predictions/features for descriptive or post-hoc statistical
  analysis.  All outputs are reported regardless of direction.

## Frozen data roles

- Source training: 42 runs, 3000-rpm tag.
- Source validation: 42 runs, 2000-rpm tag.
- Target support: 15 runs at 1000-rpm tag, five per NRF/IR/OR.
- Target query: 27 runs at 1000-rpm tag, NRF/IR/OR = 13/7/7.
- Observation for formal methods: eight distributed 3072-point blocks at
  10,20,30,40,50,60,70,80 s (24,576 points/run).

## Seeds

All methods use `20261101–20261105`.  Values summarize optimization
initialization on one fixed target cohort, not five independent target cohorts.

## T1 — query-free support stability

Views: U2-Hz-DM8 and RAMEF.

- Source-only support prediction.
- Source + support leave-one-out with support weights 1,5,10,20.
- Support-only logistic regression.
- Support-only ordinary prototype.
- Exhaustive leave-two-support-out at registered support weight 10 (105 folds,
  210 held predictions per seed).
- `C=0.1`, LBFGS, StandardScaler fit on each fold's training data.
- No target-query cache may be opened.

## T2 — post-hoc frozen fair method benchmark

Baselines:

1. WDCNN-DM8;
2. ResNet1D-DM8;
3. STFT-CNN-DM8;
4. WDCNN-DANN-DM8;
5. ProtoNet-DM8;
6. U2-Hz-DM8 (existing frozen result);
7. RAMEF (existing frozen result).

Conventional baselines:

- 128-D measurement embedding: encode each distributed block, then arithmetic
  mean over eight blocks.
- Source block training for 12 epochs, AdamW, learning rate 1e-3, weight decay
  1e-4, batch size 12, gradient clip 5.
- Class-balanced source sampler.
- Checkpoint selected by 2000-rpm source-validation measurement Macro-F1.
- Final standardized multinomial logistic head uses source class-balanced
  weights and target-support weight 10, `C=0.1`.
- DANN uses source torque (20 vs 40 Nm) as the nuisance-domain label, GRL
  coefficient linearly increasing from 0 to 0.2, and domain-loss weight 0.1.

ProtoNet-DM8:

- WDCNN encoder with 128-D embedding.
- 3-way 5-shot 3-query measurement-level episodes from 3000-rpm source runs.
- Eight distributed blocks are averaged before prototype computation.
- 12 epochs × 20 deterministic episodes/epoch.
- Euclidean distance to class prototypes; AdamW 1e-3/1e-4.
- Checkpoint selected by a deterministic 2000-rpm 5-shot validation task.
- Target inference uses the fixed 15 target supports as prototypes; no linear
  calibration head.

No architecture, epoch count, learning rate, checkpoint or target-support rule
may be changed after target scoring.

## T3 — run-level error audit

- Existing U2-Hz-LM8, U2-Hz-DM8 and RAMEF predictions.
- 27 query runs × five seeds.
- Output correctness matrix, DM8 fixes over LM8, RAMEF fixes over DM8 and new
  errors, stratified by class/context/severity/torque.

## T4 — hierarchical paired bootstrap

- 10,000 deterministic replicates, seed 20261220.
- Resample five encoder seeds with replacement.
- Within each NRF/IR/OR class, resample query runs with replacement.
- Recompute paired Accuracy and Macro-F1 differences for DM8−LM8 and
  RAMEF−DM8.
- Intervals are post-hoc uncertainty descriptions for the fixed MCC5 cohort.

## T5 — context, severity and physical evidence

- Frozen predictions and physical descriptors only.
- Report class, pure/compound context, severity and torque groups where at
  least two query runs are available.
- Analyze BPFO, BPFI and BPFO-minus-BPFI summary evidence.
- Small groups are descriptive; no universal or specimen-level claim.

## Figure contract

1. Fair-method quantitative grid: cross-method performance, seed variability
   and class recalls.
2. Run-level correction heatmap: show whether gains are widespread or driven
   by a few runs.
3. Context/physics composite: connect error correction to context, severity and
   soft-order evidence.

All figures use Python/matplotlib only and export SVG/PDF/TIFF/PNG.
