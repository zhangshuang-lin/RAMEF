# MCC5_RAMEF_CONFIRMATORY preregistered protocol

Frozen: 2026-08-24, after metadata/integrity/keyphase P1 and before any neural
feature, target-query vibration or target prediction was produced.

## Scientific question

Does Rotation-Aware Measurement Evidence Fusion (RAMEF) provide a second,
run-disjoint positive result for cross-speed bearing-race localisation when
non-bearing motor faults and electromechanical compound faults act as structured
nuisances?

This is not a pure healthy/IR/OR replication. The pure task failed the capacity
gate (N/IR/OR=6/11/13 speed-circulation runs).

## Dataset and task

- Dataset: MCC5-THU Motor, Data in Brief 65 (2026) 112583;
  DOI 10.1016/j.dib.2026.112583; data DOI 10.17632/6s3dggj9mw.1.
- Input: column 4, motor drive-end horizontal radial vibration only.
- N: no inner/outer-race fault; includes globally healthy and non-bearing motor
  faults.
- IR: pure or compound inner-race fault.
- OR: pure or compound outer-race fault.
- Ball faults and dual inner+outer-race files are excluded.
- The claim is run-disjoint, not bearing-specimen-disjoint because the release
  does not provide specimen identities.

## Frozen domains and identities

- Source training: 3000-rpm-tagged speed-circulation runs, 42 total
  (N/IR/OR=18/11/13).
- Source validation: 2000-rpm-tagged runs, 42 total (18/12/12).
- Target: 1000-rpm-tagged runs.
- Target support: five runs per class, with three 20-Nm and two 40-Nm runs
  selected by the frozen SHA256 rule.
- Target query: every remaining target run, N/IR/OR=13/7/7 (27 total).
- Manifest: `outputs/staged/MCC5_RAMEF_CONFIRMATORY/protocol/protocol_run_manifest.csv`.
- Manifest SHA256:
  `1fbf6660faee7baa4b9aa8655136190247932928812c6c67c68a7a4712649941`.

No file occurs in more than one role. Windows from a file inherit the file's
role and are never treated as independent labelled measurements.

## Frozen observation protocol

- Sampling rate: 12,800 Hz.
- Block length: 3072 samples = 0.24 s = four nominal rotations at 1000 rpm.
- Single/K1: local block index 3, start 18.72 s.
- Local-M8 starts: 18.00, 18.24, 18.48, 18.72, 18.96, 19.20, 19.44, 19.68 s.
- Distributed-M8 starts: 10, 20, 30, 40, 50, 60, 70, 80 s.
- Both M8 methods receive exactly 24,576 raw samples per run.

P1 read only time and keyphase from target-query files. Target-query vibration
remains inaccessible until all source-selected checkpoints are frozen.

## Frozen carrier model

- MCC5-native lightweight U2-Hz; no PU/SQ/CWRU weight is reused.
- Complex STFT: FFT/window 1024, hop 256; only log Hz magnitude reaches the
  encoder.
- Output grid 128x64; block embedding 1024 dimensions.
- Five audit seeds: 20261101--20261105.
- AdamW, learning rate 1e-3, weight decay 1e-4, batch size 12, 12 epochs,
  gradient clipping 5.
- Source class-balanced block sampler.
- Checkpoint criterion: source-validation Macro-F1 only.

## Frozen physical view

- SKF 6205-2ZC3 orders: BPFO=3.585, BPFI=5.415. BSF=2.357 is not used because
  ball faults are excluded.
- Resonance bands: 300--1500, 1500--3000 and 3000--5500 Hz. These cover the
  usable 6.4-kHz Nyquist range while excluding the dominant near-DC rotational
  component and the upper guard band; they were fixed before model output.
- Four harmonics per order.
- Gaussian soft-order width: max(2 frequency bins, 0.08 expected frequency).
- Per-band evidence: BPFO, BPFI, kurtosis and log-RMS.
- Cross-block mean, median, standard deviation and occurrence fraction plus the
  registered top-two statistics produce 63 dimensions, as in PU RAMEF-F.

## Frozen classifier

- Feature-wise StandardScaler fitted on source plus target support only.
- Multinomial logistic regression, C=0.1, L-BFGS, max 4000 iterations.
- Source sample weights are inverse-class-frequency balanced and normalized to
  mean one; each target-support run has weight 10.
- A new head is fitted for each encoder seed and representation.

## Compared representations

1. U2-Hz-K1.
2. U2-Hz-Local-M8.
3. U2-Hz-Distributed-M8.
4. RAMEF-F: Distributed-M8 plus 63 physical dimensions.
5. Random63: Distributed-M8 plus deterministic label-independent Gaussian
   dimensions.

## Target firewall

For each seed, all epochs and the checkpoint hash must be written before any
target-query vibration is read. All five checkpoints are frozen before the
single target evaluation event. Target metrics cannot select seeds, bands,
blocks, classifier settings or physical statistics.

## Preregistered positive-result gates

RAMEF receives an independent positive result only if all conditions hold:

1. Mean RAMEF-F Macro-F1 minus Local-M8 Macro-F1 >=2 percentage points.
2. RAMEF-F has higher accuracy than Local-M8 in at least 4/5 seeds.
3. Distributed-M8 is not worse than Local-M8 in mean Macro-F1 and is better in
   at least 3/5 seeds; a complete ceiling tie is not a positive placement result.
4. Mean RAMEF-F IR and OR recall do not fall below Local-M8.
5. RAMEF-F is not worse than Distributed-M8 by more than 1 point in mean
   Macro-F1.
6. Random63 does not match the RAMEF-F improvement.
7. No class/context suffers a catastrophic decline of more than 10 points.

All outcomes, including ceiling or failure, are retained. Gates will not be
changed after target evaluation.
