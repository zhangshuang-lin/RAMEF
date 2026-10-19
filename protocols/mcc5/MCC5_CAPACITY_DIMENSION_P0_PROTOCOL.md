# MCC5 capacity and embedding dimension audit

This protocol is frozen before this audit's source training. It is a post-target,
query-free audit using the existing 42 source-training runs, 42 source-validation
runs and 15 labelled target supports. It does not constitute a new target event.

## Fixed comparisons

1. Original source-selected U2-Hz-DM8: 1024-D carrier embedding.
2. The same U2-Hz-DM8 compressed to 128 dimensions by unwhitened PCA fitted
   exclusively on its 336 source-training block embeddings. PCA is applied to
   every block before the eight-block mean. Source run means alone have only
   42 rows and cannot support a 128-component source-only PCA; fitting on query
   or target support to increase rank is prohibited.
3. Original source-selected WDCNN-DM8: 128-D embedding and original checkpoints.
4. A width-expanded WDCNN-DM8 with 128-D embedding. Its first channel count is
   the integer in 16..128 minimizing absolute parameter-count difference from
   the original U2-Hz, counting each neural training head. All later channel
   ratios, convolution kernels, strides, pooling and network depth are retained.
   The required parameter-count difference is at most 1%.

The PCA contrast tests a change in output dimensionality of one frozen
representation. The WDCNN contrast tests channel capacity within one model
family. Neither makes all architectural inductive biases identical.

## Data and optimization

- Development cache only; role and identity checks must reject any query row.
- No target-query vibration, query feature cache, query label or prediction
  ledger is loaded by the runner. Its code has no target evaluation stage.
- Five paired seeds: 20261101, 20261102, 20261103, 20261104, 20261105.
- Eight registered distributed 3072-point blocks per run; no physical descriptor.
- Width-expanded WDCNN: AdamW, lr=0.001, weight decay=0.0001, batch size=12,
  12 epochs, source class-balanced block sampler and gradient clipping=5.
- Source-validation neural Macro-F1 alone selects each new checkpoint; every
  seed is retained. Support metrics do not select widths, epochs or settings.
- Common final head: StandardScaler plus LogisticRegression(C=0.1,
  solver=lbfgs, max_iter=4000), inverse-frequency source weights normalized to
  mean one, target-support weight=10. No parameter grid is searched.

## Readouts and interpretation

Report every seed and five-seed mean/sample SD for:

- 42-run source-validation performance of a source-only common linear head;
- source-validation performance after adding the 15 fixed target supports;
- leave-one-support-out predictions (15 folds per seed) using source plus the
  remaining 14 supports; held support excluded from scaling and head fitting;
- source-block PCA retained variance, parameters, dimensions, training history,
  best epoch and original/final frozen checkpoint hashes.

Complete means role firewall, width tolerance, PCA source-only fit and archived
metrics all passed. There is no target-accuracy success gate in this P0. Strong
source validation or support LOO cannot establish improved query accuracy.
Weak source scores cannot prove that matching dimension/capacity is useless
on a different target. No manuscript target result is replaced by this audit.
