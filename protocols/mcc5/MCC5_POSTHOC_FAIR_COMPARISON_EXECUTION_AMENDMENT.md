# MCC5 supplementary fairness event activation

This activates one execution of the already prepared eight-view, five-seed
comparison. The instruction received for this stage was:

“下一步该做什么，明确后开始下一步工作”

The scoped next action was identified as the prepared MCC5 fixed-query fairness
comparison and announced before execution. This activation applies only to a
separately locked post-target supplementary event on the existing 27 MCC5 runs.
It does not activate any PU query evaluation or replace the original MCC5 event.

All 40 scaler/head files, model checkpoints, dimensions and regularization
settings remain those in the completed preparation registry. Their original
query-disabled fields record preparation status and are not overwritten.
The executor requires this separate activation record and a new permanent lock.

All eight views and five seeds are reported once. No fitting, parameter search,
seed selection or checkpoint selection is performed after loading query data.
The cached registered query blocks and physical descriptors are used directly;
the raw MCC5 CSVs are not reread. Original prediction replay is checked only
after the complete new prediction grid is produced.

The primary fairness contrast is unique carrier128 C=0.1 minus width-matched
WDCNN128 C=0.1. Other fixed contrasts are expanded-minus-original WDCNN,
full-fusion-minus-expanded WDCNN, full-fusion-minus-original carrier and
PCA128-minus-unique carrier128. Token/head equivalence and the three original
reference prediction ledgers are checked without modifying a head on mismatch.

For these five contrasts, retain the mean/SD across seeds, paired error counts,
and 10,000 hierarchical paired bootstrap replicates (seed 20261005). Resample
five seeds and the same 13 NRF, 7 IR and 7 OR run identities within classes;
compute metrics per sampled seed and then average, matching the mean-seed
reporting unit. Percentile intervals describe internal sensitivity of this
fixed cohort only. Every outcome is retained, including negative differences.
