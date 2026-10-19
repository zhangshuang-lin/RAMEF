# MCC5 frozen supplementary capacity and dimension comparison

Status: PREPARED_QUERY_EXECUTION_DISABLED.

This defines a possible supplementary comparison on the historically evaluated
27-run MCC5 cohort. It is not a new external confirmation or a change to the
original MCC5 primary event. The present preparation reads development features
only and fits every head before any new target prediction is produced.

## Fixed data and models

Keep the original 42 source, 42 source-validation and 15 target-support roles,
eight distributed 3072-point blocks, class-balanced source weights and support
weight 10. Use all five existing paired seeds 20261101--20261105. The original
U2-Hz and WDCNN checkpoints and the new width-matched WDCNN checkpoints are
source-selected and hash-frozen. There is no query-based seed or model selection.

Prepare the following eight views for every seed:

| View | Head input | C | Purpose |
|---|---:|---:|---|
| Original carrier interface | 1024 | 0.1 | Historical carrier specification |
| Unique carrier, common penalty | 128 | 0.1 | Same dimension/common C comparison with WDCNN |
| Unique carrier, equivalent penalty | 128 | 0.8 | Analytic equivalence to duplicated carrier |
| Source-block PCA carrier | 128 | 0.1 | Initial P0 projection control; all seeds retained |
| Original WDCNN | 128 | 0.1 | Original baseline capacity |
| Width-matched WDCNN | 128 | 0.1 | Approximately matched total parameter count |
| Original full fusion | 1087 | 0.1 | Historical complete descriptor specification |
| Compact equivalent fusion | 191 | 0.1 | Carrier coordinates scaled by sqrt(8) after standardization |

StandardScaler is fitted on source plus the 15 supports for each view. All
heads use L-BFGS and max_iter=4000. PCA128 is fixed from the 336 source-block
embeddings stored during P0 and is applied before block averaging.

## Admissible contrasts

The primary capacity/dimension fairness contrast is unique U2-Hz carrier128
C=0.1 versus width-matched WDCNN128 C=0.1. Within WDCNN, compare expanded versus
original width. The C=0.8 and compact fusion controls test the analytically
specified repeated-coordinate penalty, not a searched optimum. Report all
eight views and five seeds irrespective of result direction.

If a target event is explicitly authorized later, score each of the same 27
query runs once per view/seed. Report Accuracy, Macro-F1, NRF/IR/OR recalls,
paired error transitions and the full seed table. Five seeds share the same
cohort and do not create independent target cohorts. Capacity matching does
not establish identical FLOPs, active branches, receptive fields or convergence.

## Execution permission

The prepared archive contains 40 frozen scalers/heads and their hashes, but
no target executor. A later execution requires an explicit amendment allowing
one separately locked post-target supplementary event. The original target
lock must remain intact. The new event cannot choose a method, alter the main
result or be described as preregistered external confirmation. No 27-run
query features, labels or predictions are read by the preparation program.
