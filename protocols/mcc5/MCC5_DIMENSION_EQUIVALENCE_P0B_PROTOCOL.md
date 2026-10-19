# MCC5 carrier dimension equivalence check

This exploratory follow-up was proposed after the query-free capacity P0
revealed near-total retained PCA variance. It is not part of the initial P0
protocol and must be reported separately. It uses only the frozen 99-run
development embeddings, their source/support labels and development physical
descriptors. No target-query file or prediction is read.

The Hz-only frontend supplies identical neutral coherence weights to its
eight-token pooling interface. The hypothesis is therefore deterministic:
the nominal 1024-D vector consists of eight identical copies of a 128-D vector.

Compare the following analytically defined linear-head conditions, with all
five seeds and the same source-validation/source+support/LOO readouts as P0:

1. Original 1024-D carrier, StandardScaler and C=0.1.
2. First 128-D token only, StandardScaler and C=0.1.
3. First 128-D token only, StandardScaler and C=0.8. The factor eight follows
   from the minimum L2 norm of coefficients across eight identical feature
   copies; it is not selected by a performance search.
4. Original 1087-D carrier-plus-physical view, StandardScaler and C=0.1.
5. 191-D view: first 128 carrier coordinates plus the unchanged 63 physical
   coordinates. After standardization multiply only carrier coordinates by
   sqrt(8), then fit the same C=0.1 head. This is an isometry of the duplicated
   standardized carrier and preserves the physical branch penalty.

Verify exact token equality and prediction equality between conditions 1/3
and 4/5. Record differences in fitted probabilities, rather than assuming
finite-iteration solvers return bitwise-identical coefficients. Any difference
between conditions 1/2 cannot by itself establish loss of fault information:
removing duplicated features changes the effective L2 penalty unless corrected.
