# RACP_FAIR_ABLATION protocol

Status: frozen after the final RACP candidate evaluation and before running
these ablation heads.  This is post-hoc attribution, never candidate selection.

All variants use the same 1020 source measurements, 15 target supports, 240
query measurements, frozen A1, source/support weights 1/10, StandardScaler and
LogisticRegression(C=0.1).  Query labels cannot select or alter a variant.

| ID | Fixed representation |
|---|---|
| B0 | exact centre-crop A1 |
| B1 | mean of registered blocks 0,2,5,7 |
| B2 | mean of all eight A1 blocks |
| B3 | 63-D RACP physical descriptor only |
| B4 | A1 mean8 + 30-D static physical statistics; persistence removed |
| B5 | A1 mean8 + full 63-D RACP (locked final method) |
| B6 | A1 mean8 + deterministic 63-D Gaussian nuisance control |

The 30-D static descriptor retains per-band and cross-band mean/median values
but removes standard deviation, occurrence fraction and top-two persistence.
B6 is generated from SHA256(measurement_id, seed=20260821), independent of
signals and labels.

RACP has an independent core contribution if B5 exceeds B2 by at least 2
accuracy percentage points and 5 OR-recall percentage points, and exceeds B4
by at least 1 accuracy point.  B6 must not reproduce the B5 gain.
