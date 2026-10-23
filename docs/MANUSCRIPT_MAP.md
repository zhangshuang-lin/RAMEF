# Manuscript to code and evidence

The current content binding is the final clarified English main manuscript and
supplement prepared on 23 October 2026. Exact filenames and SHA-256 values are in
`MANUSCRIPT_VERSION.json`; the manuscript files themselves are not distributed.

| Manuscript section | Code / evidence |
|---|---|
| Sections 2–3: observation and model | `code/racp_or.py`; `code/u2_stft_model.py`; `configs/active_parameters.json` |
| Section 3.3: physical63 | `code/racp_or.py` (PU); `code/mcc5_confirmatory_data.py` (MCC5) |
| Section 3.4: calibration | Historical PU calibration runners; `code/run_mcc5_ramef_confirmatory.py`; `docs/SCRIPT_INDEX.md` |
| Sections 4.2–4.4 / Tables 5–6 | `evidence/pu/predictions.csv`: locked, calibration, budget and seed cohorts |
| Section 4.5 / Tables 8–9 | `evidence/mcc5/predictions.csv`: primary and equal-input cohorts |
| Table 10 / Tables S9–S11 | Capacity/factorial prediction cohorts and archived source scripts |
| PU uncertainty | `reference/pu_paired_bootstrap/`; run in a copied/new output sandbox only |
| MCC5 support and paired statistics | `reference/outputs/staged/MCC5_TRANSFERRED_EVIDENCE_PACKAGE/` |
| Section 4.1.6 / CWRU and SQ | `code/run_cwru_ramef_cross_load.py`; `code/run_sq_ramef_carrier.py`; `protocols/` |
| Fig. 1 | Current v16 → v15 chain in `figures/source/fig1/`; exact SVG in `figures/current/Figure_01.svg` |
| Figs. 2–11 / Fig. S1 | `figures/README.md`; exact final artwork and its checksums in `figures/current/` |

RAMEF is the current display name. RAMEF-F remains a historical alias in frozen
fields and source filenames. The original MCC5 96.30% result is not replaced by
the later compact-fusion 99.26% post-target control. Later same-physics and
capacity comparisons remain post-target evidence.

The final manuscript's explanatory clarifications do not define new experiments:
Butterworth `butter` order argument 4 is distinguished from effective filter
order; descriptive comparisons are not formal non-inferiority tests; the
registered MCC5 `bearing_context` groups are not the Fig. 11d nuisance families.
Historical source/protocol wording is preserved as evidence rather than silently
rewritten to match later prose.
