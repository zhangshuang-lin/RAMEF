# Current figure source map

These are source scripts, not an all-figures one-click renderer. Stage-relative paths and auxiliary source assets must be supplied as documented. No raw waveform is fabricated or redistributed.

| Paper item | Source | Prerequisite / identity |
|---|---|---|
| Fig1 | make_ramef_workflow_v6.py + make_ramef_workflow_v5.py | Real source waveform required; not bundled |
| Fig2-4 | make_pu_unique_measurement_figures.py | Selected-panel copy; original Fig5 block removed |
| Fig5 | make_fig5_63d_focused_20261018.py | Current full63 component/support plot |
| Fig6/8/11 | refine_figures_6_8_11_v2_20261014.py + repair_figures_6_8_11_20261014.py | Both dynamic-import files required |
| Fig7 | make_pu_unique_tsne_figure.py | Frozen coordinates; no refitting |
| Fig9/10 | make_mcc5_transferred_figures.py | Current root version, not old release copy |
| FigS1 | make_pu_unique_measurement_figures.py | Original FigS5 seed plot mapped to manuscript S1 |

Numerical inputs are under `reference/outputs/staged/`. To use a historical plotting script, work in a new sandbox, recreate its original `paper/figures/scripts` / `outputs/staged` layout from these files, and supply any separately obtained waveform assets. Do not overwrite reference snapshots. Times New Roman was used in the paper; font availability on a different machine must be checked.
