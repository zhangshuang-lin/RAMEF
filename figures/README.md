# Current figure source map

`current/` contains the **exact final submission-package SVGs** for Figs. 1–11,
Fig. S1 and the graphical abstract. They were copied, not newly plotted during
packaging. `current/ARTWORK_MANIFEST.json` binds their hashes to the manuscript.
These are publication artwork, not a deposit of raw recordings.

| Paper item | Archived source | Prerequisite / identity |
|---|---|---|
| Fig. 1 | `source/fig1/make_ramef_workflow_v16_linked_observation_20261022.py` → `make_ramef_workflow_v15_corrected_20261022.py` | Current linked observation panel; external waveform-derived CSVs required |
| Figs. 2–4 | `scripts/make_pu_unique_measurement_figures.py` | Selected-panel copy; original Fig. 5 block removed |
| Fig. 5 | `scripts/make_fig5_63d_focused_20261018.py` | Full63 component/support plot |
| Figs. 6/8/11 | `scripts/refine_figures_6_8_11_v2_20261014.py` + `scripts/repair_figures_6_8_11_20261014.py` | Both dynamic-import files required; final SVGs are supplied separately |
| Fig. 7 | `scripts/make_pu_unique_tsne_figure.py` | Frozen coordinates; no t-SNE refitting |
| Figs. 9/10 | `scripts/make_mcc5_transferred_figures.py` | Staged result inputs; final SVGs are supplied separately |
| Fig. S1 | `scripts/make_pu_unique_measurement_figures.py` | Historical Fig. S5 seed plot mapped to manuscript S1 |

The old v5/v6 workflow scripts remain under `scripts/` for provenance. They are
**not** the source of the current Fig. 1. Historical plotting scripts may retain
old labels or need subsequent layout changes; use the supplied final SVGs when
an exact final-artwork identity is required, not an untested one-click rebuild.

## Figure 1 source and external inputs

The two current scripts are byte-preserved from the paper project. They expect
the original layout, which can be recreated in a **new sandbox**:

```text
SANDBOX/
  paper/figures/scripts/
    make_ramef_workflow_v15_corrected_20261022.py
    make_ramef_workflow_v16_linked_observation_20261022.py
  paper/figures/Fig1_RAMEF_CleanAxes_20261020/
    clean_axes_audit.json
    overview_bin_ranges.csv       # external; not bundled
    raw_local_detail.csv          # external; not bundled
```

The non-waveform `clean_axes_audit.json` metadata is included in `source/fig1/`.
The two CSVs are source-derived waveform assets and are not redistributed here:

| External asset | Expected SHA-256 |
|---|---|
| `overview_bin_ranges.csv` | `5bbc38a2db1bdb32ce6b4cd6f08c663efe811e223573cd5112f2350b9a0222ff` |
| `raw_local_detail.csv` | `9ec91aa7fcda05639866854e2714ccb2b3ad4f880ad5abad6e11cc80bbd80107` |

The metadata identifies PU source-training record `N15_M07_F10_KA22_14`, channel
`vibration_1`, nominal sampling rate 64 kHz, the registered block coordinates
and the original record hash. The overview is a min–max display of the measured
record; the detail is 640 raw samples from the first 10 ms of block 4. The scripts
must not substitute a synthetic trace for these inputs.

Only after obtaining/deriving the authorised inputs and matching their hashes,
install the plotting dependencies and a legally available Times New Roman font,
then run in that sandbox:

```text
python paper/figures/scripts/make_ramef_workflow_v16_linked_observation_20261022.py
```

The archived source writes its SVG/PDF and preview files to
`paper/figures/Fig1_RAMEF_Linked_Observation_20261022/`. No raw waveform loading
or figure rebuilding was performed when this upload package was assembled.
Missing CSVs do not prevent viewing `current/Figure_01.svg` or replaying metrics.
