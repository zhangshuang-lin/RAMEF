"""Python-only replot of the main-text component controls; no model evaluation."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'paper/figures/63d_focused_20261018'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'outputs/staged/PU_UNIQUE_MEASUREMENT_REANALYSIS/results.json'
SUPPORT = ROOT / 'outputs/staged/RAMEF_F_SUPPORT_STABILITY/result.json'

# Contract: panel a compares the adopted 63D workflow with its carrier,
# physical-only and random-dimension controls; panel b shows the registered
# support-deletion endpoint. Query and held-support endpoints stay distinct.
# Existing experimental values and the fixed 171-measurement unit are unchanged.
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.linewidth': 1.0, 'figure.facecolor': 'white'})
models = {r['method']: r for r in json.loads(DATA.read_text(encoding='utf-8'))['models']}
support = json.loads(SUPPORT.read_text(encoding='utf-8'))['metrics']
blue, orange = '#0878b8', '#dd6800'
names = ['Centre U2-Hz', 'U2-Hz-DM4', 'U2-Hz-DM8', 'Physical-only', 'Random63', 'RAMEF-F']
labels = ['Centre', 'DM4', 'DM8', 'Physical', 'Random63', 'RAMEF-F']
colors = ['#838383', '#58ace0', blue, '#009b78', '#8d67a5', orange]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.65), gridspec_kw={'width_ratios': [1.55, 1]})
rows = []
values = [models[k]['accuracy'] for k in names]
axes[0].bar(np.arange(6), np.asarray(values) - 45, bottom=45, color=colors,
            edgecolor='#444444', linewidth=.65, width=.68)
axes[0].set_xticks(np.arange(6), labels, rotation=28, ha='right')
axes[0].set_ylim(45, 108)
axes[0].set_ylabel('Unique-measurement Accuracy (%)')
axes[0].set_title('Post-hoc component attribution', pad=16)
for x, (name, value) in enumerate(zip(names, values)):
    axes[0].text(x, value + 1.4, f'{value:.1f}', ha='center', va='bottom', fontsize=12)
    rows.append({'panel': 'a', 'method': name, 'metric': 'Accuracy', 'value_percent': value,
                 'unit': '171 distinct query measurements'})
for offset, metric, hatch in [(-.18, 'accuracy', ''), (.18, 'recall_OR', '//')]:
    values = [100 * support[k][metric] for k in ['U2_Hz_M8', 'RAMEF_F']]
    axes[1].bar(np.arange(2) + offset, np.asarray(values) - 45, bottom=45, width=.34,
                color=[blue, orange], edgecolor='#444444', hatch=hatch, linewidth=.65)
    for x, value in enumerate(values):
        axes[1].text(x + offset, value + 1.2, f'{value:.1f}', ha='center', va='bottom', fontsize=11)
        rows.append({'panel': 'b', 'method': ['U2-Hz-DM8', 'RAMEF-F'][x],
                     'metric': metric, 'value_percent': value,
                     'unit': '210 held-support predictions over 105 folds'})
axes[1].set_xticks([0, 1], ['U2-Hz-DM8', 'RAMEF-F'])
axes[1].set_ylim(45, 119)
axes[1].set_ylabel('Held-support score (%)')
axes[1].set_title('Query-free support perturbation', pad=16)
axes[1].legend(handles=[Patch(facecolor='white', edgecolor='#444444', label='Accuracy'),
                       Patch(facecolor='white', edgecolor='#444444', hatch='//', label='OR recall')],
               loc='upper left', ncol=2, frameon=False, fontsize=11,
               handlelength=1.15, columnspacing=1)
for letter, ax in zip('ab', axes):
    ax.text(-.10, 1.10, letter, transform=ax.transAxes, fontsize=17, fontweight='bold')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
fig.subplots_adjust(left=.085, right=.99, bottom=.23, top=.84, wspace=.36)
fig.savefig(OUT / 'Fig5_63D_Focused.svg', bbox_inches='tight')
fig.savefig(OUT / 'Fig5_63D_Focused.pdf', bbox_inches='tight')
fig.savefig(OUT / 'Fig5_63D_Focused.png', dpi=260, bbox_inches='tight')
plt.close(fig)
with (OUT / 'Fig5_source.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'figure': str(OUT / 'Fig5_63D_Focused.png'), 'data_rows': len(rows),
                  'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA, SUPPORT]},
                  'new_model_evaluation': False}, indent=2))
