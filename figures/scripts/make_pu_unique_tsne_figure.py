"""Draw frozen PU feature-geometry audit with one target vote per measurement.

The t-SNE maps were fitted only on source/support in the archived audit. This
plot reuses their frozen coordinates and removes repeated target identities;
labels are applied only after embedding for a descriptive visualisation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(r"D:\u2_stft_standalone")
DATA = ROOT / "outputs/staged/PU_UNIQUE_TSNE_REANALYSIS"
OUT = ROOT / "paper/figures/unique_measurement"
QA = ROOT / "tmp/pu_unique_figure_qa"
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
METHODS = ("Centre crop", "U2-Hz-DM8", "RAMEF")
DISPLAY_NAMES = {"RAMEF": "RAMEF-F"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
METHOD_COLORS = ("#777777", "#0878b8", "#dd6800")


def main() -> None:
    with (DATA / "unique_coordinates.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = json.loads((DATA / "results.json").read_text(encoding="utf-8"))["results"]
    fig = plt.figure(figsize=(10.2, 6.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0], hspace=.34, wspace=.28)
    for idx, title in enumerate(METHODS):
        ax = fig.add_subplot(grid[0, idx])
        data = [row for row in rows if row["representation"] == title]
        for cls in (0, 1, 2):
            source = [r for r in data if r["role"] == "source" and int(r["label"]) == cls]
            support = [r for r in data if r["role"] == "support" and int(r["label"]) == cls]
            query = [r for r in data if r["role"] == "query" and int(r["label"]) == cls]
            ax.scatter([float(r["x"]) for r in source], [float(r["y"]) for r in source],
                       s=7, color=CLASS_COLORS[cls], alpha=.16, linewidths=0)
            ax.scatter([float(r["x"]) for r in query], [float(r["y"]) for r in query],
                       s=23, color=CLASS_COLORS[cls], alpha=.72, edgecolors="white", linewidths=.25)
            ax.scatter([float(r["x"]) for r in support], [float(r["y"]) for r in support],
                       s=50, marker="^", color=CLASS_COLORS[cls],
                       edgecolors="black", linewidths=.6, zorder=4)
            wrong = [r for r in query if r["correct"] == "0"]
            ax.scatter([float(r["x"]) for r in wrong], [float(r["y"]) for r in wrong],
                       marker="x", color="#111111", s=38, linewidths=1.1, zorder=5)
        ax.set_title(f"{DISPLAY_NAMES.get(title, title)}  (errors={result[title]['unique_measurements']['locked_query_errors']})")
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2" if idx == 0 else "")
        ax.text(-.08, 1.08, "abc"[idx], transform=ax.transAxes, fontweight="bold", fontsize=14)
        ax.tick_params(length=2)
    legend = [Line2D([0],[0],marker="o",linestyle="none",color=CLASS_COLORS[c],label=("N","IR","OR")[c])
              for c in (0,1,2)]
    legend += [Line2D([0],[0],marker=".",linestyle="none",color="#666666",label="Source"),
               Line2D([0],[0],marker="^",linestyle="none",markerfacecolor="white",
                      markeredgecolor="black",label="Target support"),
               Line2D([0],[0],marker="o",linestyle="none",markerfacecolor="white",
                      markeredgecolor="black",label="Target query"),
               Line2D([0],[0],marker="x",linestyle="none",color="#111111",label="Query error")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, 1.02),
               ncol=7, frameon=False, fontsize=8)

    metrics = [("silhouette_high_dim", "High-dimensional silhouette", 1.0),
               ("reference_5nn_query_accuracy", "Reference 5-NN accuracy (%)", 100.0),
               ("locked_query_errors", "Locked query errors / 171", 1.0)]
    for j, (key, title, factor) in enumerate(metrics):
        ax = fig.add_subplot(grid[1, j])
        vals = [factor * result[name]["unique_measurements"][key] for name in METHODS]
        ax.bar(range(3), vals, color=METHOD_COLORS, edgecolor="#444444", linewidth=.4, width=.6)
        ax.set_xticks(range(3), ["Centre", "DM8", "RAMEF-F"])
        ax.set_title(title, pad=6)
        ax.set_ylim(0, max(vals) * 1.2)
        for i, value in enumerate(vals):
            ax.text(i, value + max(vals) * .035, f"{value:.3f}" if factor == 1.0 and key != "locked_query_errors"
                    else f"{value:.1f}" if factor == 100.0 else str(int(value)),
                    ha="center", fontsize=8)
        ax.text(-.08, 1.12, "def"[j], transform=ax.transAxes, fontweight="bold", fontsize=14)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.subplots_adjust(top=.86, bottom=.08, left=.08, right=.98)
    stem = "Fig7_PU_unique_feature_geometry"
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(QA / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
