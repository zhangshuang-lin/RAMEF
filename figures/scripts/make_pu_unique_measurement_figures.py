"""Replot PU primary figures using one vote per physical measurement ID.

Contract: show the locked centre -> distributed -> fused comparison without
counting repeated query-window entries as independent diagnoses. Only panels used as Figures 2, 3, 4 and S1 are generated here.
Python/matplotlib is the established project plotting backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(r"D:\u2_stft_standalone")
DATA = ROOT / "outputs/staged/PU_UNIQUE_MEASUREMENT_REANALYSIS/results.json"
SUPPORT = ROOT / "outputs/staged/RAMEF_F_SUPPORT_STABILITY/result.json"
BUDGET = ROOT / "outputs/staged/PU_UNIQUE_BUDGET_RECONSTRUCTION/results.json"
OUT = ROOT / "paper/figures/unique_measurement"
QA = ROOT / "tmp/pu_unique_figure_qa"
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .9, "figure.facecolor": "white", "axes.facecolor": "white",
})
GREY, BLUE, ORANGE, PURPLE = "#838383", "#0878b8", "#dd6800", "#8d67a5"
CLASSES = ("N", "IR", "OR")


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(QA / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def heat(ax, matrix, title: str, letter: str) -> None:
    array = np.asarray(matrix)
    ax.imshow(array, cmap="Blues", vmin=0, vmax=max(array.max(), 60), aspect="equal")
    for r in range(3):
        for c in range(3):
            value = int(array[r, c])
            ax.text(c, r, str(value), ha="center", va="center",
                    color="white" if value >= 35 else "#111111", fontsize=13)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=11)
    ax.text(-.14, 1.08, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=15)
    ax.tick_params(length=0)


def main() -> None:
    content = json.loads(DATA.read_text(encoding="utf-8"))
    model = {row["method"]: row for row in content["models"]}
    centre, dm8, ramef = (model[k] for k in ("Centre U2-Hz", "U2-Hz-DM8", "RAMEF-F"))

    # Fig. 2: localise centre-crop errors, including difficult code identities.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})
    heat(axes[0], centre["confusion_matrix"], "Centre crop: 171 records", "a")
    code_items = [(code, info["accuracy"], info["total"])
                  for code, info in centre["code_accuracy"].items() if code.startswith("KA")]
    code_items.sort()
    names = [x[0] for x in code_items]
    values = [x[1] for x in code_items]
    colors = [ORANGE if name in ("KA22", "KA30") else GREY for name in names]
    axes[1].bar(names, values, color=colors, edgecolor="#333333", linewidth=.45, width=.62)
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Outer-race bearing codes", pad=11)
    axes[1].text(-.08, 1.08, "b", transform=axes[1].transAxes,
                 fontweight="bold", fontsize=15)
    for i, (_, val, count) in enumerate(code_items):
        axes[1].text(i, val + 2, f"{val:.1f}\n(n={count})", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=.22)
    axes[1].set_axisbelow(True)
    fig.subplots_adjust(wspace=.35)
    save(fig, "Fig2_PU_unique_failure")

    # Fig. 3: all four originally fixed budget levels now use the same 171 IDs.
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))["results"]
    names = ["K1", "K2", "K4", "K8"]
    xpos = np.arange(4)
    fig = plt.figure(figsize=(10.2, 4.35))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=.42)
    ax = fig.add_subplot(grid[0, 0])
    for key, label, color, style in (("accuracy", "Accuracy", BLUE, "o-"),
                                     ("macro_f1", "Macro-F1", ORANGE, "s--")):
        values = [100 * budget[name]["unique_measurements"][key] for name in names]
        ax.plot(xpos, values, style, color=color, linewidth=1.7, markersize=6, label=label)
        for x, value in zip(xpos, values):
            ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 7 if key == "accuracy" else -16),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(xpos, names)
    ax.set_ylim(82, 103)
    ax.set_ylabel("Score (%)")
    ax.set_title("Overall scores", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.text(-.14, 1.08, "a", transform=ax.transAxes, fontweight="bold", fontsize=15)
    ax = fig.add_subplot(grid[0, 1])
    for key, label, color, style in (("recall_N", "N", GREY, "o-"),
                                     ("recall_IR", "IR", BLUE, "s--"),
                                     ("recall_OR", "OR", ORANGE, "^-")):
        values = [100 * budget[name]["unique_measurements"][key] for name in names]
        ax.plot(xpos, values, style, color=color, linewidth=1.7, markersize=6, label=label)
    ax.set_xticks(xpos, names)
    ax.set_ylim(60, 107)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Class recall", pad=10)
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.text(-.14, 1.08, "b", transform=ax.transAxes, fontweight="bold", fontsize=15)
    heat(fig.add_subplot(grid[0, 2]), budget["K8"]["unique_measurements"]["confusion_matrix"],
         "K8 confusion", "c")
    save(fig, "Fig3_PU_unique_budget")

    # Fig. 4: a quantitative comparison plus paired transitions and two
    # separate confusion matrices; there is no rejected-query subset.
    fig = plt.figure(figsize=(10.2, 7.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[.9, 1.0], hspace=.48, wspace=.42)
    ax = fig.add_subplot(grid[0, :2])
    items = [centre, dm8, ramef]
    labels = ["Centre crop", "U2-Hz-DM8", "RAMEF-F"]
    colors = [GREY, BLUE, ORANGE]
    x = np.arange(3)
    for offset, metric, hatch in [(-.23, "accuracy", ""), (0, "macro_f1", "//"), (.23, "recall_OR", "xx")]:
        values = [item[metric] for item in items]
        ax.bar(x + offset, values, width=.22, color=colors,
               hatch=hatch, edgecolor="#444444", linewidth=.5,
               label={"accuracy":"Accuracy", "macro_f1":"Macro-F1", "recall_OR":"OR recall"}[metric])
    ax.set_xticks(x, labels)
    ax.set_ylim(60, 116)
    ax.set_ylabel("Score (%)")
    ax.set_title("Locked methods on unique measurements", pad=10)
    ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.text(-.08, 1.08, "a", transform=ax.transAxes, fontweight="bold", fontsize=15)
    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    ax.text(.02, 1.05, "b", transform=ax.transAxes, fontweight="bold", fontsize=15)
    ax.text(.03, .88, "Centre crop → RAMEF-F", fontsize=12)
    ax.text(.03, .64, "17 errors corrected", fontsize=12, color=ORANGE)
    ax.text(.03, .43, "0 errors introduced", fontsize=12)
    ax.text(.03, .20, "171 distinct records", fontsize=11)
    heat(fig.add_subplot(grid[1, 0]), centre["confusion_matrix"], "Centre crop", "c")
    heat(fig.add_subplot(grid[1, 1]), dm8["confusion_matrix"], "U2-Hz-DM8", "d")
    heat(fig.add_subplot(grid[1, 2]), ramef["confusion_matrix"], "RAMEF-F", "e")
    save(fig, "Fig4_PU_unique_locked")

    # Revised S5: the five new seeds are scored on the same 171 distinct IDs.
    seed_items = [model[f"RAMEF_F seed {seed}"] for seed in range(20260911, 20260916)]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for label, metric, color, ax in zip(("Accuracy", "Macro-F1"),
                                        ("accuracy", "macro_f1"), (ORANGE, BLUE), axes):
        points = np.asarray([item[metric] for item in seed_items])
        mean = points.mean()
        sd = points.std(ddof=1)
        ax.scatter(np.arange(5), points, s=90, color=color, edgecolors="white", zorder=3)
        ax.errorbar(5.8, mean, yerr=sd, fmt="D", color="#222222", capsize=8, markersize=8)
        ax.axhline(mean, color=color, linewidth=1.3, alpha=.8)
        ax.set_xlim(-.6, 6.4)
        ax.set_ylim(85, 102)
        ax.set_xticks([0,1,2,3,4,5.8], ["11","12","13","14","15","Mean\n±SD"])
        ax.set_title(f"{label}: {mean:.2f} ± {sd:.2f}%", pad=10)
        ax.set_ylabel(f"{label} (%)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.text(.5, -.02, "Encoder seed suffix (202609xx); 171 unique PU measurements",
             ha="center", fontsize=10)
    fig.subplots_adjust(wspace=.23, bottom=.25)
    save(fig, "FigS1_PU_unique_five_seeds")


if __name__ == "__main__":
    main()
