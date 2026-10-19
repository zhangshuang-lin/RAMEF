"""Create the MSSP-readable RAMEF workflow with real registered signal views."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import make_ramef_workflow_v5 as source

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "figures" / "export" / "Fig1_RAMEF_workflow_v6"
MM = 1 / 25.4

INK = "#20252B"; MUTED = "#59636E"; GREY = "#7B838B"; LINE = "#A9B4BE"
BLUE = "#1769AA"; BLUE_MID = "#58A7D5"; BLUE_LIGHT = "#E5F1F8"
ORANGE = "#D66B1F"; ORANGE_LIGHT = "#FAEBDD"; TEAL = "#16867A"; TEAL_LIGHT = "#E3F2ED"
PANEL = "#F7F9FB"; WARM = "#FCF8F4"; RED = "#B64A4A"


def setup():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 7.8, "axes.titlesize": 8.2, "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def box(ax, x, y, w, h, title, subtitle="", face="#FFFFFF", edge=GREY,
        title_size=8.0, subtitle_size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.004,rounding_size=.008",
                                facecolor=face, edgecolor=edge, linewidth=.85, zorder=2))
    ax.text(x+w/2, y+h*(.62 if subtitle else .5), title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    if subtitle:
        ax.text(x+w/2, y+h*.28, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=MUTED, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GREY, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                linewidth=.9, color=color,
                                connectionstyle=f"arc3,rad={rad}", zorder=5))


def panel(ax, x, y, w, h, letter, title, face=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.006,rounding_size=.010",
                                facecolor=face, edgecolor=GREY, linewidth=.85,
                                linestyle=(0, (3, 2)), zorder=0))
    ax.text(x+.012, y+h-.025, letter, fontsize=10.5, fontweight="bold", va="top")
    ax.text(x+.044, y+h-.025, title, fontsize=9.2, fontweight="bold", va="top")


def main():
    setup()
    signal, starts, blocks, freq, time, stft_db, orders, power, evidence = source.prepare_signal_views()
    fig = plt.figure(figsize=(190*MM, 125*MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .982, "Rotation-aware measurement evidence fusion (RAMEF)",
            ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
    ax.text(.5, .947, "Distributed observation, physically separated evidence, and support calibration",
            ha="center", va="top", fontsize=7.5, color=MUTED)

    panel(ax, .015, .145, .265, .765, "a", "Distributed observation")
    panel(ax, .290, .145, .430, .765, "b", "Carrier–soft-order evidence")
    panel(ax, .730, .145, .255, .765, "c", "Calibration and inference", WARM)

    # a — real source measurement and registered non-overlapping positions.
    aw = fig.add_axes([.047, .665, .200, .125])
    idx = np.linspace(0, len(signal)-1, 1800).astype(int)
    aw.plot(idx/source.FS, signal[idx], color=BLUE, lw=.45)
    for k, s in enumerate(starts):
        aw.axvspan(s/source.FS, (s+source.WINDOW)/source.FS,
                   color=BLUE_MID, alpha=.12 if k != 3 else .28, lw=0)
    aw.set_xlim(0, len(signal)/source.FS); aw.set_yticks([])
    aw.set_title("One physical measurement", fontweight="bold", pad=2)
    box(ax, .045, .555, .205, .075, "Eight distributed blocks", "16,384 samples · 0% overlap",
        BLUE_LIGHT, BLUE)
    for k in range(8):
        x = .050 + k*.0245
        ax.add_patch(Rectangle((x, .465), .019, .052, facecolor=BLUE if k==3 else BLUE_MID,
                               edgecolor="white", linewidth=.5))
        ax.text(x+.0095, .491, f"B{k+1}", ha="center", va="center", color="white",
                fontsize=7.2, fontweight="bold")
    arrow(ax, .148, .555, .148, .520, BLUE)
    box(ax, .050, .355, .195, .072, "Measurement-level decision", "1 measurement = 1 label = 1 shot",
        "#FFFFFF", BLUE)
    arrow(ax, .148, .465, .148, .431, BLUE)
    box(ax, .050, .225, .195, .090, "Rotation budget", "B = 8 · 30.72 nominal rev at 900 rpm",
        TEAL_LIGHT, TEAL)
    arrow(ax, .148, .355, .148, .319, TEAL)
    ax.text(.148, .180, r"Output: $\{\mathbf{x}_{i,1},\ldots,\mathbf{x}_{i,8}\}$",
            ha="center", fontsize=8.6, fontweight="bold", color=INK)

    # b — top learned carrier lane.
    ax.text(.315, .825, "Learned carrier path", fontsize=8.3, fontweight="bold", color=BLUE)
    st = fig.add_axes([.315, .605, .115, .185])
    st.imshow(stft_db, origin="lower", aspect="auto", extent=[time.min(),time.max(),0,12], cmap="cividis")
    st.set_ylabel("kHz"); st.set_title("log-magnitude STFT", pad=2)
    box(ax, .455, .650, .085, .100, "U2-Hz", "nested RSU", BLUE_LIGHT, BLUE, 8.3, 7.2)
    box(ax, .565, .650, .115, .100, "Mean over 8 blocks", r"$\bar{\mathbf{z}}_i\in\mathbb{R}^{1024}$",
        "#FFFFFF", BLUE, 7.6, 8.6)
    arrow(ax, .431, .697, .452, .697, BLUE); arrow(ax, .542, .697, .562, .697, BLUE)

    # b — bottom parameter-free physical lane.
    ax.text(.315, .535, "Parameter-free physical path", fontsize=8.3, fontweight="bold", color=ORANGE)
    od = fig.add_axes([.315, .300, .115, .175])
    od.plot(orders, power, color=INK, lw=.8)
    for name, order, color in (("BPFO", 3.05, ORANGE), ("BPFI", 4.95, TEAL)):
        for h in range(1,5): od.axvline(h*order, color=color, lw=.55, ls="--", alpha=.65)
    od.set_xlim(0,22); od.set_ylim(0,1.04); od.set_xlabel("Order"); od.set_ylabel("Norm. power")
    od.set_title("Order evidence", pad=2)
    hm = fig.add_axes([.450, .300, .090, .175])
    hm.imshow(evidence, aspect="auto", cmap="YlOrBr")
    hm.set_xticks(range(6), ["O1","I1","O2","I2","O3","I3"])
    hm.set_yticks([0,3,7], ["B1","B4","B8"]); hm.set_title("Cross-block evidence", pad=2)
    box(ax, .565, .325, .115, .105, "Cross-block statistics", "mean · median · SD · recurrence",
        ORANGE_LIGHT, ORANGE, 7.5, 7.0)
    arrow(ax, .431, .386, .447, .386, ORANGE); arrow(ax, .542, .386, .562, .386, ORANGE)
    box(ax, .500, .185, .160, .075, "Late fusion", r"$\mathbf{r}_i=[\bar{\mathbf{z}}_i;\mathbf{p}_i]\in\mathbb{R}^{1087}$",
        TEAL_LIGHT, TEAL, 8.2, 8.6)
    arrow(ax, .622, .650, .605, .263, BLUE, .16); arrow(ax, .622, .325, .605, .263, ORANGE, -.08)

    # c — calibration and query firewall.
    ax.text(.755, .850, "Calibration (labelled data only)", fontsize=8.0, fontweight="bold", color=MUTED)
    box(ax, .750, .755, .100, .070, "Source bank", "1,020 · weight 1", BLUE_LIGHT, BLUE, 7.6, 7.0)
    box(ax, .865, .755, .100, .070, "Target support", "5/class · weight 10", ORANGE_LIGHT, ORANGE, 7.5, 7.0)
    box(ax, .780, .645, .155, .075, "Fit calibration package", "scaler + linear head", "#FFFFFF", ORANGE)
    arrow(ax, .800, .755, .825, .723, BLUE); arrow(ax, .915, .755, .890, .723, ORANGE)
    box(ax, .800, .555, .115, .052, "Freeze package", "", "#FFFFFF", GREY, 7.5)
    arrow(ax, .858, .645, .858, .610, ORANGE)
    ax.plot([.750,.965],[.520,.520], ls="--", lw=.85, color=GREY)
    ax.text(.858, .532, "QUERY FIREWALL", ha="center", fontsize=7.4, fontweight="bold", color=RED)
    box(ax, .770, .425, .175, .060, "Target query measurement", "label hidden", "#FFFFFF", GREY, 7.5, 7.0)
    box(ax, .770, .330, .175, .060, "Same frozen RAMEF extractor", "8 blocks · no update", TEAL_LIGHT, TEAL, 7.4, 7.0)
    box(ax, .770, .235, .175, .060, "Apply frozen calibration", "fixed three-class boundary", "#FFFFFF", GREY, 7.4, 7.0)
    box(ax, .805, .165, .105, .043, "N  /  IR  /  OR", "", ORANGE_LIGHT, ORANGE, 8.0)
    arrow(ax,.858,.425,.858,.393,GREY); arrow(ax,.858,.330,.858,.298,TEAL); arrow(ax,.858,.235,.858,.211,ORANGE)

    # Bottom permissions strip.
    ax.add_patch(Rectangle((.03,.025),.94,.085,facecolor="#FAFBFC",edgecolor=LINE,lw=.75))
    steps=[(.045,.045,.145,"SOURCE TRAINING","816 measurements",BLUE_LIGHT,BLUE),
           (.235,.045,.145,"SOURCE VALIDATION","204 measurements",BLUE_LIGHT,BLUE),
           (.440,.045,.220,"TARGET-SUPPORT CALIBRATION","15 labelled measurements",ORANGE_LIGHT,ORANGE),
           (.720,.045,.145,"QUERY INFERENCE","frozen pipeline","#F0F2F3",GREY)]
    for x,y,w,t,s,f,e in steps: box(ax,x,y,w,.045,t,s,f,e,7.3,6.8)
    for x1,x2 in ((.190,.232),(.380,.437),(.660,.717)): arrow(ax,x1,.067,x2,.067,GREY)
    ax.text(.880,.067,"No query statistics\nNo test-time update",ha="left",va="center",
            fontsize=7.1,fontweight="bold",color=INK)

    for artist in fig.findobj(match=lambda o:isinstance(o,mpl.text.Text)):
        artist.set_fontfamily("Times New Roman")
    fig.savefig(OUT.with_suffix(".svg"),facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"),facecolor="white")
    fig.savefig(OUT.with_suffix(".png"),dpi=360,facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
