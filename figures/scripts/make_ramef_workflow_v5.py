"""Create a publication-grade RAMEF workflow as an editable vector figure.

The method figure uses one source-domain PU measurement only as an illustrative
signal example. It does not use a target query, fit a model, or report a result.
All signal thumbnails follow the frozen RAMEF preprocessing definitions.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.signal import butter, hilbert, sosfiltfilt, stft


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "figures" / "export"
SRC = ROOT / "paper" / "figures" / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

FIG_BASE = OUT / "Fig1_RAMEF_workflow_v5"
MM = 1 / 25.4
FS = 64_000.0
RPM = 1500.0
WINDOW = 16_384
BANDS = ((500.0, 2500.0), (2500.0, 6000.0), (6000.0, 12_000.0))
ORDERS = {"BPFO": 3.05, "BPFI": 4.95}
MEASUREMENT_ID = "N15_M07_F10_KA22_14"

# Restrained, print-safe palette.
INK = "#20252B"
MUTED = "#5C6773"
PANEL = "#F7F9FB"
PANEL_WARM = "#FBF8F4"
LINE = "#A9B4BE"
BLUE = "#1769AA"
BLUE_MID = "#5AA6D6"
BLUE_LIGHT = "#DDECF6"
ORANGE = "#D66B1F"
ORANGE_MID = "#E9A15D"
ORANGE_LIGHT = "#F8E7D8"
TEAL = "#16867A"
TEAL_LIGHT = "#DDEFEA"
GREY = "#747C84"
GREY_LIGHT = "#EEF1F3"
RED = "#B64A4A"


def configure():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def section(ax, x, y, w, h, title, letter, face=PANEL):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=face, edgecolor=GREY, linewidth=0.8,
        linestyle=(0, (2.2, 1.6)), zorder=0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.022, letter, fontsize=9.5, fontweight="bold",
            ha="left", va="top", color=INK)
    ax.text(x + 0.038, y + h - 0.022, title, fontsize=8.5, fontweight="bold",
            ha="left", va="top", color=INK)


def rounded_box(ax, x, y, w, h, title, subtitle=None, face=GREY_LIGHT,
                edge=GREY, title_size=6.2, subtitle_size=5.1, radius=0.008,
                linewidth=0.75):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=2,
    )
    ax.add_patch(patch)
    ty = y + h * (0.59 if subtitle else 0.50)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=INK, zorder=3)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.25, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=MUTED, zorder=3, linespacing=1.05)
    return patch


def lane(ax, x, y, w, h, title, subtitle, face, edge):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor=face, edgecolor=edge, linewidth=0.65, zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.022, title, ha="left", va="top",
            fontsize=6.1, color=INK, fontweight="bold", zorder=4)
    ax.text(x + 0.012, y + h - 0.047, subtitle, ha="left", va="top",
            fontsize=4.6, color=MUTED, zorder=4)
    return patch


def arrow(ax, start, end, color=MUTED, lw=0.85, style="-", rad=0.0,
          mutation=7.5, zorder=5):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color, linestyle=style,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=zorder,
    ))


def label_pill(ax, x, y, w, h, text, face, edge, color=INK, size=5.4):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.003,rounding_size=0.012",
                          facecolor=face, edgecolor=edge, linewidth=0.65, zorder=3)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=color, fontweight="bold", zorder=4)


def load_source_measurement():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data_loader_pu import load_vibration
    from racp_or import uniform_nonoverlap_starts

    split = json.loads((ROOT / "split_independent_measurement.json").read_text(encoding="utf-8"))
    match = None
    for item in split["windows"]:
        if item["measurement_id"] == MEASUREMENT_ID and item["role"] == "train":
            match = item
            break
    if match is None:
        raise RuntimeError("Registered illustrative source measurement was not found")
    signal = load_vibration(match["path"])
    starts = uniform_nonoverlap_starts(len(signal), count=8, window=WINDOW)
    blocks = np.stack([signal[s:s + WINDOW] for s in starts]).astype(np.float64)
    return signal.astype(np.float64), starts, blocks, match


def soft_harmonic_ratio(envelope, order):
    centered = envelope - envelope.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    freq = np.fft.rfftfreq(len(centered), 1.0 / FS)
    resolution = FS / len(centered)
    values = []
    for harmonic in range(1, 5):
        target = harmonic * order * RPM / 60.0
        sigma = max(2.0 * resolution, 0.08 * target)
        core = np.exp(-0.5 * ((freq - target) / sigma) ** 2)
        neighbourhood = np.exp(-0.5 * ((freq - target) / (4.0 * sigma)) ** 2)
        background = np.clip(neighbourhood - core, 0, None)
        core_power = np.sum(core * power) / max(np.sum(core), 1e-12)
        bg_power = np.sum(background * power) / max(np.sum(background), 1e-12)
        values.append(np.log1p(core_power / max(bg_power, 1e-12)))
    return float(np.mean(values))


def physical_evidence(blocks):
    evidence = np.empty((8, 6), dtype=float)
    envelopes = []
    for band_idx, (low, high) in enumerate(BANDS):
        sos = butter(4, (low / (FS / 2), high / (FS / 2)), btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, blocks, axis=-1)
        env = np.abs(hilbert(filtered, axis=-1))
        if band_idx == 1:
            envelopes = env
        for b in range(8):
            evidence[b, 2 * band_idx] = soft_harmonic_ratio(env[b], ORDERS["BPFO"])
            evidence[b, 2 * band_idx + 1] = soft_harmonic_ratio(env[b], ORDERS["BPFI"])
    return evidence, np.asarray(envelopes)


def prepare_signal_views():
    signal, starts, blocks, meta = load_source_measurement()
    evidence, envelopes = physical_evidence(blocks)

    selected = 3
    block = blocks[selected]
    block_std = (block - block.mean()) / max(block.std(), 1e-8)
    f_stft, t_stft, z = stft(block_std, fs=FS, window="hann", nperseg=1024,
                            noverlap=768, nfft=1024, boundary=None, padded=False)
    keep = f_stft <= 12_000
    stft_db = 20 * np.log10(np.abs(z[keep]) + 1e-5)
    stft_db -= np.nanmax(stft_db)

    env = envelopes[selected] - envelopes[selected].mean()
    freq = np.fft.rfftfreq(len(env), 1.0 / FS)
    order = freq / (RPM / 60.0)
    power = np.log1p(np.abs(np.fft.rfft(env)) ** 2)
    mask = order <= 22
    power = power[mask]
    power = (power - power.min()) / max(power.max() - power.min(), 1e-12)

    # Downsample source traces for audit-friendly CSVs.
    wave_idx = np.linspace(0, len(signal) - 1, 2400).astype(int)
    with (SRC / "Fig1_RAMEF_workflow_v4_waveform.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["time_s", "amplitude"])
        w.writerows(zip(wave_idx / FS, signal[wave_idx]))
    with (SRC / "Fig1_RAMEF_workflow_v4_order_spectrum.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["order", "normalized_log_power"])
        w.writerows(zip(order[mask], power))
    with (SRC / "Fig1_RAMEF_workflow_v4_crossblock_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["block", "band", "evidence_type", "soft_evidence"])
        for b in range(8):
            for band_idx in range(3):
                for kind_idx, kind in enumerate(("BPFO", "BPFI")):
                    w.writerow([b + 1, band_idx + 1, kind, evidence[b, 2 * band_idx + kind_idx]])
    np.savez_compressed(
        SRC / "Fig1_RAMEF_workflow_v4_stft.npz",
        frequency_hz=f_stft[keep], time_s=t_stft, relative_db=stft_db,
    )
    (SRC / "Fig1_RAMEF_workflow_v4_metadata.json").write_text(json.dumps({
        "illustrative_measurement_id": meta["measurement_id"],
        "role": "source training only",
        "class": "OR",
        "rpm": RPM,
        "sample_rate_hz": FS,
        "window_samples": WINDOW,
        "selected_block": selected + 1,
        "block_starts": starts.tolist(),
        "image_adjustment": "global standardisation for STFT thumbnail; no local editing",
    }, indent=2), encoding="utf-8")
    return signal, starts, blocks, f_stft[keep], t_stft, stft_db, order[mask], power, evidence


def draw_u2hz_icon(ax, x, y, w, h):
    ax.text(x + w / 2, y + 0.96 * h, "U2-Hz", ha="center", va="top",
            fontsize=6.0, color=BLUE, fontweight="bold")
    down_x = np.linspace(x + 0.05 * w, x + 0.38 * w, 4)
    down_y = np.linspace(y + 0.66 * h, y + 0.20 * h, 4)
    up_x = np.linspace(x + 0.58 * w, x + 0.84 * w, 3)
    up_y = np.linspace(y + 0.30 * h, y + 0.66 * h, 3)
    for i, (xx, yy) in enumerate(zip(down_x, down_y)):
        ax.add_patch(FancyBboxPatch((xx, yy), 0.18*w, 0.14*h,
                     boxstyle="round,pad=0.002,rounding_size=0.003",
                     facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.55, zorder=3))
        if i:
            arrow(ax, (down_x[i-1]+0.18*w, down_y[i-1]+0.07*h),
                  (xx, yy+0.07*h), color=BLUE, lw=0.5, mutation=4.5)
    for i, (xx, yy) in enumerate(zip(up_x, up_y)):
        ax.add_patch(FancyBboxPatch((xx, yy), 0.16*w, 0.13*h,
                     boxstyle="round,pad=0.002,rounding_size=0.003",
                     facecolor="#EAF3F8", edgecolor=BLUE_MID, linewidth=0.5, zorder=3))
        if i:
            arrow(ax, (up_x[i-1]+0.16*w, up_y[i-1]+0.065*h),
                  (xx, yy+0.065*h), color=BLUE_MID, lw=0.5, mutation=4.5)
    ax.plot([down_x[0]+0.09*w, up_x[-1]+0.08*w],
            [down_y[0]+0.16*h, up_y[-1]+0.16*h], color=BLUE, lw=0.65, zorder=2)
    ax.text(x + w / 2, y + 0.02 * h, "nested RSU encoder–decoder", ha="center",
            va="bottom", fontsize=4.5, color=MUTED)


def draw():
    configure()
    signal, starts, blocks, f_stft, t_stft, stft_db, order, power, evidence = prepare_signal_views()

    fig = plt.figure(figsize=(190 * MM, 142 * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.975, "Rotation-aware measurement evidence fusion (RAMEF)",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.944,
            "Distributed observation, carrier–soft-order evidence construction, and target-support calibration",
            ha="center", va="top", fontsize=6.2, color=MUTED)

    section(ax, 0.018, 0.135, 0.265, 0.775, "Distributed observation", "a")
    section(ax, 0.298, 0.135, 0.425, 0.775, "Carrier–soft-order evidence", "b")
    section(ax, 0.738, 0.135, 0.244, 0.775, "Calibration and inference", "c", PANEL_WARM)

    # Panel a: real waveform and registered distributed positions.
    ax_wave = fig.add_axes([0.043, 0.706, 0.214, 0.122], facecolor="white")
    idx = np.linspace(0, len(signal) - 1, 2400).astype(int)
    ax_wave.plot(idx / FS, signal[idx], color=BLUE, lw=0.45)
    ylim = np.percentile(np.abs(signal[idx]), 99.0)
    ax_wave.set_ylim(-ylim, ylim)
    for b, start in enumerate(starts):
        x0, x1 = start / FS, (start + WINDOW) / FS
        ax_wave.axvspan(x0, x1, color=BLUE_MID, alpha=0.20 if b != 3 else 0.35, lw=0)
        ax_wave.text((x0 + x1) / 2, 0.93 * ylim, str(b + 1), ha="center", va="top",
                     fontsize=4.2, color=INK)
    ax_wave.set_xlim(0, len(signal) / FS)
    ax_wave.set_xlabel("Time (s)", fontsize=5.0, labelpad=1)
    ax_wave.set_ylabel("Amplitude", fontsize=5.0, labelpad=1)
    ax_wave.tick_params(labelsize=4.4, length=2, pad=1)
    ax_wave.set_title("One physical measurement", fontsize=5.8, pad=2, fontweight="bold")

    arrow(ax, (0.150, 0.690), (0.150, 0.648), color=BLUE, lw=0.9)
    rounded_box(ax, 0.050, 0.560, 0.200, 0.082, "Eight distributed blocks",
                "16,384 samples each · 0% overlap", BLUE_LIGHT, BLUE, 6.2, 5.0)

    # Block stack.
    for k in range(8):
        x = 0.051 + k * 0.024
        ax.add_patch(Rectangle((x, 0.488), 0.020, 0.048,
                               facecolor=BLUE_MID if k != 3 else BLUE,
                               edgecolor="white", linewidth=0.45, zorder=3))
        ax.text(x + 0.010, 0.512, f"B{k+1}", ha="center", va="center",
                fontsize=3.7, color="white", fontweight="bold", zorder=4)
    ax.text(0.150, 0.466, "same label · different record positions", ha="center",
            va="top", fontsize=5.0, color=MUTED)
    arrow(ax, (0.150, 0.458), (0.150, 0.413), color=BLUE, lw=0.8)
    label_pill(ax, 0.053, 0.358, 0.194, 0.046,
               "1 measurement = 1 label = 1 shot", "#FFFFFF", BLUE, BLUE, 5.6)
    rounded_box(ax, 0.053, 0.244, 0.194, 0.088, "Rotation budget",
                "B = 8  ·  30.72 nominal rev @ 900 rpm\nblocks extend observation, not label count",
                TEAL_LIGHT, TEAL, 6.1, 4.8)
    ax.text(0.150, 0.190, "Output", ha="center", fontsize=5.0, color=MUTED)
    label_pill(ax, 0.078, 0.153, 0.144, 0.036, "{xᵢ,1, …, xᵢ,8}", GREY_LIGHT, GREY, INK, 5.7)

    # Panel b: lane backgrounds.
    lane(ax, 0.316, 0.548, 0.386, 0.292, "Learned carrier path",
         "Absolute-frequency resonance morphology", "#F4F9FC", BLUE_MID)
    lane(ax, 0.316, 0.218, 0.386, 0.292, "Parameter-free physical path",
         "Geometry-conditioned BPFO/BPFI recurrence", "#FCF7F1", ORANGE_MID)

    # Actual STFT thumbnail.
    ax_stft = fig.add_axes([0.328, 0.606, 0.105, 0.142])
    ax_stft.imshow(stft_db, origin="lower", aspect="auto",
                   extent=[t_stft.min(), t_stft.max(), f_stft.min() / 1000, f_stft.max() / 1000],
                   cmap="cividis", vmin=-55, vmax=0, interpolation="nearest")
    ax_stft.set_xlabel("Time", fontsize=4.5, labelpad=0.5)
    ax_stft.set_ylabel("kHz", fontsize=4.5, labelpad=0.5)
    ax_stft.tick_params(labelsize=3.8, length=1.5, pad=0.5)
    ax_stft.set_title("log-magnitude STFT", fontsize=5.0, pad=1.5)
    for s in ax_stft.spines.values(): s.set_linewidth(0.45)
    arrow(ax, (0.438, 0.677), (0.463, 0.677), color=BLUE, lw=0.75)

    draw_u2hz_icon(ax, 0.450, 0.588, 0.086, 0.155)
    arrow(ax, (0.538, 0.677), (0.548, 0.677), color=BLUE, lw=0.75)
    rounded_box(ax, 0.548, 0.620, 0.056, 0.112, "8 × 128",
                "interface per block\n(not fault orders)", BLUE_LIGHT, BLUE, 5.1, 3.8)
    arrow(ax, (0.604, 0.677), (0.612, 0.677), color=BLUE, lw=0.72)
    label_pill(ax, 0.612, 0.657, 0.038, 0.040, "flatten", "white", BLUE, BLUE, 4.3)
    arrow(ax, (0.650, 0.677), (0.658, 0.677), color=BLUE, lw=0.72)
    label_pill(ax, 0.658, 0.635, 0.040, 0.082, r"$\mathbf{z}_{i,b}$" + "\n1024-D", "white", BLUE, BLUE, 4.9)
    rounded_box(ax, 0.548, 0.555, 0.080, 0.052, "Mean over 8 blocks",
                r"$\bar{\mathbf{z}}_i\in\mathbb{R}^{1024}$", "#EAF3F8", BLUE_MID, 4.7, 4.0)
    arrow(ax, (0.678, 0.635), (0.610, 0.608), color=BLUE, lw=0.70, rad=0.16)

    # Actual envelope-order thumbnail.
    ax_order = fig.add_axes([0.328, 0.270, 0.115, 0.142])
    ax_order.plot(order, power, color=INK, lw=0.65)
    for h in range(1, 5):
        for name, base, color in (("BPFO", ORDERS["BPFO"], ORANGE), ("BPFI", ORDERS["BPFI"], TEAL)):
            target = h * base
            if target > 22: continue
            ax_order.axvspan(target * 0.92, target * 1.08, color=color, alpha=0.11, lw=0)
            ax_order.axvline(target, color=color, lw=0.5, ls="--", alpha=0.85)
    ax_order.set_xlim(0, 22); ax_order.set_ylim(0, 1.05)
    ax_order.set_xlabel("Order", fontsize=4.5, labelpad=0.5)
    ax_order.set_ylabel("Norm. power", fontsize=4.3, labelpad=0.5)
    ax_order.tick_params(labelsize=3.8, length=1.5, pad=0.5)
    ax_order.set_title("envelope-order neighbourhoods", fontsize=5.0, pad=1.5)
    ax_order.text(0.03, 0.90, "BPFO", transform=ax_order.transAxes, color=ORANGE,
                  fontsize=4.3, fontweight="bold")
    ax_order.text(0.03, 0.78, "BPFI", transform=ax_order.transAxes, color=TEAL,
                  fontsize=4.3, fontweight="bold")
    for s in ax_order.spines.values(): s.set_linewidth(0.45)
    arrow(ax, (0.447, 0.341), (0.465, 0.341), color=ORANGE, lw=0.75)

    # Actual cross-block soft-evidence matrix.
    ax_heat = fig.add_axes([0.452, 0.265, 0.090, 0.155])
    im = ax_heat.imshow(evidence, aspect="auto", cmap="YlOrBr", interpolation="nearest")
    ax_heat.set_xticks(range(6))
    ax_heat.set_xticklabels(["O1", "I1", "O2", "I2", "O3", "I3"], fontsize=3.6)
    ax_heat.set_yticks(range(8)); ax_heat.set_yticklabels([f"B{k}" for k in range(1, 9)], fontsize=3.5)
    ax_heat.tick_params(length=0, pad=0.7)
    ax_heat.set_title("soft evidence across blocks", fontsize=5.0, pad=1.5)
    for s in ax_heat.spines.values(): s.set_linewidth(0.45)
    ax.text(0.497, 0.246, "O = BPFO, I = BPFI; 1–3 = bands",
            ha="center", va="top", fontsize=3.8, color=MUTED)
    arrow(ax, (0.544, 0.341), (0.552, 0.392), color=ORANGE, lw=0.70, rad=-0.08)
    rounded_box(ax, 0.552, 0.355, 0.073, 0.074, "Per-band cues",
                "BPFO · BPFI\nkurtosis · log-RMS", ORANGE_LIGHT, ORANGE, 4.7, 4.0)
    arrow(ax, (0.625, 0.392), (0.632, 0.392), color=ORANGE, lw=0.70)
    rounded_box(ax, 0.632, 0.355, 0.064, 0.074, "48-D",
                "mean · median\nSD · occurrence", ORANGE_LIGHT, ORANGE, 5.1, 4.0)
    arrow(ax, (0.542, 0.300), (0.552, 0.300), color=ORANGE, lw=0.70)
    rounded_box(ax, 0.552, 0.270, 0.144, 0.060, "Best BPFO · best BPFI · Δ",
                "five statistics including top-two · 15-D", "#FFF9F3", ORANGE_MID, 4.8, 4.0)
    label_pill(ax, 0.648, 0.438, 0.048, 0.054, "concat\n" + r"$\mathbf{p}_i$ 63-D", "white", ORANGE, ORANGE, 4.5)
    arrow(ax, (0.664, 0.429), (0.672, 0.438), color=ORANGE, lw=0.68, rad=-0.08)
    arrow(ax, (0.624, 0.330), (0.659, 0.438), color=ORANGE, lw=0.68, rad=-0.18)

    # Late fusion links the two measurement-level views only after separate encoding.
    rounded_box(ax, 0.635, 0.500, 0.067, 0.074, "Late fusion",
                r"$\mathbf{r}_i=[\bar{\mathbf{z}}_i;\mathbf{p}_i]$" + "\n1087-D", TEAL_LIGHT, TEAL, 5.4, 4.4)
    arrow(ax, (0.628, 0.566), (0.635, 0.558), color=BLUE, lw=0.72)
    arrow(ax, (0.672, 0.492), (0.672, 0.500), color=ORANGE, lw=0.72)

    # Panel c: calibration lane.
    ax.text(0.860, 0.844, "Calibration (labelled data only)", ha="center",
            va="center", fontsize=5.4, color=MUTED, fontweight="bold")
    rounded_box(ax, 0.755, 0.728, 0.092, 0.086, "Source feature bank",
                "1,020 measurements\nweight = 1", BLUE_LIGHT, BLUE, 4.9, 4.2)
    rounded_box(ax, 0.871, 0.728, 0.092, 0.086, "Target-support features",
                "5 per class\nweight = 10", ORANGE_LIGHT, ORANGE, 4.9, 4.2)
    arrow(ax, (0.801, 0.728), (0.849, 0.665), color=BLUE, lw=0.7, rad=-0.05)
    arrow(ax, (0.917, 0.728), (0.874, 0.665), color=ORANGE, lw=0.7, rad=0.05)
    rounded_box(ax, 0.793, 0.585, 0.136, 0.080, "Fit calibration package",
                "StandardScaler + multinomial linear head", "#F4E8DE", ORANGE, 5.5, 4.1)
    arrow(ax, (0.861, 0.585), (0.861, 0.538), color=ORANGE, lw=0.8)
    label_pill(ax, 0.803, 0.494, 0.116, 0.040, "freeze scaler + head", "white", GREY, GREY, 5.1)

    # Query firewall.
    ax.plot([0.755, 0.963], [0.465, 0.465], color=GREY, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.859, 0.474, "QUERY FIREWALL", ha="center", va="bottom",
            fontsize=4.7, color=RED, fontweight="bold")
    ax.text(0.859, 0.440, "Inference (one measurement at a time)", ha="center",
            fontsize=5.3, color=MUTED, fontweight="bold")
    rounded_box(ax, 0.786, 0.370, 0.146, 0.058, "Target query measurement",
                "label hidden during inference", GREY_LIGHT, GREY, 5.3, 4.1)
    arrow(ax, (0.859, 0.370), (0.859, 0.333), color=GREY, lw=0.7)
    rounded_box(ax, 0.786, 0.270, 0.146, 0.063, "Same frozen RAMEF extractor",
                "8 blocks · no parameter update", TEAL_LIGHT, TEAL, 5.2, 4.1)
    arrow(ax, (0.859, 0.270), (0.859, 0.233), color=TEAL, lw=0.7)
    rounded_box(ax, 0.786, 0.175, 0.146, 0.058, "Apply frozen calibration",
                "fixed three-class boundary", GREY_LIGHT, GREY, 5.2, 4.1)
    arrow(ax, (0.859, 0.175), (0.859, 0.172), color=ORANGE, lw=0.7, mutation=5.5)
    label_pill(ax, 0.804, 0.137, 0.110, 0.035, "N   /   IR   /   OR", ORANGE_LIGHT, ORANGE, ORANGE, 5.8)

    # Bottom permissions strip.
    ax.add_patch(Rectangle((0.025, 0.025), 0.950, 0.080, facecolor="#FAFAFA",
                           edgecolor=LINE, linewidth=0.65))
    label_pill(ax, 0.036, 0.043, 0.145, 0.036, "SOURCE TRAINING\n816 measurements",
               BLUE_LIGHT, BLUE, BLUE, 4.3)
    arrow(ax, (0.184, 0.061), (0.214, 0.061), color=GREY, lw=0.7)
    ax.text(0.199, 0.090, "validate epochs", ha="center", fontsize=3.7, color=MUTED)
    label_pill(ax, 0.218, 0.043, 0.150, 0.036, "SOURCE VALIDATION\n204 measurements",
               "#EAF3F8", BLUE_MID, BLUE, 4.3)
    arrow(ax, (0.372, 0.061), (0.404, 0.061), color=GREY, lw=0.7)
    ax.text(0.388, 0.090, "select + freeze encoder", ha="center", fontsize=3.6, color=MUTED)
    label_pill(ax, 0.408, 0.043, 0.185, 0.036, "TARGET-SUPPORT CALIBRATION",
               ORANGE_LIGHT, ORANGE, ORANGE, 4.7)
    arrow(ax, (0.597, 0.061), (0.627, 0.061), color=GREY, lw=0.7)
    ax.text(0.612, 0.090, "freeze package", ha="center", fontsize=3.7, color=MUTED)
    label_pill(ax, 0.631, 0.043, 0.140, 0.036, "QUERY INFERENCE",
               GREY_LIGHT, GREY, GREY, 4.7)
    ax.text(0.785, 0.061, "No query statistics · no test-time update\nNo rejection",
            ha="left", va="center", fontsize=3.7, color=INK, fontweight="bold", linespacing=1.05)

    for artist in fig.findobj(match=lambda obj: isinstance(obj, mpl.text.Text)):
        artist.set_fontfamily("Times New Roman")
        if artist.get_fontsize() < 7.2:
            artist.set_fontsize(7.2)
    fig.savefig(FIG_BASE.with_suffix(".svg"), facecolor="white")
    fig.savefig(FIG_BASE.with_suffix(".pdf"), facecolor="white")
    fig.savefig(FIG_BASE.with_suffix(".png"), dpi=360, facecolor="white")
    plt.close(fig)

    print(FIG_BASE.with_suffix(".svg"))
    print(FIG_BASE.with_suffix(".pdf"))


if __name__ == "__main__":
    draw()
