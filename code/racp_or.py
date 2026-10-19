"""Rotation-aware cross-block persistence descriptors.

The descriptors deliberately preserve raw sensor amplitude until a local
spectral-background ratio is formed.  This differs from A1's per-image
standardisation and lets the audit test whether weak outer-race impacts recur
across independent rotations and resonance bands.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis


WINDOW = 16384
BANDS = ((500.0, 2500.0), (2500.0, 6000.0), (6000.0, 12000.0))
BLOCK4 = (0, 2, 5, 7)


def uniform_nonoverlap_starts(signal_length: int, count: int = 8,
                              window: int = WINDOW) -> np.ndarray:
    """Uniform starts with an explicit non-overlap invariant."""
    if signal_length < count * window:
        raise ValueError("Measurement is too short for the registered blocks")
    starts = np.rint(np.linspace(0, signal_length - window, count)).astype(np.int64)
    if np.any(np.diff(starts) < window):
        raise RuntimeError("Registered measurement blocks overlap")
    return starts


def _soft_harmonic_ratio(envelope: np.ndarray, sample_rate: float, rpm: float,
                         order: float) -> float:
    centred = envelope - envelope.mean()
    power = np.abs(np.fft.rfft(centred)) ** 2
    frequency = np.fft.rfftfreq(len(centred), 1.0 / sample_rate)
    resolution = sample_rate / len(centred)
    values = []
    for harmonic in range(1, 5):
        target = harmonic * order * rpm / 60.0
        sigma = max(2.0 * resolution, 0.08 * target)
        core = np.exp(-0.5 * ((frequency - target) / sigma) ** 2)
        neighbourhood = np.exp(-0.5 * ((frequency - target) / (4.0 * sigma)) ** 2)
        # Remove the core from its local background so a broad resonance hump
        # cannot masquerade as a kinematically repeated impact train.
        background = np.clip(neighbourhood - core, 0.0, None)
        core_power = float(np.sum(core * power) / max(np.sum(core), 1e-12))
        bg_power = float(np.sum(background * power) / max(np.sum(background), 1e-12))
        values.append(np.log1p(core_power / max(bg_power, 1e-12)))
    return float(np.mean(values))


def measurement_descriptor(raw_blocks: np.ndarray, rpm: float,
                           sample_rate: float = 64000.0) -> np.ndarray:
    """Return fixed multi-band, cross-block physical evidence."""
    if raw_blocks.ndim != 2 or raw_blocks.shape[1] != WINDOW:
        raise ValueError("raw_blocks must have shape [blocks, 16384]")
    per_band = []
    nyquist = sample_rate / 2.0
    for low, high in BANDS:
        sos = butter(4, (low / nyquist, high / nyquist), btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, raw_blocks, axis=-1)
        envelopes = np.abs(hilbert(filtered, axis=-1))
        rows = []
        for block, envelope in zip(filtered, envelopes):
            rows.append([
                _soft_harmonic_ratio(envelope, sample_rate, rpm, 3.05),
                _soft_harmonic_ratio(envelope, sample_rate, rpm, 4.95),
                float(kurtosis(block, fisher=True, bias=False)),
                float(np.log1p(np.sqrt(np.mean(block ** 2)))),
            ])
        per_band.append(np.asarray(rows, dtype=np.float64))

    # [band, block, {BPFO,BPFI,kurtosis,log-rms}]
    evidence = np.stack(per_band)
    features = []
    for band in evidence:
        for column in range(band.shape[1]):
            values = band[:, column]
            features.extend([
                float(values.mean()), float(np.median(values)),
                float(values.std()), float(np.mean(values > 0.5)),
            ])
    # Cross-band best evidence is retained only when it recurs across blocks.
    best_or = evidence[:, :, 0].max(axis=0)
    best_ir = evidence[:, :, 1].max(axis=0)
    for values in (best_or, best_ir, best_or - best_ir):
        features.extend([
            float(values.mean()), float(np.median(values)),
            float(values.std()), float(np.mean(values > 0.5)),
            float(np.mean(np.sort(values)[-2:])),
        ])
    output = np.asarray(features, dtype=np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite RACP descriptor")
    return output
