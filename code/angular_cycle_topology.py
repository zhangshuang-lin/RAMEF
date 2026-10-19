"""Angle-resampled complex revolution topology for X0-B P0.

Unlike averaged envelope/order features, this module preserves how amplitude
and analytic phase move from one shaft revolution to the next.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


FS = 64_000.0
CARRIER_BANDS = ((500, 1500), (1500, 3000), (3000, 6000),
                 (6000, 10000), (10000, 16000), (16000, 24000))
ORDER_ANCHORS = (1.0, 2.05, 3.05, 4.0, 4.95, 5.95)


def angular_resample(x: np.ndarray, rpm: float, n_cycles: int = 7,
                     samples_per_cycle: int = 256) -> np.ndarray:
    """Centre-crop exactly n_cycles then resample to a fixed angular grid."""
    needed = FS * 60.0 / float(rpm) * n_cycles
    if needed > len(x):
        raise ValueError(f"Need {needed:.1f} samples for {n_cycles} revolutions, got {len(x)}")
    centre = (len(x) - 1) / 2.0
    source = centre + np.linspace(-needed / 2.0, needed / 2.0,
                                  n_cycles * samples_per_cycle, endpoint=False)
    grid = np.arange(len(x), dtype=np.float64)
    if np.iscomplexobj(x):
        values = np.interp(source, grid, x.real) + 1j * np.interp(source, grid, x.imag)
    else:
        values = np.interp(source, grid, x)
    return values.reshape(n_cycles, samples_per_cycle)


def circular_shift_features(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Signed best circular displacement and normalized correlation peak."""
    aa = (a - a.mean()) / (a.std() + 1e-8)
    bb = (b - b.mean()) / (b.std() + 1e-8)
    corr = np.fft.ifft(np.fft.fft(aa) * np.conj(np.fft.fft(bb))).real / len(a)
    index = int(np.argmax(corr))
    signed = index if index <= len(a) // 2 else index - len(a)
    return signed / len(a), float(corr[index])


def _coefficient(cycles: np.ndarray, order: float) -> np.ndarray:
    angle = np.arange(cycles.shape[1], dtype=np.float64) / cycles.shape[1]
    kernel = np.exp(-2j * np.pi * order * angle)
    return (cycles * kernel[None]).mean(1)


def topology_features(analytic_cycles: np.ndarray) -> np.ndarray:
    """Describe revolution topology without collapsing its phase trajectory."""
    envelope = np.abs(analytic_cycles)
    envelope = envelope / (np.median(envelope, axis=1, keepdims=True) + 1e-6)
    centered = envelope - envelope.mean(1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
    normalized = centered / norm
    corr = normalized @ normalized.T
    upper = corr[np.triu_indices(len(corr), 1)]
    adjacent_corr = np.diag(corr, 1)

    shifts, peaks = zip(*(circular_shift_features(envelope[i], envelope[i + 1])
                          for i in range(len(envelope) - 1)))
    shifts = np.asarray(shifts); peaks = np.asarray(peaks)
    shift_phase = 2 * np.pi * shifts

    singular = np.linalg.svd(centered, compute_uv=False)
    singular = singular / (singular.sum() + 1e-8)
    entropy = -(singular * np.log(singular + 1e-8)).sum() / np.log(len(singular))

    amplitude = np.sqrt((np.abs(analytic_cycles) ** 2).mean(1))
    phase_unit = analytic_cycles / (np.abs(analytic_cycles) + 1e-8)
    adjacent_phase_lock = np.abs((phase_unit[1:] * np.conj(phase_unit[:-1])).mean(1))

    features = [
        upper.mean(), upper.std(), adjacent_corr.mean(), adjacent_corr.std(),
        shifts.mean(), shifts.std(), np.abs(shifts).mean(), peaks.mean(), peaks.std(),
        np.abs(np.exp(1j * shift_phase).mean()),
        singular[0], singular[1], singular[2], entropy,
        amplitude.mean(), amplitude.std() / (amplitude.mean() + 1e-8),
        np.corrcoef(np.arange(len(amplitude)), amplitude)[0, 1],
        adjacent_phase_lock.mean(), adjacent_phase_lock.std(),
    ]
    for anchor in ORDER_ANCHORS:
        coeff = _coefficient(analytic_cycles, anchor)
        magnitude = np.abs(coeff)
        step = coeff[1:] * np.conj(coeff[:-1])
        features.extend([
            magnitude.mean(), magnitude.std() / (magnitude.mean() + 1e-8),
            np.abs((coeff / (magnitude + 1e-8)).mean()),
            np.abs((step / (np.abs(step) + 1e-8)).mean()),
            np.angle((step / (np.abs(step) + 1e-8)).mean() + 1e-8),
        ])
    return np.nan_to_num(np.asarray(features, dtype=np.float32))


def averaged_order_features(analytic_cycles: np.ndarray) -> np.ndarray:
    """Cycle-averaged control using the identical filtered observations."""
    envelope = np.abs(analytic_cycles)
    mean_cycle = envelope.mean(0)
    mean_cycle = mean_cycle / (np.median(mean_cycle) + 1e-6)
    orders = np.arange(.5, 12.0001, .25)
    coeff = np.asarray([np.abs(_coefficient(mean_cycle[None], o)[0]) for o in orders])
    coeff = np.log1p(coeff / (np.median(coeff) + 1e-6))
    return np.concatenate((coeff.astype(np.float32), np.asarray([
        envelope.mean(), envelope.std(),
        np.sqrt((np.abs(analytic_cycles) ** 2).mean()),
    ], dtype=np.float32)))


def extract_representations(waveform: np.ndarray, rpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return paired ORDER_AVG and RPT features from one real observation."""
    x = np.asarray(waveform, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-8)
    order_parts, topology_parts = [], []
    for low, high in CARRIER_BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=FS, output="sos")
        filtered = sosfiltfilt(sos, x)
        cycles = angular_resample(hilbert(filtered), rpm)
        order_parts.append(averaged_order_features(cycles))
        topology_parts.append(topology_features(cycles))
    return np.concatenate(order_parts), np.concatenate(topology_parts)
