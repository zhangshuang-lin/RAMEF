"""Coordinate-preserving carrier/order side-front-end for frozen M2.

The absolute-Hz carrier axis and the rotational-order axis are never treated
as the same image coordinate.  Six named resonance bands are processed
independently; their analytic envelopes provide dense complex order evidence
and four-block modulation/phase statistics.  The two coordinate systems meet
only in the learned token-level pairing module.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _complex_linear_sample(spectrum: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Linearly sample a one-sided complex spectrum at fractional bin positions."""
    if spectrum.ndim != 2 or positions.ndim != 2:
        raise ValueError("spectrum and positions must have shapes [B,F] and [B,K]")
    upper = spectrum.shape[-1] - 1
    pos = positions.clamp(0.0, max(float(upper) - 1e-6, 0.0))
    left = pos.floor().long(); right = (left + 1).clamp_max(upper)
    fraction = (pos - left).to(spectrum.real.dtype)
    a = spectrum.gather(1, left); b = spectrum.gather(1, right)
    return a * (1.0 - fraction) + b * fraction


class CoordinateCarrierOrderDescriptor(nn.Module):
    """Deterministic dual-resolution signal descriptor.

    Outputs retain explicit band and order dimensions:

    - carrier: [B, bands, 4] absolute-Hz spectral descriptors;
    - dense_order: [B, bands, dense_orders, 3] complex envelope-order cues;
    - anchor_order: [B, bands, anchors, 3] exact complex anchor samples;
    - temporal: [B, bands, anchors, 5] four-block modulation/phase cues.
    """

    def __init__(self, sample_rate: float = 64000.0, window_length: int = 16384,
                 carrier_bands: Sequence[Sequence[float]] = (
                     (500., 1500.), (1500., 2500.), (2500., 4000.),
                     (4000., 6000.), (6000., 9000.), (9000., 12000.)),
                 dense_order_min: float = .25, dense_order_max: float = 12.,
                 dense_order_step: float = .25,
                 anchor_orders: Sequence[float] = (.5, 1., 2.05, 3.05, 4., 4.95, 5.95)):
        super().__init__()
        if window_length % 4:
            raise ValueError("window_length must be divisible into four blocks")
        self.sample_rate = float(sample_rate); self.window_length = int(window_length)
        self.block_length = self.window_length // 4
        bands = torch.as_tensor(carrier_bands, dtype=torch.float32)
        if bands.ndim != 2 or bands.shape[1] != 2 or not torch.all(bands[:, 1] > bands[:, 0]):
            raise ValueError("carrier_bands must be increasing [low, high] pairs")
        dense = torch.arange(dense_order_min, dense_order_max + dense_order_step / 2,
                             dense_order_step, dtype=torch.float32)
        self.register_buffer("carrier_bands", bands)
        self.register_buffer("dense_orders", dense)
        self.register_buffer("anchor_orders", torch.as_tensor(anchor_orders, dtype=torch.float32))
        self.register_buffer("full_window", torch.hann_window(self.window_length), persistent=False)
        self.register_buffer("block_window", torch.hann_window(self.block_length), persistent=False)

    @property
    def num_bands(self) -> int:
        return int(self.carrier_bands.shape[0])

    @staticmethod
    def _unit_features(value: torch.Tensor) -> torch.Tensor:
        unit = value / value.abs().clamp_min(1e-7)
        return torch.stack((torch.log1p(value.abs()), unit.real, unit.imag), dim=-1)

    def _carrier_features(self, positive_spectrum: torch.Tensor,
                          positive_hz: torch.Tensor, lo: torch.Tensor,
                          hi: torch.Tensor) -> torch.Tensor:
        mask = (positive_hz >= lo) & (positive_hz < hi)
        power = positive_spectrum.abs().square()[:, mask].clamp_min(1e-12)
        hz = positive_hz[mask]
        total = power.sum(-1).clamp_min(1e-12)
        centroid = (power * hz).sum(-1) / total
        spread = ((power * (hz[None] - centroid[:, None]).square()).sum(-1) / total).sqrt()
        energy = torch.log1p(power.mean(-1))
        flatness = power.log().mean(-1).exp() / power.mean(-1).clamp_min(1e-12)
        return torch.stack((energy, centroid / (self.sample_rate / 2),
                            spread / (hi - lo), flatness), dim=-1)

    def _envelope_for_band(self, full_spectrum: torch.Tensor,
                           full_hz: torch.Tensor, lo: torch.Tensor,
                           hi: torch.Tensor) -> torch.Tensor:
        band = ((full_hz.abs() >= lo) & (full_hz.abs() < hi)).to(full_spectrum.real.dtype)
        n = self.window_length
        hilbert = torch.zeros(n, device=full_spectrum.device, dtype=full_spectrum.real.dtype)
        hilbert[0] = 1.; hilbert[n // 2] = 1.; hilbert[1:n // 2] = 2.
        analytic = torch.fft.ifft(full_spectrum * band[None] * hilbert[None], dim=-1)
        envelope = analytic.abs()
        return envelope - envelope.mean(-1, keepdim=True)

    def _block_statistics(self, envelope: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        b = envelope.shape[0]; blocks = envelope.reshape(b, 4, self.block_length)
        blocks = blocks * self.block_window.to(blocks)
        seconds = torch.arange(self.block_length, device=envelope.device,
                               dtype=envelope.dtype) / self.sample_rate
        target_hz = rpm[:, None].to(envelope) / 60. * self.anchor_orders[None].to(envelope)
        phase = torch.exp((-2j * math.pi) * target_hz[:, :, None] * seconds[None, None])
        amplitude = torch.einsum("btn,ban->bta", blocks.to(torch.complex64), phase.to(torch.complex64))
        amplitude = amplitude / self.block_window.sum().clamp_min(1e-6)
        magnitude = amplitude.abs(); mean_magnitude = magnitude.mean(1)
        coefficient_variation = magnitude.std(1, unbiased=False) / mean_magnitude.clamp_min(1e-6)
        unit = amplitude / magnitude.clamp_min(1e-7)
        phase_stability = unit.mean(1).abs()
        mean_complex = amplitude.mean(1)
        mean_unit = mean_complex / mean_complex.abs().clamp_min(1e-7)
        return torch.stack((torch.log1p(mean_magnitude), coefficient_variation,
                            phase_stability, mean_unit.real, mean_unit.imag), dim=-1)

    def forward(self, waveform: torch.Tensor, rpm: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim == 3:
            waveform = waveform[:, 0]
        if waveform.ndim != 2 or waveform.shape[-1] != self.window_length:
            raise ValueError(f"waveform must have shape [B,{self.window_length}]")
        x = waveform.float(); x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        rpm = rpm.reshape(-1).to(x).clamp_min(1.)
        full_spectrum = torch.fft.fft(x, dim=-1)
        full_hz = torch.fft.fftfreq(self.window_length, 1. / self.sample_rate,
                                    device=x.device, dtype=x.dtype)
        positive_spectrum = torch.fft.rfft(x * self.full_window.to(x), dim=-1)
        positive_hz = torch.fft.rfftfreq(self.window_length, 1. / self.sample_rate,
                                         device=x.device, dtype=x.dtype)
        carrier_rows, dense_rows, anchor_rows, temporal_rows = [], [], [], []
        resolution = self.sample_rate / self.window_length
        for lo, hi in self.carrier_bands.to(x):
            carrier_rows.append(self._carrier_features(positive_spectrum, positive_hz, lo, hi))
            envelope = self._envelope_for_band(full_spectrum, full_hz, lo, hi)
            envelope_spectrum = torch.fft.rfft(envelope * self.full_window.to(envelope), dim=-1)
            dense_hz = rpm[:, None] / 60. * self.dense_orders[None].to(x)
            anchor_hz = rpm[:, None] / 60. * self.anchor_orders[None].to(x)
            dense_rows.append(self._unit_features(
                _complex_linear_sample(envelope_spectrum, dense_hz / resolution)))
            anchor_rows.append(self._unit_features(
                _complex_linear_sample(envelope_spectrum, anchor_hz / resolution)))
            temporal_rows.append(self._block_statistics(envelope, rpm))
        return {
            "carrier": torch.stack(carrier_rows, dim=1),
            "dense_order": torch.stack(dense_rows, dim=1),
            "anchor_order": torch.stack(anchor_rows, dim=1),
            "temporal": torch.stack(temporal_rows, dim=1),
        }


class CarrierOrderTokenAdapter(nn.Module):
    """Pair independent carrier/order encodings and emit bounded M2 residuals."""

    def __init__(self, num_bands: int = 6, num_anchors: int = 7,
                 hidden_dim: int = 32, token_dim: int = 128,
                 max_beta: float = .20):
        super().__init__()
        self.num_bands = int(num_bands); self.num_anchors = int(num_anchors)
        self.token_dim = int(token_dim); self.max_beta = float(max_beta)
        self.carrier_encoder = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.anchor_encoder = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.dense_encoder = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv1d(16, hidden_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64), nn.GELU(), nn.Linear(64, token_dim))
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.residual_projection = nn.Linear(token_dim, token_dim)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        # max_beta * sigmoid(-log(3)) = 0.05 when max_beta=0.20.
        self.gate_logit = nn.Parameter(torch.tensor(-math.log(3.0)))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pair(self, carrier: torch.Tensor, order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        paired = self.pair_encoder(torch.cat((carrier, order, carrier * order), dim=-1))
        weights = torch.softmax(self.attention(torch.cat((carrier, order), dim=-1)), dim=1)
        return (weights * paired).sum(1), weights.squeeze(-1)

    def forward(self, descriptor: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        carrier_raw = descriptor["carrier"]
        dense = descriptor["dense_order"]
        anchor = torch.cat((descriptor["anchor_order"], descriptor["temporal"]), dim=-1)
        b, bands, anchors, _ = anchor.shape
        if bands != self.num_bands or anchors != self.num_anchors:
            raise ValueError("descriptor band/anchor count does not match adapter")
        carrier = self.carrier_encoder(carrier_raw)
        order = self.anchor_encoder(anchor)
        anchor_tokens, attentions = [], []
        for index in range(self.num_anchors):
            token, weight = self._pair(carrier, order[:, :, index])
            anchor_tokens.append(token); attentions.append(weight)
        dense_flat = dense.reshape(b * bands, dense.shape[2], 3).transpose(1, 2)
        dense_encoded = self.dense_encoder(dense_flat).squeeze(-1).reshape(b, bands, -1)
        summary, summary_attention = self._pair(carrier, dense_encoded)
        raw_tokens = torch.stack((*anchor_tokens, summary), dim=1)
        delta = self.residual_projection(raw_tokens)
        beta = self.max_beta * torch.sigmoid(self.gate_logit)
        return {"delta": delta, "beta": beta,
                "attention": torch.stack((*attentions, summary_attention), dim=1),
                "raw_tokens": raw_tokens}

    def fuse(self, base_tokens: torch.Tensor,
             descriptor: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        result = self(descriptor)
        if base_tokens.shape != result["delta"].shape:
            raise ValueError("base tokens and coordinate residual must have identical shapes")
        return base_tokens + result["beta"] * result["delta"], result

