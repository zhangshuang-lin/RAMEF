"""Utilities for the SQ continuous-variable-speed bearing dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.signal import medfilt


STATE_PATTERN = re.compile(r"^(NC|IF_[123]|OF_[123])$")
REC_PATTERN = re.compile(r"^(REC\d+)_ch([23])\.txt$")
LABEL = {"NC": 0, "IF_1": 1, "IF_2": 1, "IF_3": 1,
         "OF_1": 2, "OF_2": 2, "OF_3": 2}
MACRO = {0: "N", 1: "IR", 2: "OR"}


@dataclass(frozen=True)
class SQRecord:
    state: str
    label: int
    subtype: str
    rec_id: str
    vibration_path: Path
    pulse_path: Path
    sample_rate: float
    block_size: int
    header_rows: int


def parse_header(path: Path) -> tuple[dict[str, str], int]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for index in range(64):
            line = stream.readline()
            if not line:
                raise ValueError(f"No data header terminator in {path}")
            stripped = line.strip()
            if stripped.startswith("X:Time"):
                return metadata, index + 1
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                metadata[key.strip()] = value.strip()
    raise ValueError(f"Header exceeds 64 rows in {path}")


def scan_sq(root: Path) -> list[SQRecord]:
    records: list[SQRecord] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not STATE_PATTERN.match(directory.name):
            continue
        by_id: dict[str, dict[int, Path]] = {}
        for path in directory.glob("REC*_ch*.txt"):
            match = REC_PATTERN.match(path.name)
            if match:
                by_id.setdefault(match.group(1), {})[int(match.group(2))] = path
        for rec_id, channels in sorted(by_id.items()):
            if set(channels) != {2, 3}:
                raise RuntimeError(f"Unpaired SQ channels: {directory.name}/{rec_id}")
            h2, rows2 = parse_header(channels[2]); h3, rows3 = parse_header(channels[3])
            fs2, fs3 = float(h2["Sample Rate"]), float(h3["Sample Rate"])
            n2, n3 = int(h2["Block Size"]), int(h3["Block Size"])
            if fs2 != fs3 or n2 != n3 or rows2 != rows3:
                raise RuntimeError(f"Header mismatch: {directory.name}/{rec_id}")
            records.append(SQRecord(
                state=directory.name, label=LABEL[directory.name], subtype=directory.name,
                rec_id=rec_id, vibration_path=channels[2], pulse_path=channels[3],
                sample_rate=fs2, block_size=n2, header_rows=rows2,
            ))
    return records


def load_txt_channel(path: Path, header_rows: int, expected_size: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter="\t", skiprows=header_rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2 or len(data) != expected_size:
        raise RuntimeError(f"Unexpected channel shape {data.shape} for {path}")
    return data[:, 0], data[:, 1]


def extract_speed(pulse: np.ndarray, sample_rate: float,
                  drop_threshold: float = -1.7, level_threshold: float = -4.0,
                  minimum_gap: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.diff(pulse)
    candidates = np.flatnonzero((delta < drop_threshold) & (pulse[1:] < level_threshold)) + 1
    accepted: list[int] = []
    last = -10**12
    for index in candidates.tolist():
        if index - last > minimum_gap:
            accepted.append(index); last = index
    edges = np.asarray(accepted, dtype=np.int64)
    if len(edges) < 12:
        return edges, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    intervals = np.diff(edges) / float(sample_rate)
    rpm = 60.0 / intervals
    sample_index = edges[1:]
    valid = np.isfinite(rpm) & (rpm >= 20.0) & (rpm <= 3600.0)
    return edges, sample_index[valid], rpm[valid]


def interpolated_speed(sample_count: int, indices: np.ndarray, rpm: np.ndarray,
                       sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 5:
        return np.full(sample_count, np.nan, dtype=np.float32), np.full(sample_count, np.nan, dtype=np.float32)
    maximum_odd = len(rpm) if len(rpm) % 2 else len(rpm) - 1
    smooth = medfilt(rpm, kernel_size=max(3, min(11, maximum_odd)))
    times = indices / sample_rate
    slope = np.gradient(smooth, times)
    grid = np.arange(sample_count, dtype=np.float64)
    speed_grid = np.interp(grid, indices, smooth, left=np.nan, right=np.nan).astype(np.float32)
    slope_grid = np.interp(grid, indices, slope, left=np.nan, right=np.nan).astype(np.float32)
    return speed_grid, slope_grid


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def registered_block_starts(mask: np.ndarray, window: int, count: int = 8) -> dict:
    runs = contiguous_runs(mask)
    pool: list[int] = []
    for start, end in runs:
        pool.extend(range(start, end - window + 1, window))
    distributed: list[int] = []
    if len(pool) >= count:
        selected = np.rint(np.linspace(0, len(pool) - 1, count)).astype(int)
        distributed = [int(pool[index]) for index in selected]
    longest = max(runs, key=lambda item: item[1] - item[0]) if runs else (0, 0)
    local: list[int] = []
    if longest[1] - longest[0] >= count * window:
        margin = (longest[1] - longest[0] - count * window) // 2
        first = longest[0] + margin
        local = [int(first + index * window) for index in range(count)]
    return {
        "eligible_samples": int(mask.sum()),
        "run_count": len(runs),
        "longest_run_samples": int(longest[1] - longest[0]),
        "candidate_nonoverlap_blocks": len(pool),
        "distributed_starts": distributed,
        "local_starts": local,
    }


def fixed_rotation_segments(edges: np.ndarray, speed: np.ndarray, rotations: int = 4) -> list[dict]:
    segments = []
    if len(edges) <= rotations:
        return segments
    for index in range(0, len(edges) - rotations, rotations):
        start, end = int(edges[index]), int(edges[index + rotations])
        if end <= start:
            continue
        local = speed[max(0, index - 1):min(len(speed), index + rotations)]
        segments.append({"start": start, "end": end,
                         "samples": end - start,
                         "mean_rpm": float(np.mean(local)) if len(local) else None})
    return segments
