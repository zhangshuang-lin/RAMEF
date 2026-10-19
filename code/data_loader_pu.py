"""Leakage-safe loader for official and simplified PU MAT layouts."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

CONDITIONS = {
    "D0": {"name": "N15_M07_F10", "rpm": 1500., "torque": .7, "radial_force": 1000.},
    "DS": {"name": "N09_M07_F10", "rpm": 900., "torque": .7, "radial_force": 1000.},
    "DM": {"name": "N15_M01_F10", "rpm": 1500., "torque": .1, "radial_force": 1000.},
    "DF": {"name": "N15_M07_F04", "rpm": 1500., "torque": .7, "radial_force": 400.},
}
NAME_TO_DOMAIN = {v["name"]: k for k, v in CONDITIONS.items()}
REAL_OUTER_CODES = {"KA04", "KA15", "KA16", "KA22", "KA30"}
REAL_INNER_CODES = {"KI04", "KI14", "KI16", "KI17", "KI18", "KI21"}


def resolve_split_path(cfg: dict, target: str = "DS", shot: int = 5, seed: int = 0) -> Path:
    """Resolve the immutable protocol selected in config.yaml."""
    split_type = cfg["data"].get("split_type", "original")
    if split_type == "independent":
        return Path(cfg["data"].get("independent_split", "split_independent_measurement.json"))
    if split_type == "original":
        return Path(cfg["data"]["splits"]) / f"{target}_{shot}shot_seed{seed:02d}.json"
    raise ValueError(f"Unknown split_type={split_type!r}; expected original or independent")


def infer_default_label(bearing_code: str) -> Optional[int]:
    """Conservative official-directory fallback: K=N, KI=I, KA=O.

    PU folder names alone do not provide a trustworthy compound IO mapping. A
    labels.csv is therefore required for four-class experiments.
    """
    code = bearing_code.upper()
    if re.fullmatch(r"K00[1-6]", code): return 0
    if code in REAL_INNER_CODES: return 1
    if code in REAL_OUTER_CODES: return 2
    return None


def _numeric_arrays(obj) -> Iterable[np.ndarray]:
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number): yield np.asarray(obj).squeeze()
        elif obj.dtype == object:
            for value in obj.flat: yield from _numeric_arrays(value)
    elif hasattr(obj, "_fieldnames"):
        for name in obj._fieldnames: yield from _numeric_arrays(getattr(obj, name))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if not key.startswith("__"): yield from _numeric_arrays(value)


def load_vibration(path: str | Path, preferred_key: str = "signal") -> np.ndarray:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if preferred_key in mat:
        signal = np.asarray(mat[preferred_key]).squeeze()
    else:
        # Official PU/KAt files store channels in root.Y. Channel Y[6] is the
        # 64-kHz vibration signal; Y[1:3] are motor-current channels and must
        # not be selected merely because they have the same length.
        roots = [value for key, value in mat.items() if not key.startswith("__")]
        official = roots[0] if len(roots) == 1 else None
        if official is not None and hasattr(official, "Y") and len(np.atleast_1d(official.Y)) >= 7:
            signal = np.asarray(np.atleast_1d(official.Y)[6].Data).squeeze()
            return np.asarray(signal, dtype=np.float32)
        candidates = [a for a in _numeric_arrays(mat) if a.ndim == 1 and a.size >= 8192]
        if not candidates:
            raise ValueError(f"No one-dimensional vibration array found in {path}")
        # Official PU files contain several channels. The longest high-rate channel
        # is selected; labels.csv may specify signal_key for unusual exports.
        signal = max(candidates, key=lambda a: a.size)
    return np.asarray(signal, dtype=np.float32)


class PUWindowDataset(Dataset):
    def __init__(self, split_json: str | Path, role: str, normalize: bool = True,
                 cache_waveforms: bool = False, long_window_length: int | None = None):
        payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
        self.items = [x for x in payload["windows"] if x["role"] == role]
        self.normalize = normalize
        self.cache_waveforms = cache_waveforms
        self.long_window_length = long_window_length
        self._cache = {}
        if not self.items: raise ValueError(f"No windows with role={role} in {split_json}")

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        if self.cache_waveforms and index in self._cache:
            x, long_x = self._cache[index]
            result = {"waveform": x, "label": torch.tensor(item["label"]),
                    "rpm": torch.tensor(item["rpm"], dtype=torch.float32),
                    "domain": torch.tensor(item["domain_index"]), "meta": item}
            if long_x is not None: result["long_waveform"] = long_x
            return result
        signal = load_vibration(item["path"], item.get("signal_key", "signal"))
        x = signal[item["start"]:item["end"]].copy()
        if len(x) != item["end"] - item["start"]: raise IndexError(f"Invalid window: {item}")
        if self.normalize:
            x = (x - x.mean()) / max(float(x.std()), 1e-6)
        long_x = None
        if self.long_window_length:
            center = (item["start"] + item["end"]) // 2
            long_start = max(0, min(center - self.long_window_length // 2,
                                    len(signal) - self.long_window_length))
            long_np = signal[long_start:long_start + self.long_window_length].copy()
            if len(long_np) != self.long_window_length:
                raise IndexError(f"Invalid long window for {item['measurement_id']}")
            if self.normalize:
                long_np = (long_np - long_np.mean()) / max(float(long_np.std()), 1e-6)
            long_x = torch.from_numpy(long_np)
        x = torch.from_numpy(x)
        if self.cache_waveforms: self._cache[index] = (x, long_x)
        result = {"waveform": x, "label": torch.tensor(item["label"]),
                "rpm": torch.tensor(item["rpm"], dtype=torch.float32),
                "domain": torch.tensor(item["domain_index"]), "meta": item}
        if long_x is not None: result["long_waveform"] = long_x
        return result


def collate_pu(batch):
    result = {"waveform": torch.stack([x["waveform"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "rpm": torch.stack([x["rpm"] for x in batch]),
            "domain": torch.stack([x["domain"] for x in batch]),
            "meta": [x["meta"] for x in batch]}
    if "long_waveform" in batch[0]:
        result["long_waveform"] = torch.stack([x["long_waveform"] for x in batch])
    return result
