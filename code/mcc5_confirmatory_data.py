"""Data preparation and fixed RAMEF-F descriptor for MCC5 confirmation.

Target-query vibration is materialised only when ``--only-query`` is invoked
after the runner has frozen every source-selected checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis


FS = 12_800.0
WINDOW = 3072
LOCAL_STARTS_S = tuple(18.0 + index * (WINDOW / FS) for index in range(8))
DISTRIBUTED_STARTS_S = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
BANDS = ((300.0, 1500.0), (1500.0, 3000.0), (3000.0, 5500.0))
BPFO = 3.585
BPFI = 5.415
LABELS = {"N": 0, "IR": 1, "OR": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def soft_harmonic_ratio(envelope: np.ndarray, rpm: float, order: float) -> float:
    centred = envelope - envelope.mean()
    power = np.abs(np.fft.rfft(centred)) ** 2
    frequency = np.fft.rfftfreq(len(centred), 1.0 / FS)
    resolution = FS / len(centred)
    values = []
    for harmonic in range(1, 5):
        target = harmonic * order * rpm / 60.0
        sigma = max(2.0 * resolution, 0.08 * target)
        core = np.exp(-0.5 * ((frequency - target) / sigma) ** 2)
        neighbourhood = np.exp(-0.5 * ((frequency - target) / (4.0 * sigma)) ** 2)
        background = np.clip(neighbourhood - core, 0.0, None)
        core_power = float(np.sum(core * power) / max(np.sum(core), 1e-12))
        bg_power = float(np.sum(background * power) / max(np.sum(background), 1e-12))
        values.append(np.log1p(core_power / max(bg_power, 1e-12)))
    return float(np.mean(values))


def physical_descriptor(blocks: np.ndarray, rpms: np.ndarray) -> np.ndarray:
    if blocks.shape != (8, WINDOW) or rpms.shape != (8,):
        raise ValueError("Expected eight 3072-sample blocks and eight block RPMs")
    evidence = []
    for low, high in BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=FS, output="sos")
        filtered = sosfiltfilt(sos, blocks, axis=-1)
        envelopes = np.abs(hilbert(filtered, axis=-1))
        rows = []
        for block, envelope, rpm in zip(filtered, envelopes, rpms):
            rows.append([
                soft_harmonic_ratio(envelope, float(rpm), BPFO),
                soft_harmonic_ratio(envelope, float(rpm), BPFI),
                float(kurtosis(block, fisher=True, bias=False)),
                float(np.log1p(np.sqrt(np.mean(block ** 2)))),
            ])
        evidence.append(rows)
    evidence = np.asarray(evidence, dtype=np.float64)  # band, block, cue
    features = []
    for band in evidence:
        for column in range(band.shape[1]):
            values = band[:, column]
            features.extend([values.mean(), np.median(values), values.std(), np.mean(values > .5)])
    best_or = evidence[:, :, 0].max(axis=0)
    best_ir = evidence[:, :, 1].max(axis=0)
    for values in (best_or, best_ir, best_or - best_ir):
        features.extend([values.mean(), np.median(values), values.std(),
                         np.mean(values > .5), np.mean(np.sort(values)[-2:])])
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (63,) or not np.isfinite(result).all():
        raise RuntimeError("Invalid MCC5 RAMEF-F descriptor")
    return result


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_cache(root: Path, manifest: Path, keyphase_audit: Path,
                output: Path, only_query: bool) -> None:
    rows = read_rows(manifest)
    allowed = {"target_query"} if only_query else {"source", "source_validation", "target_support"}
    rows = [row for row in rows if row["role"] in allowed]
    audit = {row["relative_path"]: row for row in read_rows(keyphase_audit)}
    local_starts = np.rint(np.asarray(LOCAL_STARTS_S) * FS).astype(np.int64)
    dist_starts = np.rint(np.asarray(DISTRIBUTED_STARTS_S) * FS).astype(np.int64)
    payload = {key: [] for key in ("ids", "roles", "labels", "contexts", "severities",
                                    "torques", "local", "distributed", "rpm_local",
                                    "rpm_distributed", "physical")}
    for index, row in enumerate(rows, 1):
        relative = row["archive_path"]
        path = root.joinpath(*relative.replace("\\", "/").split("/"))
        if relative not in audit:
            raise RuntimeError(f"No keyphase audit for {relative}")
        signal = np.loadtxt(path, delimiter=",", usecols=(3,), dtype=np.float32)
        local = np.stack([signal[start:start + WINDOW] for start in local_starts])
        distributed = np.stack([signal[start:start + WINDOW] for start in dist_starts])
        rpm_local = np.asarray(json.loads(audit[relative]["local_block_rpm"]), dtype=np.float32)
        rpm_dist = np.asarray(json.loads(audit[relative]["distributed_block_rpm"]), dtype=np.float32)
        if local.shape != (8, WINDOW) or distributed.shape != (8, WINDOW):
            raise RuntimeError(f"Block extraction failed for {relative}")
        payload["ids"].append(relative)
        payload["roles"].append(row["role"])
        payload["labels"].append(LABELS[row["localisation_class"]])
        payload["contexts"].append(row["bearing_context"])
        payload["severities"].append(row["severity"])
        payload["torques"].append(int(row["nominal_torque_nm"]))
        payload["local"].append(local)
        payload["distributed"].append(distributed)
        payload["rpm_local"].append(rpm_local)
        payload["rpm_distributed"].append(rpm_dist)
        payload["physical"].append(physical_descriptor(distributed, rpm_dist))
        print(f"cache {'query' if only_query else 'development'} {index}/{len(rows)} {path.name}", flush=True)
    arrays = {
        "ids": np.asarray(payload["ids"]),
        "roles": np.asarray(payload["roles"]),
        "labels": np.asarray(payload["labels"], dtype=np.int64),
        "contexts": np.asarray(payload["contexts"]),
        "severities": np.asarray(payload["severities"]),
        "torques": np.asarray(payload["torques"], dtype=np.int64),
        "local": np.asarray(payload["local"], dtype=np.float32),
        "distributed": np.asarray(payload["distributed"], dtype=np.float32),
        "rpm_local": np.asarray(payload["rpm_local"], dtype=np.float32),
        "rpm_distributed": np.asarray(payload["rpm_distributed"], dtype=np.float32),
        "physical": np.asarray(payload["physical"], dtype=np.float32),
        "manifest_sha256": np.asarray(sha256(manifest)),
        "query_cache": np.asarray(bool(only_query)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps({"output": str(output), "rows": len(rows), "query": only_query,
                      "sha256": sha256(output)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keyphase-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-query", action="store_true")
    args = parser.parse_args()
    build_cache(args.root, args.manifest, args.keyphase_audit, args.output, args.only_query)


if __name__ == "__main__":
    main()
