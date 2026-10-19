"""Full local-data qualification audit for MCC5_RAMEF_CONFIRMATORY.

This program performs no learning and never creates a target prediction. It
checks every CSV byte-for-byte, inventories the actual release, and derives
keyphase speed only for files eligible for the metadata-defined bearing-
localisation task. Run identities are files; windows never cross role borders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_mcc5_ramef_p0 import classify


EXPECTED_COLUMNS = 9
EXPECTED_ROWS = 1_152_000
EXPECTED_FS = 12_800.0
WINDOW = 3072
LOCAL_STARTS_S = tuple(18.0 + index * (WINDOW / EXPECTED_FS) for index in range(8))
DISTRIBUTED_STARTS_S = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)


def parse_mode(path: Path) -> str:
    name = path.parent.name.lower()
    if "speed_circulation" in name:
        return "speed_circulation"
    if "torque_circulation" in name:
        return "torque_circulation"
    return "unknown"


def stream_integrity(path: Path) -> dict:
    digest = hashlib.sha256()
    rows = 0
    first = b""
    tail = b""
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
            rows += chunk.count(b"\n")
            if not first:
                first = chunk.splitlines()[0]
            tail = (tail + chunk)[-4096:]
    lines = tail.splitlines()
    last = lines[-1] if lines else b""
    if last == b"" and len(lines) > 1:
        last = lines[-2]
    first_values = first.decode("utf-8", errors="replace").split(",")
    last_values = last.decode("utf-8", errors="replace").split(",")
    try:
        first_time = float(first_values[0])
        last_time = float(last_values[0])
        numeric_edges = True
    except (ValueError, IndexError):
        first_time = last_time = None
        numeric_edges = False
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": rows,
        "first_columns": len(first_values),
        "last_columns": len(last_values),
        "first_time": first_time,
        "last_time": last_time,
        "numeric_edges": numeric_edges,
        "structure_ok": (
            rows == EXPECTED_ROWS
            and len(first_values) == EXPECTED_COLUMNS
            and len(last_values) == EXPECTED_COLUMNS
            and numeric_edges
            and abs(first_time) < 1e-9
            and abs(last_time - (EXPECTED_ROWS - 1) / EXPECTED_FS) < 1e-6
        ),
    }


def keyphase_audit(path: Path) -> dict:
    data = np.loadtxt(path, delimiter=",", usecols=(0, 1), dtype=np.float64)
    time, key = data[:, 0], data[:, 1]
    threshold = 1.0
    edges = np.flatnonzero((key[1:] > threshold) & (key[:-1] <= threshold)) + 1
    periods = np.diff(time[edges])
    valid = (periods > 60 / 5000) & (periods < 60 / 50)
    rpm = 60.0 / periods[valid]
    rpm_time = time[edges[1:]][valid]

    def block_rpm(starts: tuple[float, ...]) -> list[float | None]:
        output = []
        duration = WINDOW / EXPECTED_FS
        for start in starts:
            values = rpm[(rpm_time >= start) & (rpm_time < start + duration)]
            output.append(float(np.median(values)) if values.size else None)
        return output

    dt = np.diff(time)
    local_rpm = block_rpm(LOCAL_STARTS_S)
    distributed_rpm = block_rpm(DISTRIBUTED_STARTS_S)
    return {
        "loaded_rows": int(data.shape[0]),
        "sample_rate_hz": float(1.0 / np.median(dt)),
        "keyphase_threshold": threshold,
        "valid_period_fraction": float(valid.mean()) if valid.size else 0.0,
        "detected_rotations": int(rpm.size),
        "rpm_min": float(np.min(rpm)) if rpm.size else None,
        "rpm_p05": float(np.quantile(rpm, .05)) if rpm.size else None,
        "rpm_median": float(np.median(rpm)) if rpm.size else None,
        "rpm_p95": float(np.quantile(rpm, .95)) if rpm.size else None,
        "rpm_max": float(np.max(rpm)) if rpm.size else None,
        "local_block_rpm": local_rpm,
        "distributed_block_rpm": distributed_rpm,
        "keyphase_ok": (
            data.shape[0] == EXPECTED_ROWS
            and abs(1.0 / np.median(dt) - EXPECTED_FS) < 1e-3
            and rpm.size >= 500
            and all(value is not None for value in local_rpm + distributed_rpm)
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.root.rglob("*.csv"))
    if not paths:
        raise RuntimeError("No MCC5 CSV files found")

    inventory = []
    for index, path in enumerate(paths, 1):
        metadata = classify(path.name)
        integrity = stream_integrity(path)
        inventory.append({
            "relative_path": str(path.relative_to(args.root)),
            "mode": parse_mode(path),
            **metadata,
            **integrity,
        })
        print(f"integrity {index}/{len(paths)} {path.name} ok={integrity['structure_ok']}", flush=True)
    write_csv(args.output / "full_file_inventory.csv", inventory)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in inventory:
        by_hash[row["sha256"]].append(row["relative_path"])
    exact_duplicates = [members for members in by_hash.values() if len(members) > 1]

    selected = [
        row for row in inventory
        if row["mode"] == "speed_circulation"
        and row["localisation_class"] in {"N", "IR", "OR"}
    ]
    speed_rows = []
    for index, row in enumerate(selected, 1):
        path = args.root / row["relative_path"]
        audit = keyphase_audit(path)
        speed_rows.append({**row, **audit})
        print(f"keyphase {index}/{len(selected)} {path.name} ok={audit['keyphase_ok']}", flush=True)
    # Lists serialize deterministically in JSON but not usefully in CSV.
    csv_rows = []
    for row in speed_rows:
        csv_rows.append({
            **row,
            "local_block_rpm": json.dumps(row["local_block_rpm"]),
            "distributed_block_rpm": json.dumps(row["distributed_block_rpm"]),
        })
    write_csv(args.output / "speed_keyphase_audit.csv", csv_rows)

    pure_counts = Counter(
        row["pure_class"] for row in inventory
        if row["mode"] == "speed_circulation" and row["pure_class"] in {"N", "IR", "OR"}
    )
    localisation_counts = Counter(row["localisation_class"] for row in selected)
    profile_counts = Counter((row["localisation_class"], row["nominal_rpm_tag"]) for row in selected)
    context_counts = Counter(row["bearing_context"] for row in inventory if row["mode"] == "speed_circulation")
    result = {
        "audit": "MCC5_RAMEF_CONFIRMATORY_FULL_DATA_P1",
        "root": str(args.root),
        "model_predictions_used": False,
        "total_csv_files": len(inventory),
        "mode_counts": dict(Counter(row["mode"] for row in inventory)),
        "all_structure_ok": all(row["structure_ok"] for row in inventory),
        "structure_failures": [row["relative_path"] for row in inventory if not row["structure_ok"]],
        "exact_duplicate_groups": exact_duplicates,
        "pure_speed_class_counts": dict(sorted(pure_counts.items())),
        "localisation_speed_class_counts": dict(sorted(localisation_counts.items())),
        "speed_context_counts": dict(sorted(context_counts.items())),
        "profile_counts": {f"{key[0]}_{key[1]}": value for key, value in sorted(profile_counts.items())},
        "selected_speed_runs": len(selected),
        "all_selected_keyphase_ok": all(row["keyphase_ok"] for row in speed_rows),
        "keyphase_failures": [row["relative_path"] for row in speed_rows if not row["keyphase_ok"]],
        "channel_choice": {
            "column_one_based": 4,
            "meaning": "motor drive-end horizontal radial vibration",
            "selection_basis": "dataset article Table 4; frozen without model performance",
        },
        "bearing_orders": {"BPFO": 3.585, "BPFI": 5.415, "BSF": 2.357},
        "block_protocol": {
            "window_samples": WINDOW,
            "duration_seconds": WINDOW / EXPECTED_FS,
            "local_starts_seconds": LOCAL_STARTS_S,
            "distributed_starts_seconds": DISTRIBUTED_STARTS_S,
            "single_local_index_zero_based": 3,
            "equal_m8_samples": 8 * WINDOW,
        },
    }
    result["p1_gate"] = bool(
        result["all_structure_ok"]
        and not result["exact_duplicate_groups"]
        and result["all_selected_keyphase_ok"]
        and all(profile_counts[(cls, rpm)] >= count for cls in ("N", "IR", "OR")
                for rpm, count in ((1000, 10), (2000, 10), (3000, 10)))
    )
    (args.output / "full_audit_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
