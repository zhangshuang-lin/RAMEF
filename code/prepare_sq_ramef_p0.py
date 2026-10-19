"""SQ_RAMEF_STRICT_V1 data/speed/split prerequisite audit."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bottleneck_deconvolution import sha256
from sq_data import (MACRO, SQRecord, extract_speed, fixed_rotation_segments,
                     interpolated_speed, load_txt_channel,
                     registered_block_starts, scan_sq)


REGIONS = {
    "low_300_1000": (300.0, 1000.0, None),
    "low_mid_300_2000": (300.0, 2000.0, None),
    "transient_300_2800": (300.0, 2800.0, None),
    "middle_1000_2000": (1000.0, 2000.0, None),
    "high_2000_2800": (2000.0, 2800.0, None),
    "plateau_source": (2000.0, 3400.0, None),
}
PRIMARY_PRIORITY = ("low_300_1000", "low_mid_300_2000", "transient_300_2800")
SEED = 20260824


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def speed_mask(speed: np.ndarray, slope: np.ndarray,
               low: float, high: float, slope_limit: float | None) -> np.ndarray:
    mask = np.isfinite(speed) & (speed >= low) & (speed < high)
    if slope_limit is not None:
        mask &= np.isfinite(slope) & (np.abs(slope) <= slope_limit)
    return mask


def subtype_balanced_query(records: list[dict], label: int, count: int,
                           rng: np.random.Generator) -> list[str]:
    by_subtype: dict[str, list[str]] = defaultdict(list)
    for row in records:
        if row["label"] == label:
            by_subtype[row["subtype"]].append(row["rec_id"])
    for values in by_subtype.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(by_subtype)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if by_subtype[key] and len(selected) < count:
                selected.append(by_subtype[key].pop()); progressed = True
        if not progressed:
            break
    if len(selected) < count:
        raise RuntimeError(f"Insufficient subtype-balanced query records for label {label}")
    return selected


def choose_support(candidates: list[dict], label: int, query_id: str,
                   fold: int, count: int = 5) -> list[str]:
    local = [row for row in candidates if row["label"] == label and row["rec_id"] != query_id]
    rng = np.random.default_rng(SEED + fold * 17 + label)
    by_subtype: dict[str, list[str]] = defaultdict(list)
    for row in local:
        by_subtype[row["subtype"]].append(row["rec_id"])
    for values in by_subtype.values():
        rng.shuffle(values)
    chosen = []
    for subtype in sorted(by_subtype):
        if by_subtype[subtype] and len(chosen) < count:
            chosen.append(by_subtype[subtype].pop())
    remainder = [value for values in by_subtype.values() for value in values]
    rng.shuffle(remainder)
    chosen.extend(remainder[:count - len(chosen)])
    if len(chosen) != count or len(set(chosen)) != count:
        raise RuntimeError(f"Invalid support selection for fold {fold}, label {label}")
    return chosen


def build_folds(metadata: list[dict], primary_region: str,
                fold_count: int) -> list[dict]:
    target = [row for row in metadata
              if row["full_profile"] and row[f"{primary_region}_local"] == 8
              and row[f"{primary_region}_distributed"] == 8]
    source_eligible = {row["rec_id"] for row in metadata
                       if row["pulse_valid"] and row["plateau_source_distributed"] == 8}
    n_query = sorted(row["rec_id"] for row in target if row["label"] == 0)
    if len(n_query) < fold_count:
        raise RuntimeError(f"Only {len(n_query)} target-eligible NC records; {fold_count} folds required")
    n_query = n_query[:fold_count]
    rng = np.random.default_rng(SEED)
    ir_query = subtype_balanced_query(target, 1, fold_count, rng)
    or_query = subtype_balanced_query(target, 2, fold_count, rng)
    lookup = {row["rec_id"]: row for row in metadata}
    folds = []
    for fold in range(fold_count):
        query = {0: n_query[fold], 1: ir_query[fold], 2: or_query[fold]}
        support = {label: choose_support(target, label, query[label], fold)
                   for label in (0, 1, 2)}
        query_ids = set(query.values())
        support_ids = {value for values in support.values() for value in values}
        source_ids = sorted(source_eligible - query_ids - support_ids)
        roles = {
            "source": source_ids,
            "support": sorted(support_ids),
            "query": sorted(query_ids),
        }
        if roles["source"] and (set(roles["source"]) & set(roles["support"])
                                or set(roles["source"]) & set(roles["query"])
                                or set(roles["support"]) & set(roles["query"])):
            raise RuntimeError("Fold leakage")
        source_counts = Counter(lookup[rec_id]["label"] for rec_id in source_ids)
        if any(source_counts[label] < 1 for label in (0, 1, 2)):
            raise RuntimeError(f"Fold {fold} lacks a source class: {source_counts}")
        folds.append({
            "fold": fold,
            "query_by_class": {MACRO[label]: query[label] for label in (0, 1, 2)},
            "support_by_class": {MACRO[label]: support[label] for label in (0, 1, 2)},
            "roles": roles,
            "source_class_counts": {MACRO[label]: source_counts[label] for label in (0, 1, 2)},
            "primary_target_region": primary_region,
            "source_region": "plateau_source",
        })
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\SQ-variable-speed-bearing"))
    parser.add_argument("--protocol", type=Path, default=Path("SQ_RAMEF_STRICT_PROTOCOL.md"))
    parser.add_argument("--output", type=Path, default=Path("outputs/staged/SQ_RAMEF_P0"))
    parser.add_argument("--fold-count", type=int, default=9)
    parser.add_argument("--raw-window", type=int, default=None)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "protocol_frozen.md"
    if frozen.exists() and sha256(frozen) != sha256(args.protocol):
        raise RuntimeError("SQ protocol changed after P0 began")
    if not frozen.exists(): shutil.copy2(args.protocol, frozen)

    records = scan_sq(args.data_root)
    raw_window = (int(args.raw_window) if args.raw_window is not None
                  else int(round(0.256 * 25600.0)))
    rows: list[dict] = []
    details = []
    for index, record in enumerate(records):
        time, pulse = load_txt_channel(record.pulse_path, record.header_rows, record.block_size)
        step = np.diff(time)
        # Crystal Instruments text export rounds timestamps, producing a
        # maximum step error of about 6.25e-8 s at 25.6 kHz.  The tolerance is
        # fixed above that quantisation error and remains far below one sample.
        time_valid = bool(np.all(step > 0) and
                          np.max(np.abs(step - 1.0 / record.sample_rate)) < 1e-7)
        edges, speed_index, rpm = extract_speed(pulse, record.sample_rate)
        speed, slope = interpolated_speed(record.block_size, speed_index, rpm, record.sample_rate)
        region_info = {}
        for name, (low, high, slope_limit) in REGIONS.items():
            info = registered_block_starts(
                speed_mask(speed, slope, low, high, slope_limit), raw_window, 8)
            region_info[name] = info
        fixed = fixed_rotation_segments(edges, rpm, rotations=4)
        pulse_valid = bool(time_valid and len(rpm) >= 20)
        full_profile = bool(pulse_valid and np.nanmax(rpm) >= 2400
                            and np.nanmin(rpm) <= 1000)
        row = {
            "rec_id": record.rec_id, "state": record.state, "label": record.label,
            "macro": MACRO[record.label], "subtype": record.subtype,
            "sample_rate": record.sample_rate, "samples": record.block_size,
            "duration_s": record.block_size / record.sample_rate,
            "time_valid": int(time_valid), "pulse_valid": int(pulse_valid),
            "full_profile": int(full_profile),
            "pulse_edges": len(edges), "valid_rpm_points": len(rpm),
            "rpm_min": float(np.min(rpm)) if len(rpm) else None,
            "rpm_p05": float(np.quantile(rpm, .05)) if len(rpm) else None,
            "rpm_median": float(np.median(rpm)) if len(rpm) else None,
            "rpm_p95": float(np.quantile(rpm, .95)) if len(rpm) else None,
            "rpm_max": float(np.max(rpm)) if len(rpm) else None,
            "fixed_4rotation_segments": len(fixed),
        }
        for name, info in region_info.items():
            row[f"{name}_distributed"] = len(info["distributed_starts"])
            row[f"{name}_local"] = len(info["local_starts"])
            row[f"{name}_eligible_samples"] = info["eligible_samples"]
            row[f"{name}_longest_run"] = info["longest_run_samples"]
        rows.append(row)
        details.append({
            "rec_id": record.rec_id, "state": record.state, "label": record.label,
            "subtype": record.subtype, "samples": record.block_size,
            "sample_rate": record.sample_rate, "header_rows": record.header_rows,
            "pulse_edges": edges.tolist(), "speed_sample_indices": speed_index.astype(int).tolist(),
            "instantaneous_rpm": rpm.astype(float).tolist(), "regions": region_info,
            "fixed_4rotation_segments": fixed,
        })
        print(f"SQ speed audit {index + 1}/{len(records)} {record.state}/{record.rec_id}", flush=True)

    coverage = {}
    valid_rows = [row for row in rows if row["pulse_valid"]]
    target_profile_rows = [row for row in rows if row["full_profile"]]
    for name in REGIONS:
        denominator_rows = valid_rows if name == "plateau_source" else target_profile_rows
        coverage[name] = {
            "denominator_records": len(denominator_rows),
            "distributed_8": sum(row[f"{name}_distributed"] == 8 for row in denominator_rows),
            "local_8": sum(row[f"{name}_local"] == 8 for row in denominator_rows),
            "both_8": sum(row[f"{name}_distributed"] == 8 and row[f"{name}_local"] == 8
                          for row in denominator_rows),
        }
        coverage[name]["both_fraction"] = coverage[name]["both_8"] / max(len(denominator_rows), 1)

    primary = next((name for name in PRIMARY_PRIORITY
                    if coverage[name]["both_fraction"] >= .90), None)
    if primary is None:
        primary = max(PRIMARY_PRIORITY, key=lambda name: coverage[name]["both_fraction"])
    folds = []
    fold_error = None
    try:
        folds = build_folds(rows, primary, args.fold_count)
    except Exception as error:
        fold_error = str(error)

    write_csv(args.output / "record_audit.csv", rows)
    (args.output / "record_speed_metadata.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8")
    (args.output / "split_schedule.json").write_text(
        json.dumps({"protocol": "SQ_RAMEF_STRICT_V1", "primary_target_region": primary,
                    "folds": folds}, indent=2), encoding="utf-8")
    gates = {
        "record_pairs_52": len(records) == 52,
        "all_sample_rates_25600": all(row["sample_rate"] == 25600 for row in rows),
        "pulse_valid_ge_90pct": len(valid_rows) / max(len(rows), 1) >= .90,
        "primary_region_both_ge_90pct": coverage[primary]["both_fraction"] >= .90,
        "registered_leakage_free_folds": len(folds) == args.fold_count and fold_error is None,
    }
    decision = {
        "experiment": "SQ_RAMEF_P0", "status": "PASS" if all(gates.values()) else "FAIL",
        "training_authorised": bool(all(gates.values())), "record_count": len(records),
        "valid_pulse_records": len(valid_rows),
        "full_profile_records": len(target_profile_rows),
        "registered_fold_count": args.fold_count,
        "raw_window_samples": raw_window,
        "coverage": coverage, "primary_target_region": primary,
        "fold_error": fold_error, "gates": gates,
        "protocol_sha256": sha256(frozen),
        "record_audit_sha256": sha256(args.output / "record_audit.csv"),
        "speed_metadata_sha256": sha256(args.output / "record_speed_metadata.json"),
        "split_schedule_sha256": sha256(args.output / "split_schedule.json"),
    }
    (args.output / "p0_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
