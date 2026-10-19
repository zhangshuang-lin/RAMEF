"""Metadata-only qualification audit for the MCC5-THU Motor dataset.

The script parses a small HTTP-range tail of the official speed-circulation ZIP.
It never reads waveform payloads and therefore cannot produce or inspect model
performance.  Its purpose is to decide whether a strict run-disjoint RAMEF
protocol is feasible before the multi-gigabyte archive is downloaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections import Counter
from pathlib import Path


CENTRAL_SIGNATURE = b"PK\x01\x02"


def parse_central_directory(tail_path: Path) -> list[dict]:
    data = tail_path.read_bytes()
    rows: list[dict] = []
    cursor = 0
    while True:
        cursor = data.find(CENTRAL_SIGNATURE, cursor)
        if cursor < 0:
            break
        if cursor + 46 > len(data):
            break
        flag = struct.unpack_from("<H", data, cursor + 8)[0]
        compression_method = struct.unpack_from("<H", data, cursor + 10)[0]
        crc32 = struct.unpack_from("<I", data, cursor + 16)[0]
        compressed = struct.unpack_from("<I", data, cursor + 20)[0]
        uncompressed = struct.unpack_from("<I", data, cursor + 24)[0]
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", data, cursor + 28)
        end = cursor + 46 + name_len + extra_len + comment_len
        if end > len(data):
            cursor += 4
            continue
        raw_name = data[cursor + 46 : cursor + 46 + name_len]
        local_header_offset = struct.unpack_from("<I", data, cursor + 42)[0]
        encoding = "utf-8" if flag & 0x800 else "cp437"
        name = raw_name.decode(encoding, errors="replace")
        rows.append({
            "archive_path": name,
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "compression_method": compression_method,
            "crc32": f"{crc32:08x}",
            "local_header_offset": local_header_offset,
            "is_directory": name.endswith("/"),
        })
        cursor = end
    if not rows:
        raise RuntimeError("No ZIP central-directory entries found in the supplied tail")
    return rows


def normalise(text: str) -> str:
    return text.lower().replace("-", "_").replace(" ", "_")


def classify(path: str) -> dict:
    stem = normalise(Path(path).stem)
    if stem.startswith("bearing_outer_h_and_inner_h_"):
        pure_class, bearing_context = "EXCLUDE", "dual_race"
    elif stem.startswith("bearing_inner_"):
        pure_class, bearing_context = "IR", "pure"
    elif stem.startswith("bearing_outer_"):
        pure_class, bearing_context = "OR", "pure"
    elif "_and_bearing_inner_" in stem:
        pure_class, bearing_context = "EXCLUDE", "compound_IR"
    elif "_and_bearing_outer_" in stem:
        pure_class, bearing_context = "EXCLUDE", "compound_OR"
    elif stem.startswith("bearing_ball_"):
        pure_class, bearing_context = "EXCLUDE", "ball"
    elif any(token in stem for token in ("health", "healthy", "normal")):
        pure_class, bearing_context = "N", "health"
    else:
        pure_class, bearing_context = "EXCLUDE", "non_bearing_fault"

    # Exploratory metadata-defined bearing-localisation task. "N" means no
    # inner/outer race fault, not globally healthy machinery. This task is kept
    # separate from the pure N/IR/OR task and cannot be substituted silently.
    if bearing_context in {"pure", "compound_IR"} and (
        pure_class == "IR" or bearing_context == "compound_IR"
    ):
        localisation_class = "IR"
    elif bearing_context in {"pure", "compound_OR"} and (
        pure_class == "OR" or bearing_context == "compound_OR"
    ):
        localisation_class = "OR"
    elif bearing_context in {"health", "non_bearing_fault"}:
        localisation_class = "N"
    else:
        localisation_class = "EXCLUDE"

    severity = "NA"
    if localisation_class in {"IR", "OR"}:
        match = re.search(r"bearing_(?:inner|outer)_([lh])(?:_|$)", stem)
        severity = match.group(1).upper() if match else "UNKNOWN"
    torque = re.search(r"_(\d+)nm(?:_|$)", stem)
    rpm = re.search(r"_(\d+)rpm(?:_|$)", stem)
    timestamp = re.search(r"_(\d{12})$", stem)
    return {
        "pure_class": pure_class,
        "bearing_context": bearing_context,
        "localisation_class": localisation_class,
        "severity": severity,
        "nominal_torque_nm": int(torque.group(1)) if torque else None,
        "nominal_rpm_tag": int(rpm.group(1)) if rpm else None,
        "run_timestamp": timestamp.group(1) if timestamp else "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    entries = parse_central_directory(args.tail)
    files = []
    for entry in entries:
        if entry["is_directory"] or not entry["archive_path"].lower().endswith(".csv"):
            continue
        files.append({**entry, **classify(entry["archive_path"])})
    write_csv(args.output / "remote_file_catalog.csv", files)

    pure = [row for row in files if row["pure_class"] in {"N", "IR", "OR"}]
    localisation = [row for row in files if row["localisation_class"] in {"N", "IR", "OR"}]
    pure_counts = Counter(row["pure_class"] for row in pure)
    localisation_counts = Counter(row["localisation_class"] for row in localisation)
    context_counts = Counter(row["bearing_context"] for row in files)
    subtype_counts = Counter((row["localisation_class"], row["severity"]) for row in localisation)
    torque_counts = Counter((row["localisation_class"], row["nominal_torque_nm"]) for row in localisation)
    rpm_counts = Counter((row["localisation_class"], row["nominal_rpm_tag"]) for row in localisation)
    duplicate_timestamps = {
        stamp: count for stamp, count in Counter(row["run_timestamp"] for row in files if row["run_timestamp"]).items()
        if count > 1
    }
    minimum_required = {"source": 5, "support": 5, "query": 5}
    required_total = sum(minimum_required.values())
    pure_capacity = {
        cls: {
            "runs": pure_counts.get(cls, 0),
            "required": required_total,
            "passes_minimum": pure_counts.get(cls, 0) >= required_total,
        }
        for cls in ("N", "IR", "OR")
    }
    localisation_capacity = {
        cls: {
            "runs": localisation_counts.get(cls, 0),
            "required": required_total,
            "passes_minimum": localisation_counts.get(cls, 0) >= required_total,
        }
        for cls in ("N", "IR", "OR")
    }
    summary = {
        "audit": "MCC5_RAMEF_DATASET_P0_REMOTE_CATALOG",
        "waveforms_loaded": False,
        "model_predictions_used": False,
        "central_entries": len(entries),
        "csv_runs": len(files),
        "pure_bearing_candidate_runs": len(pure),
        "pure_class_counts": dict(sorted(pure_counts.items())),
        "bearing_localisation_candidate_runs": len(localisation),
        "bearing_localisation_class_counts": dict(sorted(localisation_counts.items())),
        "bearing_context_counts": dict(sorted(context_counts.items())),
        "subtype_counts": {f"{key[0]}_{key[1]}": value for key, value in sorted(subtype_counts.items())},
        "torque_counts": {f"{key[0]}_{key[1]}": value for key, value in sorted(torque_counts.items(), key=str)},
        "rpm_tag_counts": {f"{key[0]}_{key[1]}": value for key, value in sorted(rpm_counts.items(), key=str)},
        "duplicate_timestamp_count": len(duplicate_timestamps),
        "strict_capacity_requirement_per_class": minimum_required,
        "pure_task_strict_capacity": pure_capacity,
        "pure_task_remote_catalog_gate": all(item["passes_minimum"] for item in pure_capacity.values()),
        "localisation_task_strict_capacity": localisation_capacity,
        "localisation_task_remote_catalog_gate": all(
            item["passes_minimum"] for item in localisation_capacity.values()
        ),
        "notes": [
            "A file is provisionally treated as one independent run; hardware/bearing identity still requires documentation audit.",
            "The remote gate only checks count capacity. Signal channels and usable speed coverage require sample-waveform P0.",
        ],
    }
    (args.output / "remote_catalog_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
