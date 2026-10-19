"""Freeze the metadata-only MCC5 bearing-localisation protocol.

No waveform or model output is read.  The 3000/2000/1000-rpm-tagged runs are
assigned to source/source-validation/target roles, respectively.  Target
support selection is deterministic and torque-stratified; every remaining
target run is query.  A run is an archive CSV path, never a derived window.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


SEED = "MCC5_RAMEF_LOCALISATION_V1_20260824"
CLASSES = ("N", "IR", "OR")


def stable_key(path: str) -> str:
    return hashlib.sha256(f"{SEED}:{path}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.catalog.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if not row.get("archive_path") and row.get("relative_path"):
            row["archive_path"] = row["relative_path"]
    rows = [
        row for row in rows
        if row["localisation_class"] in CLASSES
        and (not row.get("mode") or row["mode"] == "speed_circulation")
    ]

    assignments: list[dict] = []
    for row in rows:
        rpm = int(row["nominal_rpm_tag"])
        role = {3000: "source", 2000: "source_validation", 1000: "target_pool"}.get(rpm)
        if role:
            assignments.append({**row, "role": role})

    support_paths: set[str] = set()
    target = [row for row in assignments if row["role"] == "target_pool"]
    for cls in CLASSES:
        class_rows = [row for row in target if row["localisation_class"] == cls]
        # Five support runs per class: three at 20 Nm and two at 40 Nm. This
        # rule uses operating metadata only and was frozen before modelling.
        for torque, quota in ((20, 3), (40, 2)):
            candidates = [row for row in class_rows if int(row["nominal_torque_nm"]) == torque]
            candidates.sort(key=lambda row: stable_key(row["archive_path"]))
            if len(candidates) < quota:
                raise RuntimeError(f"Insufficient {cls}/{torque}Nm target support candidates")
            support_paths.update(row["archive_path"] for row in candidates[:quota])

    for row in assignments:
        if row["role"] == "target_pool":
            row["role"] = "target_support" if row["archive_path"] in support_paths else "target_query"

    roles = Counter(row["role"] for row in assignments)
    class_roles = Counter((row["role"], row["localisation_class"]) for row in assignments)
    if len({row["archive_path"] for row in assignments}) != len(assignments):
        raise RuntimeError("Archive path appears in more than one role")
    if any(class_roles[("target_support", cls)] != 5 for cls in CLASSES):
        raise RuntimeError("Support is not exactly five runs per class")
    if any(class_roles[("target_query", cls)] < 5 for cls in CLASSES):
        raise RuntimeError("Fewer than five independent query runs in a class")

    csv_path = args.output / "protocol_run_manifest.csv"
    fields = ["role", "localisation_class", "bearing_context", "severity",
              "nominal_torque_nm", "nominal_rpm_tag", "run_timestamp", "crc32",
              "uncompressed_bytes", "archive_path"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(assignments, key=lambda row: (row["role"], row["localisation_class"], row["archive_path"])))

    result = {
        "protocol": "MCC5_RAMEF_BEARING_LOCALISATION_V1",
        "status": "CONDITIONAL_METADATA_PASS",
        "selection_seed": SEED,
        "waveforms_used_for_split": False,
        "model_outputs_used_for_split": False,
        "task_semantics": {
            "N": "no inner/outer-race bearing fault; includes healthy and non-bearing motor faults",
            "IR": "pure or compound inner-race bearing fault",
            "OR": "pure or compound outer-race bearing fault",
            "excluded": "ball-bearing faults",
        },
        "domain_roles": {
            "source": "3000-rpm-tagged speed-circulation runs",
            "source_validation": "2000-rpm-tagged speed-circulation runs",
            "target_support_query": "1000-rpm-tagged speed-circulation runs",
        },
        "counts_by_role": dict(sorted(roles.items())),
        "counts_by_role_and_class": {
            f"{role}_{cls}": class_roles[(role, cls)]
            for role in ("source", "source_validation", "target_support", "target_query")
            for cls in CLASSES
        },
        "support_rule": "per class, SHA256 order selects 3 runs at 20 Nm and 2 runs at 40 Nm",
        "block_protocol_draft": {
            "sample_rate_hz": 12800,
            "block_samples": 3072,
            "nominal_rotations_at_1000rpm": 4.0,
            "single": "one preregistered local block",
            "local_m8": "eight contiguous non-overlapping blocks within a stable high plateau",
            "distributed_m8": "eight non-overlapping blocks spread over the 90-s speed profile",
            "equal_m8_input_samples": 24576,
        },
        "remaining_gates": [
            "all selected files must decode to time plus eight synchronous channels",
            "keyphase must provide valid speed coverage for all selected files",
            "bearing specimen identity is not documented; claims are run-disjoint, not specimen-disjoint",
            "BPFO/BPFI order constants must be taken from the dataset article, not tuned on query",
        ],
    }
    json_path = args.output / "conditional_protocol.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["manifest_sha256"] = sha256(csv_path)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
