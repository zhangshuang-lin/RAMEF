"""Replay the PU unique-measurement paired statistics from frozen predictions.

No waveform, feature extractor, training, model selection or target-query
evaluation is performed here. The 240 archived query entries are collapsed to
171 physical measurement identities before every calculation. This is a
post-hoc, within-development-cohort statistical replay.

Run once to prepare the package from the archived ledger:
    python replay.py --prepare-source PATH_TO_ARCHIVED_PREDICTIONS.csv
                     --crosscheck-head PATH_TO_HEAD_LEDGER.csv
                     --crosscheck-final PATH_TO_LOCKED_LEDGER.csv

Thereafter the package is self-contained:
    python replay.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = "f51d74badfbccaede3b4442c75186a0e7cd560d73104be60db45005786110007"
HEAD_SHA256 = "f8a21ca0f09c8147dc86645e59ed9286d1e453ce35559ac34da3f1e15179bf09"
FINAL_SHA256 = "d0216382dc28a72fd10b30e6f497be93aa79b285a5baad7a61901425903c5e64"
LABELS = ("N", "IR", "OR")
INPUT = HERE / "frozen_240_entry_predictions.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(source: Path, head: Path | None, final: Path | None) -> None:
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Archived fair-ablation prediction ledger SHA-256 mismatch")
    raw = load_csv(source)
    required = {"measurement_id", "bearing_code", "true", "B0_A1_SINGLE", "B5_A1_MEAN8_RACP"}
    if len(raw) != 240 or not required.issubset(raw[0]):
        raise ValueError("Archived prediction ledger has unexpected schema or length")

    by_id = {row["measurement_id"]: row for row in raw}
    if len(by_id) != 171:
        raise ValueError("Expected 171 distinct physical measurement IDs")
    checks = {}
    if head is not None:
        if digest(head) != HEAD_SHA256:
            raise ValueError("Independent classifier-head ledger SHA-256 mismatch")
        other = {row["measurement_id"]: row for row in load_csv(head)}
        if set(other) != set(by_id) or any(
            row["true"] != other[identity]["true"]
            or row["B5_A1_MEAN8_RACP"] != other[identity]["RAMEF-F__H2_registered_w10"]
            for identity, row in by_id.items()
        ):
            raise ValueError("RAMEF-F predictions disagree with independent head ledger")
        checks["independent_head_ledger_sha256"] = HEAD_SHA256
    if final is not None:
        if digest(final) != FINAL_SHA256:
            raise ValueError("Independent locked-run ledger SHA-256 mismatch")
        other = {row["measurement_id"]: row for row in load_csv(final)}
        if set(other) != set(by_id) or any(
            row["true"] != other[identity]["true"]
            or row["B0_A1_SINGLE"] != other[identity]["A1_SINGLE"]
            for identity, row in by_id.items()
        ):
            raise ValueError("Centre-crop predictions disagree with locked-run ledger")
        checks["independent_locked_run_ledger_sha256"] = FINAL_SHA256

    compact = [
        {
            "entry_index": index,
            "measurement_id": row["measurement_id"],
            "bearing_code": row["bearing_code"],
            "true": row["true"],
            "centre_pred": row["B0_A1_SINGLE"],
            "ramef_f_pred": row["B5_A1_MEAN8_RACP"],
        }
        for index, row in enumerate(raw)
    ]
    write_csv(INPUT, list(compact[0]), compact)
    provenance = {
        "archived_fair_ablation_ledger_sha256": SOURCE_SHA256,
        **checks,
        "packaged_input_sha256": digest(INPUT),
        "source_role": "frozen PU target-query predictions, not raw vibration or new model output",
    }
    (HERE / "source_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validated_unique() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    entries = load_csv(INPUT)
    if len(entries) != 240 or [int(row["entry_index"]) for row in entries] != list(range(240)):
        raise ValueError("Packaged query entries are not the 240-entry frozen order")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in entries:
        if row["true"] not in {"0", "1", "2"}:
            raise ValueError("Unexpected class label")
        grouped[row["measurement_id"]].append(row)
    if len(grouped) != 171:
        raise ValueError("Expected 171 distinct measurement IDs")
    for identity, copies in grouped.items():
        if len({(r["bearing_code"], r["true"], r["centre_pred"], r["ramef_f_pred"])
                for r in copies}) != 1:
            raise ValueError(f"Repeated entries disagree for {identity}")
    unique = [grouped[identity][0] for identity in sorted(grouped)]
    if Counter(row["true"] for row in unique) != {"0": 55, "1": 62, "2": 54}:
        raise ValueError("Unique-measurement class denominators changed")
    return entries, unique


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    cm = np.zeros((3, 3), dtype=np.int64)
    for truth, estimate in zip(y, pred):
        cm[int(truth), int(estimate)] += 1
    f1 = [2 * cm[c, c] / (cm[c].sum() + cm[:, c].sum()) for c in range(3)]
    recall = [cm[c, c] / cm[c].sum() for c in range(3)]
    return {
        "accuracy": float(np.trace(cm) / len(y)),
        "macro_f1": float(np.mean(f1)),
        "recall_N": float(recall[0]),
        "recall_IR": float(recall[1]),
        "recall_OR": float(recall[2]),
    }


def summarize(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "ci_low": float(np.quantile([row[key] for row in rows], 0.025)),
            "ci_high": float(np.quantile([row[key] for row in rows], 0.975)),
        }
        for key in keys
    }


def replay() -> dict:
    entries, unique = validated_unique()
    frame = [
        {
            "measurement_id": row["measurement_id"],
            "bearing_code": row["bearing_code"],
            "true": int(row["true"]),
            "centre_pred": int(row["centre_pred"]),
            "ramef_f_pred": int(row["ramef_f_pred"]),
        }
        for row in unique
    ]
    y = np.asarray([row["true"] for row in frame], dtype=np.int64)
    centre = np.asarray([row["centre_pred"] for row in frame], dtype=np.int64)
    ramef = np.asarray([row["ramef_f_pred"] for row in frame], dtype=np.int64)
    observed_a, observed_b = metrics(y, centre), metrics(y, ramef)
    corrected = int(np.sum((centre != y) & (ramef == y)))
    introduced = int(np.sum((centre == y) & (ramef != y)))
    if (corrected, introduced) != (17, 0):
        raise ValueError("Frozen paired error transition changed")
    exact_p = min(1.0, 2 * sum(math.comb(corrected + introduced, k)
                              for k in range(min(corrected, introduced) + 1))
                  / 2 ** (corrected + introduced))

    unique_rows = []
    for row in frame:
        base_ok = row["centre_pred"] == row["true"]
        full_ok = row["ramef_f_pred"] == row["true"]
        transition = "corrected" if not base_ok and full_ok else (
            "introduced" if base_ok and not full_ok else ("both_correct" if base_ok else "both_wrong")
        )
        unique_rows.append({**row, "transition": transition})
    write_csv(HERE / "unique_171_measurement_predictions.csv", list(unique_rows[0]), unique_rows)

    rng = np.random.default_rng(20260924)
    by_class = [np.flatnonzero(y == cls) for cls in range(3)]
    measurement_rows = []
    keys = ("accuracy", "macro_f1", "recall_OR")
    for replicate in range(10_000):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in by_class
        ])
        ma, mb = metrics(y[indices], centre[indices]), metrics(y[indices], ramef[indices])
        measurement_rows.append({"replicate": replicate + 1, **{
            key: 100 * (mb[key] - ma[key]) for key in keys
        }})
    write_csv(HERE / "measurement_bootstrap_replicates.csv",
              ["replicate", *keys], measurement_rows)

    by_code: dict[str, np.ndarray] = {}
    for code in sorted({row["bearing_code"] for row in frame}):
        by_code[code] = np.asarray([
            i for i, row in enumerate(frame) if row["bearing_code"] == code
        ], dtype=np.int64)
    codes_by_class = {
        cls: sorted(code for code, indices in by_code.items() if int(y[indices[0]]) == cls)
        for cls in range(3)
    }
    rng_codes = random.Random(20260925)
    cluster_rows = []
    all_keys = ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR")
    for replicate in range(10_000):
        selected = [
            rng_codes.choice(codes_by_class[cls])
            for cls in range(3) for _ in codes_by_class[cls]
        ]
        indices = np.concatenate([by_code[code] for code in selected])
        ma, mb = metrics(y[indices], centre[indices]), metrics(y[indices], ramef[indices])
        cluster_rows.append({"replicate": replicate + 1, "n_measurements": len(indices), **{
            key: 100 * (mb[key] - ma[key]) for key in all_keys
        }})
    write_csv(HERE / "code_cluster_bootstrap_replicates.csv",
              ["replicate", "n_measurements", *all_keys], cluster_rows)

    code_rows = []
    for code, indices in by_code.items():
        fixed = int(np.sum((centre[indices] != y[indices]) & (ramef[indices] == y[indices])))
        new = int(np.sum((centre[indices] == y[indices]) & (ramef[indices] != y[indices])))
        code_rows.append({
            "bearing_code": code, "class": int(y[indices[0]]), "n_measurements": len(indices),
            "errors_corrected": fixed, "errors_introduced": new, "net_corrections": fixed - new,
        })
    write_csv(HERE / "code_contributions.csv", list(code_rows[0]), code_rows)

    def sign_flip(class_filter: int | None = None) -> dict:
        values = [row["net_corrections"] for row in code_rows
                  if row["net_corrections"]
                  and (class_filter is None or row["class"] == class_filter)]
        observed = sum(values)
        p = sum(abs(sum(s * v for s, v in zip(signs, values))) >= abs(observed)
                for signs in itertools.product((-1, 1), repeat=len(values))) / 2 ** len(values)
        return {"nonzero_clusters": len(values), "observed_net_corrections": observed,
                "two_sided_p": p}

    leave_code_out = []
    for code, indices in by_code.items():
        remaining = np.asarray([i for i in range(len(frame)) if i not in set(indices)])
        ma, mb = metrics(y[remaining], centre[remaining]), metrics(y[remaining], ramef[remaining])
        leave_code_out.append({"bearing_code": code,
                               "accuracy_gain_pp": 100 * (mb["accuracy"] - ma["accuracy"]),
                               "or_recall_gain_pp": 100 * (mb["recall_OR"] - ma["recall_OR"])})
    write_csv(HERE / "leave_one_code_out.csv", list(leave_code_out[0]), leave_code_out)

    summary = {
        "permission": "post-hoc fixed-prediction replay on historically accessed PU development cohort",
        "score_unit": "one vote per distinct physical measurement ID",
        "query_entries": len(entries), "unique_measurements": len(unique),
        "class_counts": {LABELS[c]: int(np.sum(y == c)) for c in range(3)},
        "comparison": "locked centre-crop U2-Hz versus locked RAMEF-F",
        "observed": {
            "centre": {key: 100 * value for key, value in observed_a.items()},
            "ramef_f": {key: 100 * value for key, value in observed_b.items()},
            "corrected": corrected, "introduced": introduced,
            "exact_two_sided_p": exact_p,
        },
        "measurement_bootstrap": {
            "seed": 20260924, "replicates": 10_000,
            "resampling": "measurement IDs with replacement within N, IR and OR",
            "summary_pp": summarize(measurement_rows, keys),
        },
        "bearing_code_cluster_bootstrap": {
            "seed": 20260925, "replicates": 10_000,
            "resampling": "bearing-code clusters with replacement within N, IR and OR",
            "summary_pp": summarize(cluster_rows, all_keys),
        },
        "code_sign_flip_all": sign_flip(),
        "code_sign_flip_OR": sign_flip(2),
        "leave_one_code_out_accuracy_gain_range_pp": [
            min(row["accuracy_gain_pp"] for row in leave_code_out),
            max(row["accuracy_gain_pp"] for row in leave_code_out),
        ],
    }
    # Exact expected statistics were archived before this package was built.
    historical = json.loads((HERE / "historical_expected.json").read_text(encoding="utf-8"))
    for family, metric in historical["checks"]:
        current = summary
        for key in family:
            current = current[key]
        if not math.isclose(float(current), float(metric), rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Replay does not reproduce historical result at {family}: {current}")

    (HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_rows = []
    for family, unit in (("measurement_bootstrap", "measurement"),
                         ("bearing_code_cluster_bootstrap", "bearing_code_cluster")):
        result = summary[family]
        for metric, values in result["summary_pp"].items():
            summary_rows.append({
                "resampling_unit": unit,
                "metric": metric,
                "replicates": result["replicates"],
                "seed": result["seed"],
                "mean_gain_pp": values["mean"],
                "ci_low_pp": values["ci_low"],
                "ci_high_pp": values["ci_high"],
            })
    write_csv(HERE / "summary_table.csv", list(summary_rows[0]), summary_rows)
    files = [
        "README.md", "protocol_frozen.md", "replay.py", "historical_expected.json",
        "source_manifest.json", "frozen_240_entry_predictions.csv",
        "unique_171_measurement_predictions.csv", "measurement_bootstrap_replicates.csv",
        "code_cluster_bootstrap_replicates.csv", "code_contributions.csv",
        "leave_one_code_out.csv", "summary.json", "summary_table.csv",
    ]
    (HERE / "MANIFEST.sha256").write_text(
        "".join(f"{digest(HERE / filename)}  {filename}\n" for filename in files),
        encoding="ascii",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-source", type=Path)
    parser.add_argument("--crosscheck-head", type=Path)
    parser.add_argument("--crosscheck-final", type=Path)
    args = parser.parse_args()
    if args.prepare_source is not None:
        prepare(args.prepare_source, args.crosscheck_head, args.crosscheck_final)
    elif not INPUT.exists():
        parser.error("Packaged input is absent; use --prepare-source once")
    summary = replay()
    print(json.dumps({
        "unique_measurements": summary["unique_measurements"],
        "corrected": summary["observed"]["corrected"],
        "introduced": summary["observed"]["introduced"],
        "measurement_accuracy_gain": summary["measurement_bootstrap"]["summary_pp"]["accuracy"],
        "code_cluster_accuracy_gain": summary["bearing_code_cluster_bootstrap"]["summary_pp"]["accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
