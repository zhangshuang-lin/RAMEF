"""Replay RAMEF paper scores from frozen prediction CSVs, using stdlib only.

This entry point reads neither waveforms nor model files and never fits a model.
PU is a historically accessed development cohort; MCC5 is a fixed run-level
mechanistic stress test. Seed dispersion is not external-cohort uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


PU_COUNTS = {0: 55, 1: 62, 2: 54}
MCC5_COUNTS = {0: 13, 1: 7, 2: 7}
METRICS = ("accuracy", "macro_f1", "recall_0", "recall_1", "recall_2")
REQUIRED_PU = {
    "cohort", "seed", "method", "measurement_id", "bearing_code", "true", "prediction"
}
REQUIRED_MCC5 = {
    "cohort", "seed", "method", "run_id", "true", "prediction", "context", "severity", "torque"
}


def compute_metrics(truth: Iterable[int], prediction: Iterable[int]) -> dict:
    """Compute a 3-class confusion matrix and scores in percentage points."""
    truth, prediction = list(truth), list(prediction)
    if not truth or len(truth) != len(prediction):
        raise ValueError("Truth and prediction must be nonempty and equally sized")
    cm = [[0] * 3 for _ in range(3)]
    for y, p in zip(truth, prediction):
        if y not in (0, 1, 2) or p not in (0, 1, 2):
            raise ValueError(f"Class outside 0/1/2: true={y!r}, prediction={p!r}")
        cm[y][p] += 1
    recalls, f1s = [], []
    for c in range(3):
        actual = sum(cm[c])
        predicted = sum(row[c] for row in cm)
        recalls.append(100 * cm[c][c] / actual if actual else 0.0)
        f1s.append(200 * cm[c][c] / (actual + predicted) if actual + predicted else 0.0)
    correct = sum(cm[c][c] for c in range(3))
    return {
        "n": len(truth), "correct": correct, "errors": len(truth) - correct,
        "accuracy": 100 * correct / len(truth), "macro_f1": mean(f1s),
        **{f"recall_{c}": recalls[c] for c in range(3)}, "confusion_matrix": cm,
    }


def deduplicate_records(records: Iterable[dict], identity: str,
                        consistency: tuple[str, ...] = ("true", "prediction")) -> list[dict]:
    """Collapse repeated identities only when every relevant value agrees."""
    seen = {}
    for record in records:
        key = record.get(identity)
        if key is None or not str(key).strip():
            raise ValueError(f"Missing {identity}")
        if key in seen:
            conflicts = [field for field in consistency if record[field] != seen[key][field]]
            if conflicts:
                raise ValueError(f"Conflicting duplicate {identity}={key!r}: {', '.join(conflicts)}")
        else:
            seen[key] = record
    return [seen[key] for key in sorted(seen)]


def load_predictions(path: Path, dataset: str) -> list[dict]:
    required = REQUIRED_PU if dataset == "pu" else REQUIRED_MCC5
    if not path.is_file():
        raise ValueError(f"Missing frozen prediction CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"{path}: missing columns {missing}")
        records = []
        for line, source in enumerate(reader, 2):
            if None in source or any(source.get(field) is None for field in required):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            row = {field: str(value).strip() for field, value in source.items()}
            for field in ("cohort", "seed", "method", "measurement_id" if dataset == "pu" else "run_id"):
                if not row[field]:
                    raise ValueError(f"{path}:{line}: empty {field}")
            if dataset == "pu" and not row["bearing_code"]:
                raise ValueError(f"{path}:{line}: empty bearing_code")
            for field in ("true", "prediction"):
                if row[field] not in ("0", "1", "2"):
                    raise ValueError(f"{path}:{line}: {field} must be 0, 1 or 2")
                row[field] = int(row[field])
            records.append(row)
    if not records:
        raise ValueError(f"Empty frozen prediction CSV: {path}")
    return records


def score_groups(records: list[dict], dataset: str) -> list[dict]:
    identity = "measurement_id" if dataset == "pu" else "run_id"
    expected = PU_COUNTS if dataset == "pu" else MCC5_COUNTS
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["cohort"], row["seed"], row["method"])].append(row)
    results = []
    common_ids = None
    for (cohort, seed, method), group in sorted(grouped.items()):
        consistency = ("true", "prediction", "bearing_code") if dataset == "pu" else (
            "true", "prediction", "context", "severity", "torque"
        )
        unique = deduplicate_records(group, identity, consistency)
        if dataset == "mcc5" and len(unique) != len(group):
            raise ValueError(f"MCC5 {cohort}/{seed}/{method}: duplicate run IDs are not permitted")
        counts = dict(Counter(row["true"] for row in unique))
        if counts != expected:
            raise ValueError(f"{dataset} {cohort}/{seed}/{method}: class counts {counts}; expected {expected}")
        identities = {row[identity]: row["true"] for row in unique}
        if common_ids is None:
            common_ids = identities
        elif identities != common_ids:
            raise ValueError(f"{dataset} {cohort}/{seed}/{method}: cohort identities or truths differ")
        scores = compute_metrics((row["true"] for row in unique), (row["prediction"] for row in unique))
        results.append({
            "dataset": dataset, "cohort": cohort, "seed": seed, "method": method,
            "entry_count": len(group), "unique_count": len(unique),
            "duplicate_count": len(group) - len(unique), **scores,
        })
    if not results:
        raise ValueError(f"No {dataset} prediction groups")
    return results


def summarize_seeds(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["cohort"], row["method"])].append(row)
    result = []
    for (dataset, cohort, method), group in sorted(groups.items()):
        seeds = [row["seed"] for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Repeated seed metric: {dataset}/{cohort}/{method}")
        item = {"dataset": dataset, "cohort": cohort, "method": method,
                "seed_count": len(group), "seeds": ";".join(sorted(seeds))}
        for field in METRICS:
            values = [row[field] for row in group]
            item[f"{field}_mean"] = mean(values)
            item[f"{field}_sample_sd"] = stdev(values) if len(values) > 1 else ""
        result.append(item)
    return result


def _assert_rounded(value: float, expected: str, description: str) -> None:
    if not math.isfinite(value) or f"{value:.2f}" != expected:
        raise ValueError(f"Headline mismatch: {description}={value!r}; expected {expected} at 2 decimals")


def validate_headlines(pu: list[dict], summaries: list[dict]) -> dict:
    checks = {}
    for method, expected in (("centre", "88.89"), ("U2-Hz-DM8", "97.66"), ("RAMEF-F", "98.83")):
        selected = [r for r in pu if r["cohort"] == "locked" and r["method"] == method]
        if len(selected) != 1:
            raise ValueError(f"Require exactly one locked PU group for {method}; found {len(selected)}")
        _assert_rounded(selected[0]["accuracy"], expected, f"PU locked {method} Accuracy")
        checks[f"pu_locked_{method}_accuracy"] = selected[0]["accuracy"]
    expected_summary = (
        ("pu", "new_five_seed", "RAMEF-F", "accuracy", "94.04", "1.91"),
        ("pu", "new_five_seed", "RAMEF-F", "macro_f1", "93.76", "2.00"),
        ("mcc5", "primary", "RAMEF", "accuracy", "96.30", "2.62"),
    )
    for dataset, cohort, method, metric, average, sd in expected_summary:
        selected = [r for r in summaries if (r["dataset"], r["cohort"], r["method"]) == (dataset, cohort, method)]
        if len(selected) != 1 or selected[0]["seed_count"] != 5:
            raise ValueError(f"Require five seeds for {dataset}/{cohort}/{method}")
        row = selected[0]
        if dataset == "pu" and set(row["seeds"].split(";")) != {str(s) for s in range(20260911, 20260916)}:
            raise ValueError("New PU five-seed identifiers must be 20260911--20260915")
        _assert_rounded(row[f"{metric}_mean"], average, f"{dataset} {cohort} {method} {metric} mean")
        _assert_rounded(row[f"{metric}_sample_sd"], sd, f"{dataset} {cohort} {method} {metric} sample SD")
        checks[f"{dataset}_{cohort}_{method}_{metric}"] = {
            "mean": row[f"{metric}_mean"], "sample_sd": row[f"{metric}_sample_sd"]
        }
    return checks


def create_output_dir(path: Path) -> None:
    """Even an existing empty directory is rejected to protect all old results."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True, exist_ok=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            if "confusion_matrix" in row:
                row["confusion_matrix"] = json.dumps(row["confusion_matrix"], separators=(",", ":"))
            writer.writerow(row)


def replay(repo_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    inputs = {
        "pu": repo_root / "evidence/pu/predictions.csv",
        "mcc5": repo_root / "evidence/mcc5/predictions.csv",
    }
    pu = score_groups(load_predictions(inputs["pu"], "pu"), "pu")
    mcc5 = score_groups(load_predictions(inputs["mcc5"], "mcc5"), "mcc5")
    summaries = summarize_seeds(pu + mcc5)
    headlines = validate_headlines(pu, summaries)
    report = {
        "status": "PASS_FROZEN_PREDICTION_REPLAY", "units": "percent",
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
        "pu_unique_measurements": 171, "pu_class_counts": PU_COUNTS,
        "mcc5_query_runs": 27, "mcc5_class_counts": MCC5_COUNTS,
        "group_counts": {"pu": len(pu), "mcc5": len(mcc5)}, "headline_checks": headlines,
        "evidence_scope": {
            "pu": "Historically accessed locked development cohort; one vote per measurement ID",
            "mcc5": "Locally prespecified run-level mechanistic stress test; subsequent audits are post-target",
            "seed_dispersion": "Sample SD across encoder seeds on a shared fixed query cohort, not a population CI",
        },
        "waveforms_read": False, "models_loaded": False, "training_performed": False,
    }
    create_output_dir(output_dir)
    write_csv(output_dir / "pu_metrics.csv", pu)
    write_csv(output_dir / "mcc5_seed_metrics.csv", mcc5)
    write_csv(output_dir / "seed_summary.csv", summaries)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; an existing directory is always rejected")
    args = parser.parse_args()
    try:
        report = replay(args.repo_root.resolve(), args.output_dir.resolve())
    except (ValueError, OSError) as error:
        parser.exit(2, f"Replay failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
