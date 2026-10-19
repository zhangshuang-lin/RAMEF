"""Post-target label-semantics and matched-nuisance sensitivity audit.

The audit reuses frozen prediction ledgers. It does not fit a new model, alter
labels, or change the confirmatory PASS. Interpretation criteria are explicitly
post-hoc and determine manuscript placement only.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "staged" / "MCC5_RAMEF_CONFIRMATORY"
PRED = BASE / "run" / "predictions.csv"
PHYS = BASE / "posthoc_physical_only" / "predictions.csv"
OUT = BASE / "label_semantics_audit"
SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
VARIANTS = ("U2-Hz-K1", "U2-Hz-Local-M8", "U2-Hz-Distributed-M8", "RAMEF-F", "Random63")
LABEL_NAMES = {0: "NRF", 1: "IR", 2: "OR"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def nuisance_family(run_id: str, true: int) -> str:
    name = Path(run_id).name.lower()
    if true == 0:
        if name.startswith("health_"): return "none"
        if name.startswith("broken_bar_"): return "broken_bar"
        if name.startswith("dynamic_eccentricity_"): return "dynamic_eccentricity"
        if name.startswith("static_eccentricity_h_"): return "static_eccentricity_H"
        if name.startswith("winding_h_"): return "winding_H"
        if name.startswith("bend_"): return "bend_only_NRF"
        if name.startswith("static_eccentricity_l_"): return "static_eccentricity_L_only_NRF"
        if name.startswith("voltage_unbalance_"): return "voltage_unbalance_only_NRF"
        if name.startswith("winding_l_"): return "winding_L_only_NRF"
    else:
        race = "inner" if true == 1 else "outer"
        if name.startswith(f"bearing_{race}_"): return "none"
        if name.startswith(f"broken_bar_and_bearing_{race}_"): return "broken_bar"
        if name.startswith(f"dynamic_eccentricity_and_bearing_{race}_"): return "dynamic_eccentricity"
        if name.startswith(f"static_eccentricity_h_and_bearing_{race}_"): return "static_eccentricity_H"
        if name.startswith(f"winding_h_and_bearing_{race}_"): return "winding_H"
    return "unmapped"


def metric(truth: np.ndarray, pred: np.ndarray) -> dict:
    recall = recall_score(truth, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(truth == pred)),
            "macro_f1": float(f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_NRF": float(recall[0]), "recall_IR": float(recall[1]), "recall_OR": float(recall[2])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(PRED)
    for row in rows:
        row["seed"] = int(row["seed"]); row["true"] = int(row["true"])
        row["prediction"] = int(row["prediction"]); row["correct"] = int(row["correct"])
        row["torque"] = int(row["torque"])
        row["family"] = nuisance_family(row["run_id"], row["true"])

    # One metadata record per query run.
    metadata = {}
    for row in rows:
        metadata.setdefault(row["run_id"], {key: row[key] for key in
                                            ("run_id", "true", "context", "severity", "torque", "family")})
    family_by_class = {label: {item["family"] for item in metadata.values() if item["true"] == label}
                       for label in (0, 1, 2)}
    common_families = sorted(set.intersection(*(family_by_class[label] for label in (0, 1, 2))))
    matched_ids = {item["run_id"] for item in metadata.values() if item["family"] in common_families}
    matched_counts = Counter(metadata[item]["true"] for item in matched_ids)
    family_counts = Counter((item["family"], LABEL_NAMES[item["true"]]) for item in metadata.values())

    matched_metrics = []
    predictions = {(row["seed"], row["variant"], row["run_id"]): row["prediction"] for row in rows}
    ordered_ids = sorted(matched_ids)
    truth = np.asarray([metadata[item]["true"] for item in ordered_ids])
    by_seed_variant = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            pred = np.asarray([predictions[(seed, variant, item)] for item in ordered_ids])
            value = metric(truth, pred); by_seed_variant[(seed, variant)] = value
            matched_metrics.append({"seed": seed, "variant": variant, "matched_runs": len(ordered_ids), **value})
    write_csv(OUT / "matched_nuisance_metrics.csv", matched_metrics)

    # Per-family and clinically interpretable subgroups, averaged across seeds.
    subgroup_rows = []
    groups = defaultdict(list)
    for item in metadata.values():
        groups[f"family:{item['family']}"].append(item["run_id"])
        groups[f"torque:{item['torque']}Nm"].append(item["run_id"])
        if item["true"] in {1, 2}:
            groups[f"fault_context:{item['context']}"].append(item["run_id"])
            groups[f"severity:{item['severity']}"].append(item["run_id"])
    for group, ids in sorted(groups.items()):
        for variant in VARIANTS:
            seed_acc = []
            for seed in SEEDS:
                seed_acc.append(np.mean([predictions[(seed, variant, item)] == metadata[item]["true"] for item in ids]))
            subgroup_rows.append({"group": group, "runs": len(ids), "variant": variant,
                                  "accuracy_mean": float(np.mean(seed_acc)),
                                  "accuracy_sd": float(np.std(seed_acc, ddof=1))})
    write_csv(OUT / "subgroup_accuracy.csv", subgroup_rows)

    # Paired transitions on the matched subset.
    transition_rows = []
    for seed in SEEDS:
        for baseline in ("U2-Hz-Local-M8", "U2-Hz-Distributed-M8", "Random63"):
            fixed = added = 0
            for item in ordered_ids:
                base_ok = predictions[(seed, baseline, item)] == metadata[item]["true"]
                ramef_ok = predictions[(seed, "RAMEF-F", item)] == metadata[item]["true"]
                fixed += int(not base_ok and ramef_ok); added += int(base_ok and not ramef_ok)
            transition_rows.append({"seed": seed, "baseline": baseline, "corrected": fixed,
                                    "introduced": added, "net": fixed - added})
    write_csv(OUT / "matched_transitions.csv", transition_rows)

    # Physical-only is one deterministic post-hoc head.
    phys = read_csv(PHYS)
    phys_map = {row["run_id"]: int(row["prediction"]) for row in phys}
    phys_pred = np.asarray([phys_map[item] for item in ordered_ids])
    physical_metric = metric(truth, phys_pred)

    delta_local = np.asarray([by_seed_variant[(seed, "RAMEF-F")]["macro_f1"] -
                              by_seed_variant[(seed, "U2-Hz-Local-M8")]["macro_f1"] for seed in SEEDS])
    delta_dist = np.asarray([by_seed_variant[(seed, "RAMEF-F")]["macro_f1"] -
                             by_seed_variant[(seed, "U2-Hz-Distributed-M8")]["macro_f1"] for seed in SEEDS])
    delta_random = np.asarray([by_seed_variant[(seed, "RAMEF-F")]["macro_f1"] -
                               by_seed_variant[(seed, "Random63")]["macro_f1"] for seed in SEEDS])
    recall_safe = all(
        np.mean([by_seed_variant[(seed, "RAMEF-F")][key] for seed in SEEDS]) + 1e-12 >=
        np.mean([by_seed_variant[(seed, "U2-Hz-Distributed-M8")][key] for seed in SEEDS])
        for key in ("recall_NRF", "recall_IR", "recall_OR")
    )
    criteria = {
        "matched_subset_at_least_9_runs": len(ordered_ids) >= 9,
        "each_class_at_least_2_runs": all(matched_counts[label] >= 2 for label in (0, 1, 2)),
        "ramef_over_local_positive_at_least_4of5": int(np.sum(delta_local > 0)) >= 4,
        "ramef_over_distributed_positive_at_least_4of5": int(np.sum(delta_dist > 0)) >= 4,
        "ramef_over_random_positive_at_least_4of5": int(np.sum(delta_random > 0)) >= 4,
        "matched_class_recalls_nonworse_than_distributed": recall_safe,
        "no_matched_error_introductions_vs_distributed": all(
            row["introduced"] == 0 for row in transition_rows if row["baseline"] == "U2-Hz-Distributed-M8"),
    }
    if all(criteria.values()):
        placement = "RETAIN_MAIN_TEXT_AS_EXTERNAL_MECHANISTIC_STRESS_TEST"
    elif len(ordered_ids) < 9 or not all(matched_counts[label] >= 2 for label in (0, 1, 2)):
        placement = "DEMOTE_OR_QUALIFY_DUE_TO_SMALL_MATCHED_SUBSET"
    else:
        placement = "DEMOTE_TO_SUPPLEMENTARY_DUE_TO_NUISANCE_SENSITIVITY"

    result = {
        "audit": "MCC5_LABEL_SEMANTICS_AND_MATCHED_NUISANCE",
        "status": placement,
        "post_target_descriptive": True,
        "changes_confirmatory_result": False,
        "canonical_labels": {"0": "NRF", "1": "IR", "2": "OR"},
        "common_nuisance_families": common_families,
        "matched_runs": len(ordered_ids),
        "matched_class_counts": {LABEL_NAMES[label]: matched_counts[label] for label in (0, 1, 2)},
        "all_family_class_counts": {f"{key[0]}_{key[1]}": value for key, value in sorted(family_counts.items())},
        "matched_macro_f1_deltas": {
            "RAMEF_minus_Local": delta_local.tolist(),
            "RAMEF_minus_Distributed": delta_dist.tolist(),
            "RAMEF_minus_Random63": delta_random.tolist(),
        },
        "physical_only_matched": physical_metric,
        "interpretation_criteria": criteria,
        "limitations": [
            "Criteria are post-hoc and govern manuscript placement, not model selection.",
            "The matched subset may be small because support allocation leaves no query run for some class-family cells.",
            "Runs are not documented as bearing-specimen independent.",
        ],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(OUT / "matched_run_manifest.csv", [metadata[item] for item in ordered_ids])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
