"""Retrospective ten-episode LCGA-S stability analysis on PAPER_MAIN_V1.

The target query is historically exposed. This runner therefore estimates
support-set stability only and cannot create a new confirmatory claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from bottleneck_deconvolution import TokenExtractor, load_model
from mc_sbt import (BoundaryConfig, calibrate_margin_threshold, fit_boundary,
                    margin_gated_prediction)
from prepare_splits import scan_records
from run_c2of import metrics
from run_lcga_fresh_p0 import digest_text
from run_mc_sbt_p0 import (SOURCE_DOMAINS, extract_source_bank, ordinary,
                           write_csv)
from run_paper_main import (CHECKPOINT_SHA256, EPISODES,
                            PROTOCOL_MANIFEST_SHA256, WINDOW_LENGTH,
                            load_contract, sha256)


FIXED_BOUNDARY = {
    "c": 0.1,
    "support_weight": 10.0,
    "pseudo_weight": 0.0,
    "confidence": 1.0,
    "iterations": 0,
}


def empty_unlabelled(feature_dim: int) -> np.ndarray:
    """No target-unlabelled cohort is permitted by PAPER_MAIN."""
    return np.empty((0, 3, feature_dim), dtype=np.float64)


def boundary_config(seed: int) -> BoundaryConfig:
    return BoundaryConfig(**FIXED_BOUNDARY, seed=int(seed) % (2**32 - 1))


def fit_support_only(source_x: np.ndarray, source_y: np.ndarray,
                     support_x: np.ndarray, support_y: np.ndarray,
                     seed: int):
    return fit_boundary(source_x, source_y, support_x, support_y,
                        empty_unlabelled(source_x.shape[1]),
                        boundary_config(seed))


def calibrate_support_only(model, device: torch.device,
                           source_x: np.ndarray, source_y: np.ndarray,
                           support_x: np.ndarray, support_y: np.ndarray,
                           seed: int) -> tuple[float, dict]:
    """Nested support LOO calibration; this function has no query argument."""
    boundary_rows, base_rows = [], []
    for left in range(len(support_x)):
        keep = np.arange(len(support_x)) != left
        boundary = fit_support_only(source_x, source_y, support_x[keep],
                                    support_y[keep], seed + left + 1)
        boundary_rows.append(boundary.decision_function(support_x[left:left + 1])[0])
        base_rows.append(ordinary(model, support_x[keep], support_y[keep],
                                  support_x[left:left + 1], device)[0])
    return calibrate_margin_threshold(np.asarray(boundary_rows),
                                      np.asarray(base_rows), support_y)


def paired_hierarchical_bootstrap(delta_correct: np.ndarray, repeats: int,
                                  seed: int = 20260816) -> dict:
    rng = np.random.default_rng(seed)
    episode_count, query_count = delta_correct.shape
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        episodes = rng.integers(0, episode_count, episode_count)
        measurements = rng.integers(0, query_count, query_count)
        values[index] = delta_correct[np.ix_(episodes, measurements)].mean()
    low, high = np.quantile(values, [0.025, 0.975])
    return {"repeats": repeats, "mean": float(values.mean()),
            "ci_low": float(low), "ci_high": float(high), "seed": seed}


def reference_predictions(path: Path, episode: int) -> dict[str, int]:
    result = {}
    with (path / f"predictions_episode_{episode:02d}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            result[row["measurement_id"]] = int(row["prediction"])
    return result


def aggregate(rows: list[dict], key: str) -> dict:
    fields = ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR",
              "worst_fault_code_accuracy")
    output = {}
    for field in fields:
        values = np.asarray([row[key][field] for row in rows], dtype=np.float64)
        output[field] = {"mean": float(values.mean()),
                         "std": float(values.std(ddof=1)),
                         "min": float(values.min()), "max": float(values.max())}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path,
                        default=Path("paper/evidence/protocol/PAPER_MAIN_V1"))
    parser.add_argument("--method-protocol", type=Path,
                        default=Path("LCGA_PAPER_MAIN_V1_PROTOCOL.md"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/staged/LCGA_PAPER_MAIN_V1"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()

    checkpoint = Path("frozen/M2_LOCKED_v1_20260715/model/checkpoints/best.pt")
    config = Path("frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen M2 checkpoint mismatch")
    if sha256(args.protocol_dir / "MANIFEST.sha256") != PROTOCOL_MANIFEST_SHA256:
        raise RuntimeError("PAPER_MAIN protocol mismatch")
    query, episode_supports, split_hashes = load_contract(args.protocol_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_method = args.output / "method_protocol_frozen.md"
    if frozen_method.exists() and digest_text(frozen_method) != digest_text(args.method_protocol):
        raise RuntimeError("Frozen LCGA-S method protocol changed")
    if not frozen_method.exists():
        frozen_method.write_bytes(args.method_protocol.read_bytes())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(config, checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = scan_records(Path("D:/Paderborn"))
    bank = extract_source_bank(records, model, device, WINDOW_LENGTH,
                               Path("outputs/staged/MC_SBT_P0"))
    source_records = [row for row in records if row["domain"] in SOURCE_DOMAINS]
    source_x = bank.centre(source_records)
    source_y = np.asarray([int(row["label"]) for row in source_records])

    support_by_id = {}
    for support in episode_supports:
        for row in support:
            support_by_id.setdefault(row["measurement_id"], row)
    unique_support = sorted(support_by_id.values(), key=lambda row: row["measurement_id"])
    extractor = TokenExtractor(model, device)
    query_tokens, _ = extractor.extract(query, WINDOW_LENGTH, args.batch_size)
    support_tokens, _ = extractor.extract(unique_support, WINDOW_LENGTH, args.batch_size)
    query_x = query_tokens.flatten(1).numpy()
    support_x_by_id = {row["measurement_id"]: support_tokens[index].flatten().numpy()
                       for index, row in enumerate(unique_support)}

    truth = np.asarray([int(row["label"]) for row in query])
    codes = [row["bearing_code"] for row in query]
    episode_rows, all_predictions = [], []
    base_correct, candidate_correct = [], []
    for episode, support in enumerate(episode_supports):
        tx = np.stack([support_x_by_id[row["measurement_id"]] for row in support])
        ty = np.asarray([int(row["label"]) for row in support])
        base_logits = ordinary(model, tx, ty, query_x, device)
        base_prediction = base_logits.argmax(1)
        expected = reference_predictions(Path("outputs/staged/PAPER_MAIN_V1_B2"), episode)
        mismatches = [row["measurement_id"] for row, pred in zip(query, base_prediction)
                      if expected.get(row["measurement_id"]) != int(pred)]
        if mismatches:
            raise RuntimeError(f"Episode {episode} baseline reproduction failed: {mismatches[:5]}")

        threshold, calibration = calibrate_support_only(
            model, device, source_x, source_y, tx, ty, 20260816 + episode * 1000)
        boundary = fit_support_only(source_x, source_y, tx, ty, 20260816 + episode)
        boundary_logits = boundary.decision_function(query_x)
        prediction = margin_gated_prediction(boundary_logits, base_logits, threshold)
        base_metric = metrics(truth, base_prediction, codes)
        candidate_metric = metrics(truth, prediction, codes)
        for value, name in ((base_metric, "base"), (candidate_metric, "candidate")):
            value["recall_N"], value["recall_IR"], value["recall_OR"] = value["recall"]
        fixes = int(np.sum((base_prediction != truth) & (prediction == truth)))
        new_errors = int(np.sum((base_prediction == truth) & (prediction != truth)))
        changes = int(np.sum(base_prediction != prediction))
        episode_rows.append({"episode": episode, "split_sha256": split_hashes[episode],
            "threshold": threshold, "support_loo": calibration["chosen"],
            "base": base_metric, "candidate": candidate_metric,
            "changes": changes, "fixes": fixes, "new_errors": new_errors,
            "baseline_reproduction": True})
        base_correct.append(base_prediction == truth)
        candidate_correct.append(prediction == truth)
        for item, target, base, pred in zip(query, truth, base_prediction, prediction):
            all_predictions.append({"episode": episode,
                "measurement_id": item["measurement_id"],
                "bearing_code": item["bearing_code"], "true": int(target),
                "m2_prediction": int(base), "lcga_s_prediction": int(pred),
                "m2_correct": int(base == target), "lcga_s_correct": int(pred == target)})

    base_correct_np = np.stack(base_correct)
    candidate_correct_np = np.stack(candidate_correct)
    noninferior = sum(row["candidate"]["accuracy"] >= row["base"]["accuracy"]
                      for row in episode_rows)
    base_aggregate = aggregate(episode_rows, "base")
    candidate_aggregate = aggregate(episode_rows, "candidate")
    targets = {
        "mean_accuracy_ge_90": candidate_aggregate["accuracy"]["mean"] >= 0.90,
        "mean_macro_f1_ge_895": candidate_aggregate["macro_f1"]["mean"] >= 0.895,
        "mean_n_recall_eq_100": candidate_aggregate["recall_N"]["mean"] == 1.0,
        "mean_ir_recall_ge_85": candidate_aggregate["recall_IR"]["mean"] >= 0.85,
        "mean_or_recall_ge_85": candidate_aggregate["recall_OR"]["mean"] >= 0.85,
        "noninferior_episodes_ge_7": noninferior >= 7,
    }
    summary = {
        "status": "RETROSPECTIVE_STABILITY_COMPLETE",
        "method": "LCGA-S_support_only",
        "epistemic_status": "retrospective; query historically exposed",
        "query_used_for_selection": False,
        "additional_target_unlabelled_used": False,
        "checkpoint_sha256": sha256(checkpoint),
        "method_protocol_sha256": sha256(frozen_method),
        "episodes": EPISODES, "query_measurements": len(query),
        "unique_support_measurements": len(unique_support),
        "fixed_boundary": FIXED_BOUNDARY,
        "baseline": base_aggregate, "candidate": candidate_aggregate,
        "noninferior_episodes": noninferior,
        "total_fixes": int(sum(row["fixes"] for row in episode_rows)),
        "total_new_errors": int(sum(row["new_errors"] for row in episode_rows)),
        "paired_bootstrap_accuracy_delta": paired_hierarchical_bootstrap(
            candidate_correct_np.astype(float) - base_correct_np.astype(float),
            args.bootstrap_repeats),
        "success_criteria": targets,
        "all_success_criteria_passed": all(targets.values()),
        "episode_results": episode_rows,
    }
    write_csv(args.output / "episode_metrics.csv", [{
        "episode": row["episode"], "threshold": row["threshold"],
        "base_accuracy": row["base"]["accuracy"],
        "lcga_s_accuracy": row["candidate"]["accuracy"],
        "base_macro_f1": row["base"]["macro_f1"],
        "lcga_s_macro_f1": row["candidate"]["macro_f1"],
        "base_ir_recall": row["base"]["recall_IR"],
        "lcga_s_ir_recall": row["candidate"]["recall_IR"],
        "base_or_recall": row["base"]["recall_OR"],
        "lcga_s_or_recall": row["candidate"]["recall_OR"],
        "changes": row["changes"], "fixes": row["fixes"],
        "new_errors": row["new_errors"]} for row in episode_rows])
    write_csv(args.output / "predictions.csv", all_predictions)
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artefacts = sorted(path for path in args.output.iterdir()
                       if path.is_file() and path.name != "RESULTS_FREEZE.sha256")
    with (args.output / "RESULTS_FREEZE.sha256").open("w", encoding="utf-8", newline="\n") as stream:
        for path in artefacts:
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps({"status": summary["status"],
        "baseline": base_aggregate, "candidate": candidate_aggregate,
        "noninferior_episodes": noninferior,
        "fixes": summary["total_fixes"], "new_errors": summary["total_new_errors"],
        "bootstrap": summary["paired_bootstrap_accuracy_delta"],
        "success": summary["all_success_criteria_passed"]}, indent=2))


if __name__ == "__main__":
    main()
