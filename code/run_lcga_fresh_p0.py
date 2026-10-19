"""LCGA-FRESH P0: support-calibrated local arbitration on fresh DS records."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from bottleneck_deconvolution import SPLIT_SHA256, load_contract, load_model, sha256
from mc_sbt import (BoundaryConfig, calibrate_margin_threshold, fit_boundary,
                    margin_gated_prediction)
from prepare_splits import scan_records
from run_mc_sbt_p0 import (SOURCE_DOMAINS, aggregate_episodes, digest_text,
                           episode_pools, extract_items, extract_source_bank,
                           metric, ordinary, round_robin_order,
                           target_adaptation_records, view_item, write_csv)


def boundary_config(config: dict, seed: int) -> BoundaryConfig:
    value = config["boundary"]
    return BoundaryConfig(c=float(value["c"]), support_weight=float(value["support_weight"]),
        confidence=float(value["confidence"]), pseudo_weight=float(value["pseudo_weight"]),
        iterations=int(value["iterations"]), seed=int(seed) % (2**32 - 1))


def calibrate_from_support(model, device, sx, sy, tx, ty, ux,
                           config: dict, seed: int) -> tuple[float, dict]:
    boundary_rows, base_rows = [], []
    for left in range(len(tx)):
        keep = np.arange(len(tx)) != left
        boundary = fit_boundary(sx, sy, tx[keep], ty[keep], ux,
                                boundary_config(config, seed + left + 1))
        boundary_rows.append(boundary.decision_function(tx[left:left + 1])[0])
        base_rows.append(ordinary(model, tx[keep], ty[keep], tx[left:left + 1], device)[0])
    return calibrate_margin_threshold(np.asarray(boundary_rows), np.asarray(base_rows), ty)


def source_episode(records, bank, model, device, domain: str, seed: int,
                   config: dict) -> dict:
    data = config["data"]
    support, adaptation, query = episode_pools(records, domain, seed,
        int(data["source_support_per_class"]), int(data["source_adaptation_per_class"]),
        int(data["source_query_per_class"]))
    source = [x for x in records if x["domain"] in SOURCE_DOMAINS and x["domain"] != domain]
    sx, sy = bank.centre(source), np.asarray([int(x["label"]) for x in source])
    tx, ty = bank.centre(support), np.asarray([int(x["label"]) for x in support])
    ux = bank.views(adaptation)
    qx, qy = bank.centre(query), np.asarray([int(x["label"]) for x in query])
    threshold, support_audit = calibrate_from_support(model, device, sx, sy, tx, ty, ux,
                                                       config, seed * 1000)
    boundary = fit_boundary(sx, sy, tx, ty, ux, boundary_config(config, seed))
    base_logits = ordinary(model, tx, ty, qx, device)
    boundary_logits = boundary.decision_function(qx)
    base_prediction = base_logits.argmax(1)
    prediction = margin_gated_prediction(boundary_logits, base_logits, threshold)
    return {"pseudo_domain": domain, "seed": seed, "threshold": threshold,
            "support_calibration": support_audit, "boundary_audit": boundary.audit,
            "baseline": metric(qy, base_prediction), "mc_sbt": metric(qy, prediction),
            "support_measurements": [x["measurement_id"] for x in support],
            "adaptation_measurements": [x["measurement_id"] for x in adaptation],
            "query_measurements": [x["measurement_id"] for x in query]}


def fresh_target_records(records: list[dict], support: list[dict], old_query: list[dict],
                         adaptation: list[dict], per_class: int, seed: int) -> list[dict]:
    forbidden = {x["measurement_id"] for x in support + old_query + adaptation}
    selected = []
    for label in range(3):
        candidates = [x for x in records if x["domain"] == "DS" and int(x["label"]) == label
                      and x["measurement_id"] not in forbidden]
        ordered = round_robin_order(candidates, seed + label * 47)
        if len(ordered) < per_class:
            raise RuntimeError(f"Fresh DS class {label}: only {len(ordered)}, need {per_class}")
        selected.extend(ordered[:per_class])
    if {x["measurement_id"] for x in selected} & forbidden:
        raise RuntimeError("Fresh target measurement was previously exposed")
    return selected


def target_evaluation(config, records, bank, model, device, output: Path) -> dict:
    _, support, old_query = load_contract(Path(config["data"]["split"]))
    adaptation = target_adaptation_records(records, support, old_query,
        int(config["data"]["target_adaptation_per_class"]),
        int(config["data"]["previous_adaptation_seed"]))
    fresh = fresh_target_records(records, support, old_query, adaptation,
        int(config["data"]["fresh_test_per_class"]), int(config["seed"]))
    # Freeze identities before loading a fresh waveform.
    manifest_rows = []
    for role, values in (("support", support), ("unlabeled_adaptation", adaptation),
                         ("fresh_test", fresh)):
        for row in values:
            manifest_rows.append({"role": role, "measurement_id": row["measurement_id"],
                "bearing_code": row["bearing_code"], "label_audit_only": row["label"],
                "path": row["path"]})
    manifest = output / "fresh_target_manifest.csv"
    if manifest.exists():
        import csv
        with manifest.open(encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
        expected_identity = [(str(x["role"]), str(x["measurement_id"]), str(x["bearing_code"]))
                             for x in manifest_rows]
        existing_identity = [(x["role"], x["measurement_id"], x["bearing_code"])
                             for x in existing]
        if existing_identity != expected_identity or (output / "fresh_target_evaluation.lock").exists():
            raise RuntimeError("Fresh target manifest mismatch or evaluation already completed")
    else:
        write_csv(manifest, manifest_rows)
    manifest_sha = sha256(manifest)

    length = int(config["data"]["window_length"])
    source = [x for x in records if x["domain"] in SOURCE_DOMAINS]
    sx, sy = bank.centre(source), np.asarray([int(x["label"]) for x in source])
    tx = extract_items(support, model, device, length)
    ty = np.asarray([int(x["label"]) for x in support])
    adaptation_items = [view_item(row, view, length) for row in adaptation for view in range(3)]
    ux = extract_items(adaptation_items, model, device, length).reshape(len(adaptation), 3, -1)
    threshold, support_audit = calibrate_from_support(model, device, sx, sy, tx, ty, ux,
                                                       config, int(config["seed"]) * 1000)
    boundary = fit_boundary(sx, sy, tx, ty, ux,
                            boundary_config(config, int(config["seed"])))

    fresh_items = [view_item(row, view, length) for row in fresh for view in range(3)]
    qx = extract_items(fresh_items, model, device, length).reshape(len(fresh), 3, -1)
    qy = np.asarray([int(x["label"]) for x in fresh])
    flat = qx.reshape(-1, qx.shape[-1])
    base_flat = ordinary(model, tx, ty, flat, device).reshape(len(fresh), 3, 3)
    boundary_flat = boundary.decision_function(flat).reshape(len(fresh), 3, 3)
    window_truth = np.repeat(qy, 3)
    window_base = base_flat.reshape(-1, 3).argmax(1)
    window_prediction = margin_gated_prediction(boundary_flat.reshape(-1, 3),
                                                 base_flat.reshape(-1, 3), threshold)
    base_mean, boundary_mean = base_flat.mean(1), boundary_flat.mean(1)
    measurement_base = base_mean.argmax(1)
    measurement_prediction = margin_gated_prediction(boundary_mean, base_mean, threshold)
    rows = []
    for row, truth, base, pred, base_logit, new_logit in zip(
            fresh, qy, measurement_base, measurement_prediction, base_mean, boundary_mean):
        rows.append({"measurement_id": row["measurement_id"],
            "bearing_code": row["bearing_code"], "truth": int(truth),
            "m2_prediction": int(base), "lcga_prediction": int(pred),
            "m2_correct": int(base == truth), "lcga_correct": int(pred == truth),
            "m2_ir_or_margin": float(abs(base_logit[1] - base_logit[2])),
            "boundary_ir_or_margin": float(abs(new_logit[1] - new_logit[2]))})
    write_csv(output / "fresh_target_predictions.csv", rows)
    measurement_metrics = {"baseline": metric(qy, measurement_base),
                           "lcga": metric(qy, measurement_prediction)}
    window_metrics = {"baseline": metric(window_truth, window_base),
                      "lcga": metric(window_truth, window_prediction)}
    target_cfg = config["target"]; m = measurement_metrics["lcga"]
    passed = (m["accuracy"] >= target_cfg["minimum_accuracy"] and
              m["macro_f1"] >= target_cfg["minimum_macro_f1"] and
              m["n_recall"] >= target_cfg["minimum_n_recall"] and
              m["ir_recall"] >= target_cfg["minimum_ir_recall"] and
              m["or_recall"] >= target_cfg["minimum_or_recall"])
    return {"fresh_manifest_sha256": manifest_sha, "threshold": threshold,
            "support_calibration": support_audit, "boundary_audit": boundary.audit,
            "measurement_level": measurement_metrics, "window_level": window_metrics,
            "passed_target_thresholds": bool(passed), "fresh_measurements": len(fresh),
            "fresh_measurements_per_class": int(config["data"]["fresh_test_per_class"]),
            "historical_test_waveforms_used": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_lcga_fresh_p0.yaml")
    parser.add_argument("--protocol", default="LCGA_FRESH_P0_PROTOCOL.md")
    parser.add_argument("--skip-target", action="store_true")
    args = parser.parse_args()
    config_path, protocol_path = Path(args.config), Path(args.protocol)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    frozen_config, frozen_protocol = output / "config_frozen.yaml", output / "protocol_frozen.md"
    for source, frozen in ((config_path, frozen_config), (protocol_path, frozen_protocol)):
        if frozen.exists() and digest_text(frozen) != digest_text(source):
            raise RuntimeError(f"Frozen contract changed: {source}")
        if not frozen.exists(): shutil.copyfile(source, frozen)
    if sha256(config["model"]["checkpoint"]) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("M2 checkpoint mismatch")
    if sha256(config["data"]["split"]) != SPLIT_SHA256:
        raise RuntimeError("Split mismatch")
    records = scan_records(Path(config["data"]["root"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(config["model"]["config"]),
                          Path(config["model"]["checkpoint"]), device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    bank = extract_source_bank(records, model, device,
        int(config["data"]["window_length"]), Path("outputs/staged/MC_SBT_P0"))

    seed_reports = []
    for seed in config["confirmation"]["seeds"]:
        episodes = [source_episode(records, bank, model, device, domain, int(seed), config)
                    for domain in SOURCE_DOMAINS]
        summary = aggregate_episodes(episodes)
        fields = ("accuracy", "macro_f1", "n_recall", "ir_recall", "or_recall")
        passed = all(summary["mc_sbt"][field] >= summary["baseline"][field]
                     for field in fields)
        seed_reports.append({"seed": int(seed), "passed": bool(passed),
                             "summary": summary, "episodes": episodes})
    passing = sum(int(x["passed"]) for x in seed_reports)
    authorised = passing >= int(config["confirmation"]["minimum_passing_seeds"])
    confirmation = {"passing_seeds": passing,
        "required_passing_seeds": int(config["confirmation"]["minimum_passing_seeds"]),
        "target_evaluation_authorized": bool(authorised), "seed_reports": seed_reports,
        "uses_target_test_for_selection": False}
    (output / "source_confirmation.json").write_text(json.dumps(confirmation, indent=2),
                                                       encoding="utf-8")
    target = None
    if authorised and config["target"]["evaluate_if_unlocked"] and not args.skip_target:
        target = target_evaluation(config, records, bank, model, device, output)
        (output / "fresh_target_result.json").write_text(json.dumps(target, indent=2), encoding="utf-8")
        lock = {"candidate": "LCGA_FRESH_P0", "query_evaluations": 1,
                "fresh_manifest_sha256": target["fresh_manifest_sha256"],
                "historical_test_waveforms_used": False,
                "exploratory_after_prior_target_exposure": True}
        (output / "fresh_target_evaluation.lock").write_text(json.dumps(lock, indent=2),
                                                              encoding="utf-8")
    decision = {"experiment": "LCGA_FRESH_P0", "source_gate_passed": bool(authorised),
                "passing_seeds": passing, "target_evaluated": target is not None,
                "target_result": target,
                "decision": ("PASS_LCGA_FRESH_P0" if target and target["passed_target_thresholds"]
                             else "STOP_LCGA_FRESH_P0")}
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__": main()
