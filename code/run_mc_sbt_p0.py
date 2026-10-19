"""Run MC-SBT P0 with source-only selection and a gated target confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from bottleneck_deconvolution import (CHECKPOINT_SHA256, SPLIT_SHA256,
                                      TokenExtractor, load_contract, load_model,
                                      ordinary_logits, sha256)
from mc_sbt import BoundaryConfig, coarse_safe_prediction, fit_boundary
from prepare_splits import scan_records


CLASS_NAMES = ("N", "IR", "OR")
SOURCE_DOMAINS = ("D0", "DM", "DF")


def digest_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metric(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    recall = recall_score(truth, prediction, labels=[0, 1, 2], average=None,
                          zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=[0, 1, 2],
                                    average="macro", zero_division=0)),
        "n_recall": float(recall[0]), "ir_recall": float(recall[1]),
        "or_recall": float(recall[2]),
        "confusion_matrix": confusion_matrix(truth, prediction,
                                               labels=[0, 1, 2]).tolist(),
        "n": int(len(truth)), "errors": int(np.sum(truth != prediction)),
    }


def view_item(record: dict, view: int, length: int) -> dict:
    fractions = (0.15, 0.50, 0.85)
    if view not in range(len(fractions)):
        raise ValueError(view)
    available = int(record.get("signal_length", 256001)) - length
    start = int(round(max(0, available) * fractions[view]))
    return {**record, "role": "mc_sbt_view", "view": view,
            "start": start, "end": start + length}


def round_robin_order(records: list[dict], seed: int) -> list[dict]:
    """Deterministic code-diverse order without using windows as samples."""
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["bearing_code"]].append(row)
    for code in grouped:
        order = rng.permutation(len(grouped[code]))
        grouped[code] = [grouped[code][int(i)] for i in order]
    codes = sorted(grouped)
    codes = [codes[int(i)] for i in rng.permutation(len(codes))]
    output = []
    while True:
        advanced = False
        for code in codes:
            if grouped[code]:
                output.append(grouped[code].pop()); advanced = True
        if not advanced:
            return output


def episode_pools(records: list[dict], domain: str, seed: int,
                  support_count: int, adaptation_count: int,
                  query_count: int) -> tuple[list[dict], list[dict], list[dict]]:
    support, adaptation, query = [], [], []
    for label in range(3):
        candidates = [x for x in records if x["domain"] == domain and int(x["label"]) == label]
        ordered = round_robin_order(candidates, seed * 101 + label * 17 + 7)
        required = support_count + adaptation_count + query_count
        if len(ordered) < required:
            raise RuntimeError(f"{domain}/{label}: need {required} measurements, got {len(ordered)}")
        support.extend(ordered[:support_count])
        adaptation.extend(ordered[support_count:support_count + adaptation_count])
        query.extend(ordered[support_count + adaptation_count:required])
    sets = [{x["measurement_id"] for x in values}
            for values in (support, adaptation, query)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Episode measurement leakage")
    return support, adaptation, query


class RecordFeatureBank:
    def __init__(self, records: list[dict], tokens: torch.Tensor):
        self.records = records
        self.tokens = tokens.flatten(2).numpy()  # [record, view, feature]
        self.index = {row["measurement_id"]: index for index, row in enumerate(records)}

    def centre(self, records: list[dict]) -> np.ndarray:
        return np.stack([self.tokens[self.index[x["measurement_id"]], 1] for x in records])

    def views(self, records: list[dict]) -> np.ndarray:
        return np.stack([self.tokens[self.index[x["measurement_id"]]] for x in records])


def extract_source_bank(records: list[dict], model, device, length: int,
                        output: Path) -> RecordFeatureBank:
    source_records = sorted((x for x in records if x["domain"] in SOURCE_DOMAINS),
                            key=lambda x: x["measurement_id"])
    cache = output / f"m2_source_record_views_{length}.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        expected = [x["measurement_id"] for x in source_records]
        if payload["checkpoint_sha256"] != CHECKPOINT_SHA256 or payload["measurement_ids"] != expected:
            raise RuntimeError("Unsafe or stale source feature cache")
        tokens = payload["tokens"]
    else:
        items = [view_item(row, view, length) for row in source_records for view in range(3)]
        tokens, bounds = TokenExtractor(model, device).extract(items, length=length, batch_size=24)
        tokens = tokens.reshape(len(source_records), 3, *tokens.shape[1:])
        torch.save({"tokens": tokens, "bounds": bounds,
                    "measurement_ids": [x["measurement_id"] for x in source_records],
                    "checkpoint_sha256": CHECKPOINT_SHA256}, cache)
    return RecordFeatureBank(source_records, tokens)


def ordinary(model, support: np.ndarray, support_labels: np.ndarray,
             query: np.ndarray, device: torch.device) -> np.ndarray:
    # Restore [sample, token, channel] expected by the M2 prototype module.
    token_count = 8
    support_t = torch.from_numpy(support.reshape(len(support), token_count, -1)).to(device)
    query_t = torch.from_numpy(query.reshape(len(query), token_count, -1)).to(device)
    labels_t = torch.from_numpy(support_labels).to(device)
    return ordinary_logits(model, support_t, labels_t, query_t).cpu().numpy()


def evaluate_source_episode(records: list[dict], bank: RecordFeatureBank, model,
                            device: torch.device, pseudo_domain: str, seed: int,
                            hyper: dict, data_cfg: dict) -> dict:
    support, adaptation, query = episode_pools(
        records, pseudo_domain, seed, data_cfg["source_support_per_class"],
        data_cfg["source_adaptation_per_class"], data_cfg["source_query_per_class"])
    labelled_source = [x for x in records if x["domain"] in SOURCE_DOMAINS
                       and x["domain"] != pseudo_domain]
    sx, sy = bank.centre(labelled_source), np.asarray([int(x["label"]) for x in labelled_source])
    tx, ty = bank.centre(support), np.asarray([int(x["label"]) for x in support])
    ux = bank.views(adaptation)
    uy = np.asarray([int(x["label"]) for x in adaptation])
    qx, qy = bank.centre(query), np.asarray([int(x["label"]) for x in query])
    base_logits = ordinary(model, tx, ty, qx, device)
    base_prediction = base_logits.argmax(1)
    boundary = fit_boundary(sx, sy, tx, ty, ux,
        BoundaryConfig(c=hyper["c"], support_weight=hyper["support_weight"],
                       pseudo_weight=hyper["pseudo_weight"], confidence=hyper["confidence"],
                       iterations=hyper["iterations"], seed=seed),
        unlabeled_audit_labels=uy)
    logits = boundary.decision_function(qx)
    prediction = coarse_safe_prediction(logits, base_logits)
    return {"pseudo_domain": pseudo_domain, "seed": seed,
            "baseline": metric(qy, base_prediction), "mc_sbt": metric(qy, prediction),
            "boundary_audit": boundary.audit,
            "support_measurements": [x["measurement_id"] for x in support],
            "adaptation_measurements": [x["measurement_id"] for x in adaptation],
            "query_measurements": [x["measurement_id"] for x in query]}


def aggregate_episodes(episodes: list[dict]) -> dict:
    fields = ("accuracy", "macro_f1", "n_recall", "ir_recall", "or_recall")
    result = {}
    for method in ("baseline", "mc_sbt"):
        result[method] = {field: float(np.mean([x[method][field] for x in episodes]))
                          for field in fields}
        result[method]["errors"] = int(sum(x[method]["errors"] for x in episodes))
        result[method]["n"] = int(sum(x[method]["n"] for x in episodes))
    result["delta"] = {field: result["mc_sbt"][field] - result["baseline"][field]
                       for field in fields}
    return result


def seed_pass(summary: dict, confirmation_cfg: dict) -> bool:
    tolerances = {
        "accuracy": confirmation_cfg["accuracy_tolerance"],
        "macro_f1": confirmation_cfg["macro_f1_tolerance"],
        "n_recall": confirmation_cfg["recall_tolerance"],
        "ir_recall": confirmation_cfg["recall_tolerance"],
        "or_recall": confirmation_cfg["recall_tolerance"],
    }
    return all(summary["mc_sbt"][field] + float(tolerances[field]) >=
               summary["baseline"][field] for field in tolerances)


def target_adaptation_records(records: list[dict], fixed_support: list[dict],
                              fixed_query: list[dict], per_class: int, seed: int) -> list[dict]:
    forbidden = {x["measurement_id"] for x in fixed_support + fixed_query}
    selected = []
    for label in range(3):
        candidates = [x for x in records if x["domain"] == "DS" and int(x["label"]) == label
                      and x["measurement_id"] not in forbidden]
        ordered = round_robin_order(candidates, seed + label * 31)
        if len(ordered) < per_class:
            raise RuntimeError(f"Only {len(ordered)} unused DS records for class {label}")
        selected.extend(ordered[:per_class])
    ids = {x["measurement_id"] for x in selected}
    if ids & forbidden or len(ids) != 3 * per_class:
        raise RuntimeError("Target support/adaptation/test measurement leakage")
    return selected


def extract_items(items: list[dict], model, device, length: int) -> np.ndarray:
    tokens, _ = TokenExtractor(model, device).extract(items, length=length, batch_size=24)
    return tokens.flatten(1).numpy()


def evaluate_target(config: dict, records: list[dict], source_bank: RecordFeatureBank,
                    model, device: torch.device, hyper: dict, output: Path) -> dict:
    _, support, query = load_contract(Path(config["data"]["split"]))
    adaptation = target_adaptation_records(
        records, support, query, int(config["data"]["target_adaptation_per_class"]),
        int(config["seed"]))
    support_ids = {x["measurement_id"] for x in support}
    adaptation_ids = {x["measurement_id"] for x in adaptation}
    query_ids = {x["measurement_id"] for x in query}
    if support_ids & adaptation_ids or support_ids & query_ids or adaptation_ids & query_ids:
        raise RuntimeError("Target measurement sets are not mutually disjoint")

    length = int(config["data"]["window_length"])
    # Fit the candidate before any target-test waveform is loaded.
    source_records = [x for x in records if x["domain"] in SOURCE_DOMAINS]
    sx = source_bank.centre(source_records)
    sy = np.asarray([int(x["label"]) for x in source_records])
    tx = extract_items(support, model, device, length)
    ty = np.asarray([int(x["label"]) for x in support])
    adaptation_items = [view_item(row, view, length) for row in adaptation for view in range(3)]
    ux = extract_items(adaptation_items, model, device, length).reshape(len(adaptation), 3, -1)
    boundary = fit_boundary(sx, sy, tx, ty, ux,
        BoundaryConfig(c=hyper["c"], support_weight=hyper["support_weight"],
                       pseudo_weight=hyper["pseudo_weight"], confidence=hyper["confidence"],
                       iterations=hyper["iterations"], seed=int(config["seed"])))

    # Candidate-specific target confirmation: query extraction occurs once,
    # after source gates and the complete classifier have been fixed.
    qx = extract_items(query, model, device, length)
    qy = np.asarray([int(x["label"]) for x in query])
    base_logits = ordinary(model, tx, ty, qx, device)
    base_prediction = base_logits.argmax(1)
    logits = boundary.decision_function(qx)
    prediction = coarse_safe_prediction(logits, base_logits)
    base_metrics, proposed_metrics = metric(qy, base_prediction), metric(qy, prediction)
    rows = []
    for item, truth, base, pred, score in zip(query, qy, base_prediction, prediction, logits):
        rows.append({"measurement_id": item["measurement_id"],
                     "bearing_code": item["bearing_code"], "start": item["start"],
                     "end": item["end"], "truth": CLASS_NAMES[int(truth)],
                     "m2_prediction": CLASS_NAMES[int(base)],
                     "mc_sbt_prediction": CLASS_NAMES[int(pred)],
                     "mc_sbt_correct": int(pred == truth),
                     "logit_n": float(score[0]), "logit_ir": float(score[1]),
                     "logit_or": float(score[2])})
    write_csv(output / "target_predictions.csv", rows)
    manifest = [{"role": "support", **x} for x in support]
    manifest += [{"role": "unlabeled_adaptation", **x} for x in adaptation]
    manifest += [{"role": "test", **x} for x in query]
    write_csv(output / "target_protocol_manifest.csv", [{
        "role": x["role"], "measurement_id": x["measurement_id"],
        "bearing_code": x["bearing_code"], "label_audit_only": x["label"],
        "path": x["path"], "start": x.get("start", ""), "end": x.get("end", "")}
        for x in manifest])
    thresholds = config["target"]
    passed = (proposed_metrics["accuracy"] >= thresholds["minimum_accuracy"] and
              proposed_metrics["macro_f1"] >= thresholds["minimum_macro_f1"] and
              proposed_metrics["n_recall"] >= thresholds["minimum_n_recall"] and
              proposed_metrics["ir_recall"] >= thresholds["minimum_ir_recall"] and
              proposed_metrics["or_recall"] >= thresholds["minimum_or_recall"])
    return {"baseline": base_metrics, "mc_sbt": proposed_metrics,
            "boundary_audit": boundary.audit, "passed_target_thresholds": bool(passed),
            "measurement_disjoint": True, "query_uses_this_candidate": 1,
            "support_measurements": len(support_ids),
            "adaptation_measurements": len(adaptation_ids),
            "test_measurements": len(query_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_mc_sbt_p0.yaml")
    parser.add_argument("--protocol", default="MC_SBT_P0_PROTOCOL.md")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-target", action="store_true")
    args = parser.parse_args()
    config_path, protocol_path = Path(args.config), Path(args.protocol)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(args.output or config["output"]); output.mkdir(parents=True, exist_ok=True)
    frozen_config, frozen_protocol = output / "config_frozen.yaml", output / "protocol_frozen.md"
    for source, frozen in ((config_path, frozen_config), (protocol_path, frozen_protocol)):
        if frozen.exists() and digest_text(frozen) != digest_text(source):
            raise RuntimeError(f"Frozen experiment contract changed: {source}")
        if not frozen.exists(): shutil.copyfile(source, frozen)
    if sha256(config["model"]["checkpoint"]) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("M2 checkpoint hash mismatch")
    if sha256(config["data"]["split"]) != SPLIT_SHA256:
        raise RuntimeError("Independent split hash mismatch")

    records = scan_records(Path(config["data"]["root"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(config["model"]["config"]),
                          Path(config["model"]["checkpoint"]), device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_hash = sha256(config["model"]["checkpoint"])
    bank = extract_source_bank(records, model, device,
                               int(config["data"]["window_length"]), output)

    candidates = []
    for c, support_weight, confidence in itertools.product(
            config["selection"]["c_candidates"],
            config["selection"]["support_weight_candidates"],
            config["selection"]["confidence_candidates"]):
        hyper = {"c": float(c), "support_weight": float(support_weight),
                 "confidence": float(confidence),
                 "pseudo_weight": float(config["selection"]["pseudo_weight"]),
                 "iterations": int(config["selection"]["iterations"])}
        episodes = [evaluate_source_episode(records, bank, model, device, domain, int(seed),
                                             hyper, config["data"])
                    for seed in config["selection"]["seeds"] for domain in SOURCE_DOMAINS]
        summary = aggregate_episodes(episodes)
        # Source-only ranking prioritises balanced IR/OR recovery, then overall
        # accuracy.  Simpler regularisation wins exact ties.
        score = (summary["mc_sbt"]["macro_f1"] + summary["mc_sbt"]["accuracy"] +
                 min(summary["mc_sbt"]["ir_recall"], summary["mc_sbt"]["or_recall"]))
        candidates.append({"hyperparameters": hyper, "score": float(score),
                           "summary": summary, "episodes": episodes})
    chosen = max(candidates, key=lambda x: (x["score"], x["hyperparameters"]["confidence"],
                                            -x["hyperparameters"]["c"],
                                            -x["hyperparameters"]["support_weight"]))
    (output / "source_selection.json").write_text(json.dumps({
        "selection_uses_target": False, "selected": chosen,
        "candidates": candidates}, indent=2), encoding="utf-8")

    confirmation_rows, seed_reports = [], []
    for seed in config["confirmation"]["seeds"]:
        episodes = [evaluate_source_episode(records, bank, model, device, domain, int(seed),
                                             chosen["hyperparameters"], config["data"])
                    for domain in SOURCE_DOMAINS]
        summary = aggregate_episodes(episodes)
        passed = seed_pass(summary, config["confirmation"])
        seed_reports.append({"seed": int(seed), "passed": bool(passed),
                             "summary": summary, "episodes": episodes})
        confirmation_rows.append({"seed": int(seed), "passed": int(passed),
            **{f"baseline_{k}": v for k, v in summary["baseline"].items() if k != "confusion_matrix"},
            **{f"mc_sbt_{k}": v for k, v in summary["mc_sbt"].items() if k != "confusion_matrix"}})
    passing = sum(int(x["passed"]) for x in seed_reports)
    authorised = passing >= int(config["confirmation"]["minimum_passing_seeds"])
    confirmation = {"selected_hyperparameters": chosen["hyperparameters"],
                    "passing_seeds": passing, "required_passing_seeds": int(
                        config["confirmation"]["minimum_passing_seeds"]),
                    "target_evaluation_authorized": bool(authorised),
                    "seed_reports": seed_reports,
                    "selection_uses_target": False,
                    "checkpoint_sha256_before": before_hash,
                    "checkpoint_sha256_after": sha256(config["model"]["checkpoint"]),
                    "m2_parameters_frozen": all(not p.requires_grad for p in model.parameters())}
    # load_model does not alter requires_grad, so explicitly audit values rather
    # than claiming the default flags.  No optimiser or backward pass exists.
    confirmation["m2_state_unchanged"] = before_hash == confirmation["checkpoint_sha256_after"]
    write_csv(output / "source_confirmation.csv", confirmation_rows)
    (output / "source_confirmation.json").write_text(json.dumps(confirmation, indent=2),
                                                       encoding="utf-8")

    target = None
    if authorised and bool(config["target"]["evaluate_if_unlocked"]) and not args.skip_target:
        target = evaluate_target(config, records, bank, model, device,
                                 chosen["hyperparameters"], output)
        lock = {"candidate": "MC_SBT_P0", "protocol_sha256": digest_text(frozen_protocol),
                "config_sha256": digest_text(frozen_config),
                "checkpoint_sha256": before_hash, "query_evaluations": 1,
                "historical_query_exposure_disclosed": True,
                "metrics_file": "target_result.json"}
        (output / "target_evaluation.lock").write_text(json.dumps(lock, indent=2),
                                                        encoding="utf-8")
        (output / "target_result.json").write_text(json.dumps(target, indent=2), encoding="utf-8")
    if target is None:
        decision_name = "STOP_WITHOUT_TARGET_EVALUATION"
    elif target["passed_target_thresholds"]:
        decision_name = "PASS_MC_SBT_P0"
    else:
        decision_name = "STOP_TARGET_THRESHOLD_FAILED"
    decision = {"experiment": "MC_SBT_P0", "source_gate_passed": bool(authorised),
                "passing_seeds": passing, "selected_hyperparameters": chosen["hyperparameters"],
                "target_evaluated": target is not None, "target_result": target,
                "decision": decision_name}
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
