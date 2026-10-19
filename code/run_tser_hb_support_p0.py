"""Nested 900-rpm support-only audit for TSER-HB.

The runner consumes the authorised feature cache created by
RACP_OR_INFORMATION_P0.  It never loads a DS query record or waveform.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from bottleneck_deconvolution import sha256
from prepare_splits import scan_records
from run_mc_sbt_p0 import SOURCE_DOMAINS
from tser_hb import choose_kappa_nested, predict as tser_predict


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def support_only(split: Path) -> tuple[list[dict], int]:
    payload = json.loads(split.read_text(encoding="utf-8"))
    rows = [row for row in payload["windows"]
            if row.get("domain") == "DS" and row.get("role") == "support"]
    query_count = sum(row.get("domain") == "DS" and row.get("role") == "query"
                      for row in payload["windows"])
    if len(rows) != 15 or query_count != 240:
        raise RuntimeError("Locked split mismatch")
    return rows, query_count


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(y == pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_N": float(recall[0]), "recall_IR": float(recall[1]),
            "recall_OR": float(recall[2]),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "N_to_OR": int(np.sum((y == 0) & (pred == 2)))}


def linear_predict(source_x: np.ndarray, source_y: np.ndarray,
                   support_x: np.ndarray, support_y: np.ndarray,
                   query_x: np.ndarray, seed: int) -> np.ndarray:
    x = np.vstack([source_x, support_x]); y = np.concatenate([source_y, support_y])
    scaler = StandardScaler().fit(x)
    head = LogisticRegression(C=.1, max_iter=1500, solver="lbfgs", random_state=seed)
    weights = np.concatenate([np.ones(len(source_x)), np.full(len(support_x), 10.)])
    head.fit(scaler.transform(x), y, sample_weight=weights)
    return head.predict(scaler.transform(query_x))


def build_evidence(blocks: np.ndarray, physical: np.ndarray, source_y: np.ndarray,
                   source_count: int) -> tuple[np.ndarray, dict]:
    """Source-only feature construction; target support cannot select dimensions."""
    mean8 = blocks.mean(axis=1)
    scaler = StandardScaler().fit(mean8[:source_count])
    head = LogisticRegression(C=.1, max_iter=1500, solver="lbfgs", random_state=20260920)
    head.fit(scaler.transform(mean8[:source_count]), source_y)
    flat = blocks.reshape(-1, blocks.shape[-1])
    block_logits = head.decision_function(scaler.transform(flat)).reshape(len(blocks), 8, 3)
    summaries = np.concatenate([
        block_logits.mean(1), np.median(block_logits, axis=1),
        block_logits.std(1), block_logits.max(1) - block_logits.min(1),
    ], axis=1)
    selector = SelectKBest(f_classif, k=12).fit(physical[:source_count], source_y)
    selected = selector.transform(physical)
    evidence = np.concatenate([summaries, selected], axis=1)
    evidence_scaler = StandardScaler().fit(evidence[:source_count])
    evidence = evidence_scaler.transform(evidence).astype(np.float32)
    meta = {"dimensions": int(evidence.shape[1]),
            "physical_indices": selector.get_support(indices=True).tolist()}
    return evidence, meta


def outer_loo(source_y, support, views, evidence):
    ty = np.asarray([int(row["label"]) for row in support]); n = len(ty)
    methods = {"A1_SINGLE": [], "A1_MEAN8_RACP_LINEAR": [], "TSER_HB": []}
    detail, selections = [], []
    for held in range(n):
        keep = np.asarray([i for i in range(n) if i != held])
        methods["A1_SINGLE"].append(int(linear_predict(
            views["single_source"], source_y, views["single_support"][keep], ty[keep],
            views["single_support"][held:held + 1], 20260930 + held)[0]))
        methods["A1_MEAN8_RACP_LINEAR"].append(int(linear_predict(
            views["racp_source"], source_y, views["racp_support"][keep], ty[keep],
            views["racp_support"][held:held + 1], 20261030 + held)[0]))
        kappa, inner = choose_kappa_nested(evidence["source"], source_y,
                                           evidence["support"][keep], ty[keep])
        guess = int(tser_predict(evidence["source"], source_y,
                                 evidence["support"][keep], ty[keep],
                                 evidence["support"][held:held + 1], kappa)[0])
        methods["TSER_HB"].append(guess)
        selections.append({"held_measurement_id": support[held]["measurement_id"],
                           "held_code": support[held]["bearing_code"], "kappa": kappa,
                           "inner_scores": json.dumps(inner)})
    for method, guesses in methods.items():
        for row, truth, guess in zip(support, ty, guesses):
            detail.append({"validation": "LOO", "method": method,
                           "measurement_id": row["measurement_id"], "bearing_code": row["bearing_code"],
                           "true": int(truth), "prediction": int(guess), "correct": int(truth == guess)})
    return {method: metric(ty, np.asarray(guess)) for method, guess in methods.items()}, detail, selections


def leave_two_out(source_y, support, views, evidence):
    ty = np.asarray([int(row["label"]) for row in support]); details, folds = [], []
    for fold, pair in enumerate(itertools.combinations(range(len(ty)), 2)):
        held = np.asarray(pair); keep = np.asarray([i for i in range(len(ty)) if i not in pair])
        base = linear_predict(views["single_source"], source_y,
                              views["single_support"][keep], ty[keep],
                              views["single_support"][held], 20262000 + fold)
        kappa, _ = choose_kappa_nested(evidence["source"], source_y,
                                       evidence["support"][keep], ty[keep])
        candidate = tser_predict(evidence["source"], source_y,
                                 evidence["support"][keep], ty[keep],
                                 evidence["support"][held], kappa)
        base_acc = float(np.mean(base == ty[held])); candidate_acc = float(np.mean(candidate == ty[held]))
        folds.append({"fold": fold, "held": "+".join(support[i]["measurement_id"] for i in held),
                      "kappa": kappa, "base_accuracy": base_acc,
                      "candidate_accuracy": candidate_acc,
                      "candidate_not_lower": int(candidate_acc >= base_acc)})
        for index, b, c in zip(held, base, candidate):
            details.append({"validation": "L2O", "fold": fold,
                            "measurement_id": support[index]["measurement_id"],
                            "bearing_code": support[index]["bearing_code"], "true": int(ty[index]),
                            "A1_SINGLE": int(b), "TSER_HB": int(c)})
    truth = np.asarray([row["true"] for row in details]); candidate = np.asarray([row["TSER_HB"] for row in details])
    return metric(truth, candidate), folds, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/staged/TSER_HB_SUPPORT_P0"))
    parser.add_argument("--protocol", type=Path, default=Path("TSER_HB_SUPPORT_P0_PROTOCOL.md"))
    parser.add_argument("--cache", type=Path, default=Path(
        "outputs/staged/RACP_OR_INFORMATION_P0/authorised_source_support_features.npz"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = sorted([row for row in scan_records(Path("D:/Paderborn"))
                     if row["domain"] in SOURCE_DOMAINS], key=lambda row: row["measurement_id"])
    support, query_count = support_only(Path("split_independent_measurement.json"))
    cache = np.load(args.cache, allow_pickle=False); ids = [row["measurement_id"] for row in source + support]
    if cache["ids"].tolist() != ids:
        raise RuntimeError("Authorised cache order mismatch")
    ns = len(source); source_y = np.asarray([int(row["label"]) for row in source])
    blocks = cache["blocks"].astype(np.float32); centre = cache["centre"].astype(np.float32)
    physical = cache["physical"].astype(np.float32); mean8 = blocks.mean(1)
    racp = np.concatenate([mean8, physical], axis=1)
    views = {"single_source": centre[:ns], "single_support": centre[ns:],
             "racp_source": racp[:ns], "racp_support": racp[ns:]}
    evidence_all, evidence_meta = build_evidence(blocks, physical, source_y, ns)
    evidence = {"source": evidence_all[:ns], "support": evidence_all[ns:]}
    loo, loo_detail, selections = outer_loo(source_y, support, views, evidence)
    l2o, folds, l2o_detail = leave_two_out(source_y, support, views, evidence)
    base_errors = {row["measurement_id"] for row in loo_detail
                   if row["method"] == "A1_SINGLE" and not row["correct"] and row["bearing_code"] == "KA30"}
    candidate = {row["measurement_id"]: row for row in loo_detail if row["method"] == "TSER_HB"}
    ka30_fixed = sum(candidate[mid]["correct"] for mid in base_errors)
    nonlower = float(np.mean([row["candidate_not_lower"] for row in folds]))
    gates = {
        "loo_accuracy_ge_9333": loo["TSER_HB"]["accuracy"] >= .9333,
        "loo_macro_f1_ge_93": loo["TSER_HB"]["macro_f1"] >= .93,
        "loo_OR_recall_ge_80": loo["TSER_HB"]["recall_OR"] >= .8,
        "loo_N_recall_100": loo["TSER_HB"]["recall_N"] == 1.0,
        "loo_zero_N_to_OR": loo["TSER_HB"]["N_to_OR"] == 0,
        "loo_corrects_KA30": ka30_fixed >= 1,
        "loo_not_below_RACP_linear": loo["TSER_HB"]["accuracy"] >= loo["A1_MEAN8_RACP_LINEAR"]["accuracy"],
        "l2o_accuracy_ge_90": l2o["accuracy"] >= .9,
        "l2o_N_recall_100": l2o["recall_N"] == 1.0,
        "l2o_OR_recall_ge_80": l2o["recall_OR"] >= .8,
        "l2o_nonlower_folds_ge_80pct": nonlower >= .8,
    }
    result = {"experiment": "TSER_HB_SUPPORT_P0", "status": "PASS" if all(gates.values()) else "FAIL",
              "query_candidate_freeze_authorised": bool(all(gates.values())),
              "epistemic_status": "900-rpm support-only nested validation",
              "firewall": {"query_rows_in_split": query_count, "query_measurements_loaded": 0,
                           "query_waveforms_loaded": False, "query_labels_used": False},
              "protocol_sha256": sha256(args.protocol), "source_cache_sha256": sha256(args.cache),
              "evidence": evidence_meta, "loo": loo, "leave_two_out": l2o,
              "leave_two_out_nonlower_fold_rate": nonlower, "KA30_corrections": ka30_fixed,
              "kappa_selection_counts": {str(k): sum(row["kappa"] == k for row in selections)
                                         for k in (0.0, .25, 1.0, 4.0)},
              "gates": gates}
    write_csv(args.output / "loo_predictions.csv", loo_detail)
    write_csv(args.output / "loo_kappa_selections.csv", selections)
    write_csv(args.output / "leave_two_out_folds.csv", folds)
    write_csv(args.output / "leave_two_out_predictions.csv", l2o_detail)
    path = args.output / "result.json"; path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
