"""Post-hoc direct ablation of the PU support-calibrated decision head.

The frozen 20260825 encoder, measurement identities, block features and
physical descriptors are unchanged. Only the data used to fit the final head
vary. Target query has already been evaluated historically; this audit is
component attribution and cannot select a model or hyperparameter.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from bottleneck_deconvolution import sha256
from prepare_splits import scan_records
from run_a1_u2hz_confirmation import CHECKPOINT, CHECKPOINT_SHA
from run_mc_sbt_p0 import SOURCE_DOMAINS


ROOT = Path(__file__).resolve().parent
SOURCE_CACHE = ROOT / "outputs/staged/RACP_OR_INFORMATION_P0/authorised_source_support_features.npz"
QUERY_CACHE = ROOT / "outputs/staged/RACP_MEASUREMENT_FINAL/query_features/authorised_source_support_features.npz"
LOCKED_PRED = ROOT / "outputs/staged/RACP_MEASUREMENT_FINAL/predictions.csv"
FAIR_PRED = ROOT / "outputs/staged/RACP_FAIR_ABLATION/predictions.csv"
SPLIT = ROOT / "split_independent_measurement.json"
OUT = ROOT / "outputs/staged/SUPPORT_CALIBRATION_DIRECT_ABLATION"
VIEWS = ("U2-Hz-M8", "RAMEF-F")
HEADS = ("H0_source_only", "H1_source_plus_support_w1", "H2_registered_w10",
         "H3_support_only", "H4_support_prototype")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def split_rows() -> tuple[list[dict], list[dict], list[dict]]:
    source = sorted([row for row in scan_records(Path("D:/Paderborn"))
                     if row["domain"] in SOURCE_DOMAINS], key=lambda row: row["measurement_id"])
    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    support = [row for row in payload["windows"] if row.get("domain") == "DS" and row.get("role") == "support"]
    query = [row for row in payload["windows"] if row.get("domain") == "DS" and row.get("role") == "query"]
    if len(source) != 1020 or len(support) != 15 or len(query) != 240:
        raise RuntimeError("Frozen cohort size mismatch")
    if {row["measurement_id"] for row in support} & {row["measurement_id"] for row in query}:
        raise RuntimeError("Support/query measurement leakage")
    return source, support, query


def representations(cache: dict) -> dict[str, np.ndarray]:
    carrier = cache["blocks"].astype(np.float32).mean(axis=1)
    physical = cache["physical"].astype(np.float32)
    return {"U2-Hz-M8": carrier, "RAMEF-F": np.hstack([carrier, physical])}


def logistic(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray,
             weights: np.ndarray | None = None) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    head = LogisticRegression(C=.1, solver="lbfgs", max_iter=4000, random_state=20260825)
    head.fit(scaler.transform(train_x), train_y, sample_weight=weights)
    return head.predict(scaler.transform(query_x))


def prototype(support_x: np.ndarray, support_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(support_x)
    support_z = scaler.transform(support_x); query_z = scaler.transform(query_x)
    centres = np.stack([support_z[support_y == label].mean(axis=0) for label in (0, 1, 2)])
    distance = ((query_z[:, None, :] - centres[None, :, :]) ** 2).mean(axis=2)
    return distance.argmin(axis=1)


def metric(truth: np.ndarray, pred: np.ndarray, rows: list[dict]) -> dict:
    recall = recall_score(truth, pred, labels=[0, 1, 2], average=None, zero_division=0)
    by_code = {}
    codes = [row["bearing_code"] for row in rows]
    for code in sorted(set(codes)):
        indices = np.asarray([index for index, value in enumerate(codes) if value == code])
        by_code[code] = float(np.mean(truth[indices] == pred[indices]))
    return {"accuracy": float(np.mean(truth == pred)),
            "macro_f1": float(f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_N": float(recall[0]), "recall_IR": float(recall[1]), "recall_OR": float(recall[2]),
            "errors": int(np.sum(truth != pred)),
            "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
            "worst_code_accuracy": float(min(by_code.values())), "code_accuracy": by_code}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(Path(CHECKPOINT)) != CHECKPOINT_SHA:
        raise RuntimeError("Frozen 20260825 checkpoint mismatch")
    source_rows, support_rows, query_rows = split_rows()
    source_cache = np.load(SOURCE_CACHE, allow_pickle=False)
    query_cache = np.load(QUERY_CACHE, allow_pickle=False)
    authorised = source_rows + support_rows
    if source_cache["ids"].tolist() != [row["measurement_id"] for row in authorised]:
        raise RuntimeError("Source/support cache identity mismatch")
    if query_cache["ids"].tolist() != [row["measurement_id"] for row in query_rows]:
        raise RuntimeError("Query cache identity mismatch")
    if source_cache["checkpoint_sha"].item() != CHECKPOINT_SHA or query_cache["checkpoint_sha"].item() != CHECKPOINT_SHA:
        raise RuntimeError("Feature cache checkpoint mismatch")

    source_views = representations(source_cache); query_views = representations(query_cache)
    ns = len(source_rows)
    source_y = np.asarray([int(row["label"]) for row in source_rows])
    support_y = np.asarray([int(row["label"]) for row in support_rows])
    query_y = np.asarray([int(row["label"]) for row in query_rows])
    predictions: dict[tuple[str, str], np.ndarray] = {}
    result_rows = []
    for view in VIEWS:
        source_x = source_views[view][:ns]
        support_x = source_views[view][ns:]
        query_x = query_views[view]
        predictions[(view, HEADS[0])] = logistic(source_x, source_y, query_x)
        predictions[(view, HEADS[1])] = logistic(
            np.vstack([source_x, support_x]), np.concatenate([source_y, support_y]), query_x)
        predictions[(view, HEADS[2])] = logistic(
            np.vstack([source_x, support_x]), np.concatenate([source_y, support_y]), query_x,
            np.concatenate([np.ones(ns), np.full(len(support_y), 10.0)]))
        predictions[(view, HEADS[3])] = logistic(support_x, support_y, query_x)
        predictions[(view, HEADS[4])] = prototype(support_x, support_y, query_x)
        for head in HEADS:
            value = metric(query_y, predictions[(view, head)], query_rows)
            result_rows.append({"view": view, "head": head,
                                **{key: value[key] for key in ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR", "errors", "worst_code_accuracy")},
                                "confusion_matrix": json.dumps(value["confusion_matrix"]),
                                "code_accuracy": json.dumps(value["code_accuracy"], sort_keys=True)})
    write_csv(OUT / "metrics.csv", result_rows)

    prediction_rows = []
    for index, row in enumerate(query_rows):
        item = {"measurement_id": row["measurement_id"], "bearing_code": row["bearing_code"],
                "true": int(query_y[index])}
        for view in VIEWS:
            for head in HEADS:
                item[f"{view}__{head}"] = int(predictions[(view, head)][index])
        prediction_rows.append(item)
    write_csv(OUT / "predictions.csv", prediction_rows)

    transitions = []
    for view in VIEWS:
        for left, right in ((HEADS[0], HEADS[1]), (HEADS[1], HEADS[2]), (HEADS[0], HEADS[2])):
            a = predictions[(view, left)] == query_y; b = predictions[(view, right)] == query_y
            transitions.append({"view": view, "from": left, "to": right,
                                "corrected": int(np.sum(~a & b)), "introduced": int(np.sum(a & ~b)),
                                "net": int(np.sum(~a & b) - np.sum(a & ~b))})
    write_csv(OUT / "paired_transitions.csv", transitions)

    locked = {row["measurement_id"]: int(row["RACP_MEASUREMENT"]) for row in read_csv(LOCKED_PRED)}
    fair = {row["measurement_id"]: int(row["B2_A1_MEAN8"]) for row in read_csv(FAIR_PRED)}
    h2_ramef = predictions[("RAMEF-F", HEADS[2])]
    h2_carrier = predictions[("U2-Hz-M8", HEADS[2])]
    reproduction = {
        "RAMEF_F_H2_matches_locked_predictions": all(
            int(h2_ramef[i]) == locked[row["measurement_id"]] for i, row in enumerate(query_rows)),
        "U2_Hz_M8_H2_matches_fair_ablation": all(
            int(h2_carrier[i]) == fair[row["measurement_id"]] for i, row in enumerate(query_rows)),
    }
    summaries = {f"{row['view']}__{row['head']}": {key: row[key] for key in
                 ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR", "errors", "worst_code_accuracy")}
                 for row in result_rows}
    result = {
        "experiment": "SUPPORT_CALIBRATION_DIRECT_ABLATION",
        "epistemic_status": "post-hoc component attribution on historically accessed fixed PU query",
        "changes_locked_model": False, "query_used_for_selection": False,
        "views": list(VIEWS), "heads": list(HEADS), "summaries": summaries,
        "paired_transitions": transitions, "reproduction": reproduction,
        "artifacts": {"checkpoint_sha256": CHECKPOINT_SHA,
                      "split_sha256": sha256(SPLIT),
                      "source_support_cache_sha256": sha256(SOURCE_CACHE),
                      "query_cache_sha256": sha256(QUERY_CACHE)},
    }
    if not all(reproduction.values()):
        raise RuntimeError("Registered H2 did not reproduce locked predictions")
    path = OUT / "result.json"; path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
