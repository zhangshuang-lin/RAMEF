"""T1: query-free MCC5 target-support calibration and perturbation audit."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from mcc5_confirmatory_data import sha256
from run_mcc5_ramef_confirmatory import (
    SEEDS, balanced_source_weights, load_model, load_npz,
    measurement_embeddings,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "staged" / "MCC5_RAMEF_CONFIRMATORY"
OUT = ROOT / "outputs" / "staged" / "MCC5_TRANSFERRED_EVIDENCE_PACKAGE" / "T1_support"
DEVELOPMENT = BASE / "cache" / "development.npz"
PROTOCOL = ROOT / "MCC5_TRANSFERRED_EVIDENCE_PACKAGE_PROTOCOL.md"
WEIGHTS = (1.0, 5.0, 10.0, 20.0)
VIEWS = ("U2-Hz-DM8", "RAMEF")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall_NRF": float(recall[0]), "recall_IR": float(recall[1]), "recall_OR": float(recall[2]),
        "errors": int(np.sum(y != pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def fit_predict(train_x, train_y, test_x, seed, weights=None):
    scaler = StandardScaler().fit(train_x)
    head = LogisticRegression(C=.1, solver="lbfgs", max_iter=2000, random_state=seed)
    head.fit(scaler.transform(train_x), train_y, sample_weight=weights)
    return head.predict(scaler.transform(test_x))


def prototype_predict(train_x, train_y, test_x):
    scaler = StandardScaler().fit(train_x)
    z = scaler.transform(train_x); q = scaler.transform(test_x)
    centres = np.stack([z[train_y == label].mean(0) for label in (0, 1, 2)])
    return ((q[:, None] - centres[None]) ** 2).mean(2).argmin(1)


def representation(name, carrier, physical):
    return carrier if name == "U2-Hz-DM8" else np.hstack([carrier, physical])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = load_npz(DEVELOPMENT)
    if bool(cache["query_cache"]) or np.any(cache["roles"] == "target_query"):
        raise RuntimeError("T1 development cache contains query")
    source = np.flatnonzero(cache["roles"] == "source")
    validation = np.flatnonzero(cache["roles"] == "source_validation")
    support = np.flatnonzero(cache["roles"] == "target_support")
    if (len(source), len(validation), len(support)) != (42, 42, 15):
        raise RuntimeError("Frozen MCC5 role sizes changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows, prediction_rows, fold_rows = [], [], []
    embedding_payload = {"ids": cache["ids"], "roles": cache["roles"], "labels": cache["labels"]}

    for seed in SEEDS:
        checkpoint = BASE / "run" / "checkpoints" / f"u2hz_seed{seed}.pt"
        model = load_model(checkpoint, device)
        block_z = measurement_embeddings(model, cache["distributed"], cache["rpm_distributed"], device)
        carrier = block_z.mean(1)
        embedding_payload[f"carrier_seed{seed}"] = carrier.astype(np.float32)
        source_y = cache["labels"][source]; support_y = cache["labels"][support]
        src_w = balanced_source_weights(source_y)

        for view in VIEWS:
            x = representation(view, carrier, cache["physical"])
            sx, vx, tx = x[source], x[validation], x[support]

            conditions: dict[str, np.ndarray] = {}
            conditions["source_only"] = fit_predict(sx, source_y, tx, seed, src_w)
            for weight in WEIGHTS:
                preds = np.empty(len(support), dtype=int)
                for held in range(len(support)):
                    keep = np.arange(len(support)) != held
                    train_x = np.vstack([sx, tx[keep]])
                    train_y = np.concatenate([source_y, support_y[keep]])
                    train_w = np.concatenate([src_w, np.full(keep.sum(), weight)])
                    preds[held] = fit_predict(train_x, train_y, tx[[held]], seed + held, train_w)[0]
                conditions[f"source_support_w{weight:g}_loo"] = preds

            support_only = np.empty(len(support), dtype=int)
            support_proto = np.empty(len(support), dtype=int)
            for held in range(len(support)):
                keep = np.arange(len(support)) != held
                support_only[held] = fit_predict(tx[keep], support_y[keep], tx[[held]], seed + 100 + held)[0]
                support_proto[held] = prototype_predict(tx[keep], support_y[keep], tx[[held]])[0]
            conditions["support_only_logistic_loo"] = support_only
            conditions["support_only_prototype_loo"] = support_proto

            for condition, pred in conditions.items():
                value = metric(support_y, pred)
                summary_rows.append({"seed": seed, "view": view, "condition": condition,
                                     **{k: value[k] for k in value if k != "confusion_matrix"},
                                     "confusion_matrix": json.dumps(value["confusion_matrix"])})
                for pos, prediction in enumerate(pred):
                    prediction_rows.append({"seed": seed, "view": view, "condition": condition,
                                            "support_id": str(cache["ids"][support[pos]]),
                                            "true": int(support_y[pos]), "prediction": int(prediction),
                                            "correct": int(prediction == support_y[pos])})

            # Registered full support head evaluated only on source validation.
            full_x = np.vstack([sx, tx]); full_y = np.concatenate([source_y, support_y])
            full_w = np.concatenate([src_w, np.full(len(support), 10.0)])
            val_pred = fit_predict(full_x, full_y, vx, seed, full_w)
            val_value = metric(cache["labels"][validation], val_pred)
            summary_rows.append({"seed": seed, "view": view, "condition": "source_validation_registered_w10",
                                 **{k: val_value[k] for k in val_value if k != "confusion_matrix"},
                                 "confusion_matrix": json.dumps(val_value["confusion_matrix"])})

            # Exhaustive leave-two-support-out, registered weight 10.
            for fold, pair in enumerate(itertools.combinations(range(len(support)), 2)):
                held = np.asarray(pair); keep = np.ones(len(support), dtype=bool); keep[held] = False
                train_x = np.vstack([sx, tx[keep]])
                train_y = np.concatenate([source_y, support_y[keep]])
                train_w = np.concatenate([src_w, np.full(keep.sum(), 10.0)])
                pred = fit_predict(train_x, train_y, tx[held], seed + 1000 + fold, train_w)
                for local, pos in enumerate(held):
                    fold_rows.append({"seed": seed, "view": view, "fold": fold,
                                      "support_id": str(cache["ids"][support[pos]]),
                                      "true": int(support_y[pos]), "prediction": int(pred[local]),
                                      "correct": int(pred[local] == support_y[pos])})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"T1 seed={seed} complete", flush=True)

    np.savez_compressed(OUT / "development_embeddings.npz", **embedding_payload)
    write_csv(OUT / "support_loo_metrics.csv", summary_rows)
    write_csv(OUT / "support_loo_predictions.csv", prediction_rows)
    write_csv(OUT / "leave_two_out_predictions.csv", fold_rows)

    # Aggregate leave-two-out metrics and paired transitions.
    aggregate_rows, paired_rows = [], []
    for seed in SEEDS:
        by_view = {}
        for view in VIEWS:
            rows = [row for row in fold_rows if row["seed"] == seed and row["view"] == view]
            y = np.asarray([row["true"] for row in rows]); pred = np.asarray([row["prediction"] for row in rows])
            by_view[view] = (y, pred)
            value = metric(y, pred)
            aggregate_rows.append({"seed": seed, "view": view, **{k: value[k] for k in value if k != "confusion_matrix"},
                                   "confusion_matrix": json.dumps(value["confusion_matrix"])})
        y, base = by_view["U2-Hz-DM8"]; _, candidate = by_view["RAMEF"]
        paired_rows.append({"seed": seed, "corrected": int(np.sum((base != y) & (candidate == y))),
                            "introduced": int(np.sum((base == y) & (candidate != y))),
                            "both_correct": int(np.sum((base == y) & (candidate == y))),
                            "both_wrong": int(np.sum((base != y) & (candidate != y)))})
    write_csv(OUT / "leave_two_out_metrics.csv", aggregate_rows)
    write_csv(OUT / "leave_two_out_paired.csv", paired_rows)

    result = {
        "experiment": "MCC5_T1_QUERY_FREE_SUPPORT_STABILITY",
        "status": "COMPLETE_QUERY_FREE",
        "query_cache_opened": False,
        "seeds": list(SEEDS), "views": list(VIEWS), "support_weights": list(WEIGHTS),
        "source_runs": len(source), "source_validation_runs": len(validation), "support_runs": len(support),
        "leave_two_out_folds_per_seed": 105,
        "protocol_sha256": sha256(PROTOCOL), "development_cache_sha256": sha256(DEVELOPMENT),
        "checkpoint_sha256": {str(seed): sha256(BASE / "run" / "checkpoints" / f"u2hz_seed{seed}.pt") for seed in SEEDS},
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
