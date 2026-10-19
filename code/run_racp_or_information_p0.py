"""Run the firewalled RACP-OR information audit.

No DS query row or waveform is loaded.  Target labels are used only for the 15
authorised support measurements in leave-one-out evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from bottleneck_deconvolution import load_vibration, sha256
from prepare_splits import scan_records
from racp_or import BLOCK4, WINDOW, measurement_descriptor, uniform_nonoverlap_starts
from run_a1_u2hz_confirmation import CHECKPOINT, CHECKPOINT_SHA, embed
from run_core_component_ablation import make_u2
from run_mc_sbt_p0 import SOURCE_DOMAINS


CONDITIONS = ("A1_SINGLE", "A1_MEAN4", "A1_MEAN8", "A1_MEAN8_RACP")
SEEDS = (20260891, 20260892, 20260893, 20260894, 20260895)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_support_only(split: Path) -> tuple[list[dict], int]:
    """Read the locked file but materialise only authorised support rows."""
    payload = json.loads(split.read_text(encoding="utf-8"))
    support = [row for row in payload["windows"]
               if row.get("domain") == "DS" and row.get("role") == "support"]
    query_count = sum(row.get("domain") == "DS" and row.get("role") == "query"
                      for row in payload["windows"])
    if len(support) != 15 or query_count != 240:
        raise RuntimeError("Locked split structure mismatch")
    # Query rows are counted for the firewall audit only.  Their IDs, labels,
    # paths, starts and signals are never copied or accessed below.
    return support, query_count


def standardise(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32).copy()
    return (block - block.mean()) / max(float(block.std()), 1e-6)


def extract_cache(rows: list[dict], output: Path, device: torch.device) -> dict:
    cache = output / "authorised_source_support_features.npz"
    ids = [row["measurement_id"] for row in rows]
    if cache.exists():
        payload = np.load(cache, allow_pickle=False)
        if payload["ids"].tolist() != ids or payload["checkpoint_sha"].item() != CHECKPOINT_SHA:
            raise RuntimeError("Stale RACP feature cache")
        return {key: payload[key] for key in payload.files}

    model = make_u2("hz", "frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state"]); model.to(device).eval()
    block_features, centre_features, physical = [], [], []
    batch_records = 6
    for group_start in range(0, len(rows), batch_records):
        group = rows[group_start:group_start + batch_records]
        wave_batch, physical_batch = [], []
        for row in group:
            signal = np.asarray(load_vibration(row["path"], row.get("signal_key", "signal")),
                                dtype=np.float32).reshape(-1)
            starts = uniform_nonoverlap_starts(len(signal))
            raw = np.stack([signal[start:start + WINDOW] for start in starts])
            centre_start = len(signal) // 2 - WINDOW // 2
            centre = signal[centre_start:centre_start + WINDOW]
            wave_batch.extend([standardise(block) for block in raw])
            wave_batch.append(standardise(centre))
            physical_batch.append(measurement_descriptor(raw, float(row["rpm"])))
        waves = torch.from_numpy(np.stack(wave_batch))
        rpms = torch.tensor([float(row["rpm"]) for row in group for _ in range(9)])
        z = embed(model, waves, rpms, device, batch=18).reshape(len(group), 9, -1)
        block_features.extend(z[:, :8]); centre_features.extend(z[:, 8])
        physical.extend(physical_batch)
        print(f"features {min(group_start + len(group), len(rows))}/{len(rows)}", flush=True)
    payload = {
        "ids": np.asarray(ids), "checkpoint_sha": np.asarray(CHECKPOINT_SHA),
        "blocks": np.asarray(block_features, dtype=np.float16),
        "centre": np.asarray(centre_features, dtype=np.float16),
        "physical": np.asarray(physical, dtype=np.float32),
    }
    np.savez_compressed(cache, **payload)
    return payload


def representations(cache: dict) -> dict[str, np.ndarray]:
    blocks = cache["blocks"].astype(np.float32)
    centre = cache["centre"].astype(np.float32)
    physical = cache["physical"].astype(np.float32)
    mean4 = blocks[:, BLOCK4].mean(axis=1)
    mean8 = blocks.mean(axis=1)
    return {
        "A1_SINGLE": centre,
        "A1_MEAN4": mean4,
        "A1_MEAN8": mean8,
        "A1_MEAN8_RACP": np.concatenate([mean8, physical], axis=1),
    }


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                seed: int, weights: np.ndarray | None = None) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    head = LogisticRegression(C=.1, max_iter=1500, solver="lbfgs", random_state=seed)
    head.fit(scaler.transform(train_x), train_y, sample_weight=weights)
    return head.predict(scaler.transform(test_x))


def stratified_keep(rows: list[dict], eligible: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); groups: dict[tuple, list[int]] = defaultdict(list)
    for index in eligible:
        row = rows[int(index)]
        groups[(row["domain"], int(row["label"]), row["bearing_code"])].append(int(index))
    keep = []
    for indices in groups.values():
        indices = np.asarray(indices)
        count = max(1, int(np.floor(.9 * len(indices))))
        keep.extend(rng.choice(indices, count, replace=False).tolist())
    return np.asarray(sorted(keep), dtype=np.int64)


def metric(y: np.ndarray, pred: np.ndarray, codes: list[str]) -> dict:
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    code_accuracy = {}
    for code in sorted(set(codes)):
        idx = np.asarray([i for i, value in enumerate(codes) if value == code])
        code_accuracy[code] = float(np.mean(pred[idx] == y[idx]))
    or_values = [value for code, value in code_accuracy.items()
                 if int(y[codes.index(code)]) == 2]
    return {
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro",
                                   zero_division=0)),
        "recall_N": float(recalls[0]), "recall_IR": float(recalls[1]),
        "recall_OR": float(recalls[2]),
        "worst_OR_code_accuracy": float(min(or_values)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "code_accuracy": code_accuracy,
    }


def source_loco(rows: list[dict], x: np.ndarray, condition: str,
                seeds: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    y = np.asarray([int(row["label"]) for row in rows]); codes = [row["bearing_code"] for row in rows]
    unique_codes = sorted(set(codes)); metrics, predictions = [], []
    for seed in seeds:
        truth_all, pred_all, code_all, id_all = [], [], [], []
        for held_code in unique_codes:
            test = np.asarray([i for i, code in enumerate(codes) if code == held_code])
            eligible = np.asarray([i for i, code in enumerate(codes) if code != held_code])
            train = stratified_keep(rows, eligible, seed)
            pred = fit_predict(x[train], y[train], x[test], seed)
            truth_all.extend(y[test]); pred_all.extend(pred); code_all.extend([held_code] * len(test))
            id_all.extend([rows[i]["measurement_id"] for i in test])
        truth_np, pred_np = np.asarray(truth_all), np.asarray(pred_all)
        value = metric(truth_np, pred_np, code_all); value.update({"condition": condition, "seed": seed})
        metrics.append(value)
        predictions.extend({"condition": condition, "seed": seed, "measurement_id": mid,
                            "bearing_code": code, "true": int(truth), "prediction": int(pred),
                            "correct": int(truth == pred)}
                           for mid, code, truth, pred in zip(id_all, code_all, truth_np, pred_np))
    return metrics, predictions


def support_loo(source_rows: list[dict], support_rows: list[dict], source_x: np.ndarray,
                support_x: np.ndarray, condition: str) -> tuple[dict, list[dict]]:
    sy = np.asarray([int(row["label"]) for row in source_rows])
    ty = np.asarray([int(row["label"]) for row in support_rows])
    predictions = []
    for held in range(len(support_rows)):
        keep = np.asarray([i for i in range(len(support_rows)) if i != held])
        train_x = np.vstack([source_x, support_x[keep]])
        train_y = np.concatenate([sy, ty[keep]])
        weights = np.concatenate([np.ones(len(source_x)), np.full(len(keep), 10.)])
        pred = int(fit_predict(train_x, train_y, support_x[held:held + 1],
                               20260910 + held, weights)[0])
        row = support_rows[held]
        predictions.append({"condition": condition, "measurement_id": row["measurement_id"],
                            "bearing_code": row["bearing_code"], "true": int(ty[held]),
                            "prediction": pred, "correct": int(pred == ty[held])})
    truth = ty; pred = np.asarray([row["prediction"] for row in predictions])
    return metric(truth, pred, [row["bearing_code"] for row in support_rows]), predictions


def aggregate(values: list[dict]) -> dict:
    keys = ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR",
            "worst_OR_code_accuracy")
    return {key: {"mean": float(np.mean([row[key] for row in values])),
                  "std": (float(np.std([row[key] for row in values], ddof=1))
                          if len(values) > 1 else 0.0)}
            for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/staged/RACP_OR_INFORMATION_P0"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("RACP_OR_INFORMATION_P0_PROTOCOL.md"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if sha256(Path(CHECKPOINT)) != CHECKPOINT_SHA:
        raise RuntimeError("Frozen A1 hash mismatch")
    source = sorted([row for row in scan_records(Path("D:/Paderborn"))
                     if row["domain"] in SOURCE_DOMAINS], key=lambda row: row["measurement_id"])
    support, query_count = load_support_only(Path("split_independent_measurement.json"))
    authorised = source + support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = extract_cache(authorised, args.output, device); views = representations(cache)
    ns = len(source)

    source_metrics, source_predictions, support_metrics, support_predictions = [], [], {}, []
    for condition in CONDITIONS:
        seeds = SEEDS if condition in ("A1_SINGLE", "A1_MEAN8_RACP") else (SEEDS[0],)
        values, rows = source_loco(source, views[condition][:ns], condition, seeds)
        source_metrics.extend(values); source_predictions.extend(rows)
        support_value, support_rows = support_loo(
            source, support, views[condition][:ns], views[condition][ns:], condition)
        support_metrics[condition] = support_value; support_predictions.extend(support_rows)
        print(condition, aggregate(values), support_value, flush=True)

    base_rows = [row for row in source_predictions if row["condition"] == "A1_SINGLE"]
    candidate_rows = [row for row in source_predictions if row["condition"] == "A1_MEAN8_RACP"]
    per_seed_net = []
    for seed in SEEDS:
        base = {row["measurement_id"]: row for row in base_rows if row["seed"] == seed}
        cand = {row["measurement_id"]: row for row in candidate_rows if row["seed"] == seed}
        fixes = sum(not base[mid]["correct"] and cand[mid]["correct"] for mid in base)
        new = sum(base[mid]["correct"] and not cand[mid]["correct"] for mid in base)
        per_seed_net.append({"seed": seed, "fixes": fixes, "new_errors": new,
                             "net": fixes - new})

    source_summary = {}
    for condition in CONDITIONS:
        source_summary[condition] = aggregate(
            [row for row in source_metrics if row["condition"] == condition])
    base, cand = source_summary["A1_SINGLE"], source_summary["A1_MEAN8_RACP"]
    support_base, support_cand = support_metrics["A1_SINGLE"], support_metrics["A1_MEAN8_RACP"]
    sb = {row["measurement_id"]: row for row in support_predictions
          if row["condition"] == "A1_SINGLE"}
    sc = {row["measurement_id"]: row for row in support_predictions
          if row["condition"] == "A1_MEAN8_RACP"}
    ka30_corrections = sum(row["bearing_code"] == "KA30" and not row["correct"] and
                           sc[mid]["correct"] for mid, row in sb.items())
    n_to_or = sum(row["true"] == 0 and row["prediction"] == 2
                  for row in support_predictions if row["condition"] == "A1_MEAN8_RACP")
    gates = {
        "source_accuracy_not_lower": cand["accuracy"]["mean"] >= base["accuracy"]["mean"] - 1e-12,
        "source_N_recall_not_lower": cand["recall_N"]["mean"] >= base["recall_N"]["mean"] - 1e-12,
        "source_OR_recall_gain_ge_5pp": cand["recall_OR"]["mean"] - base["recall_OR"]["mean"] >= .05 - 1e-12,
        "source_worst_OR_not_lower": cand["worst_OR_code_accuracy"]["mean"] >= base["worst_OR_code_accuracy"]["mean"] - 1e-12,
        "source_net_positive_at_least_4of5": sum(row["net"] > 0 for row in per_seed_net) >= 4,
        "support_zero_N_to_OR": n_to_or == 0,
        "support_corrects_KA30": ka30_corrections >= 1,
        "support_OR_not_lower": support_cand["recall_OR"] >= support_base["recall_OR"] - 1e-12,
        "support_accuracy_not_lower": support_cand["accuracy"] >= support_base["accuracy"] - 1e-12,
    }
    result = {
        "experiment": "RACP_OR_INFORMATION_P0", "status": "PASS" if all(gates.values()) else "FAIL",
        "failure_type": (None if all(gates.values()) else
                         "FAIL_TO_QUALIFY_DUE_TO_SOURCE_CEILING"),
        "trainable_RACP_authorised": bool(all(gates.values())),
        "epistemic_status": "source and DS-support-only information audit",
        "protocol_sha256": sha256(args.protocol), "checkpoint_sha256": CHECKPOINT_SHA,
        "split_sha256": sha256(Path("split_independent_measurement.json")),
        "firewall": {"ds_query_rows_in_split": query_count, "ds_query_measurements_loaded": 0,
                     "ds_query_waveforms_loaded": False, "ds_query_labels_used": False},
        "source_summary": source_summary, "support_loo": support_metrics,
        "source_candidate_net_by_seed": per_seed_net,
        "support_KA30_corrections": ka30_corrections, "support_N_to_OR": n_to_or,
        "gates": gates,
    }
    write_csv(args.output / "source_metrics.csv", source_metrics)
    write_csv(args.output / "source_predictions.csv", source_predictions)
    write_csv(args.output / "support_predictions.csv", support_predictions)
    path = args.output / "result.json"; path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
