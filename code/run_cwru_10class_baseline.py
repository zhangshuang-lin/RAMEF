"""Frozen PU-M2 representation audit on CWRU ten-class cross-load episodes."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from bottleneck_deconvolution import CHECKPOINT_SHA256, load_model, sha256
from run_lcga_fresh_p0 import digest_text
from run_mc_sbt_p0 import write_csv


METHODS = ("target_euclidean_proto", "target_diagonal_proto", "source_support_linear")


def prototype_logits(support: torch.Tensor, labels: np.ndarray,
                     query: torch.Tensor, diagonal: bool) -> np.ndarray:
    means, variances = [], []
    y = torch.as_tensor(labels, dtype=torch.long)
    for label in range(10):
        values = support[y == label].float()
        if not len(values): raise RuntimeError(f"Missing support class {label}")
        means.append(values.mean(0)); variances.append(values.var(0, unbiased=False) + .05)
    mean = torch.stack(means); diff = query.float()[:, None] - mean[None]
    distance = ((diff.square() / torch.stack(variances)[None]).mean((2, 3))
                if diagonal else diff.square().mean((2, 3)))
    return (-distance).numpy()


def metric(truth, prediction, classes=10):
    labels = list(range(classes)); recalls = recall_score(truth, prediction, labels=labels,
        average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "recall": [float(x) for x in recalls],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
        "errors": int(np.sum(np.asarray(truth) != np.asarray(prediction))), "n": int(len(truth))}


def coarse_metric(truth, prediction):
    mapping = np.asarray([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    return metric(mapping[np.asarray(truth)], mapping[np.asarray(prediction)], classes=4)


def extract_features(pool, model, device, cache: Path, split_hash: str, batch_size=24):
    ids = [x["window_id"] for x in pool]
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if (payload["checkpoint_sha256"] != CHECKPOINT_SHA256 or
                payload["split_sha256"] != split_hash or payload["window_ids"] != ids):
            raise RuntimeError("Unsafe CWRU feature cache")
        return payload["tokens"]
    signals = {}; tokens = []
    for begin in range(0, len(pool), batch_size):
        rows = pool[begin:begin + batch_size]; waves = []
        for row in rows:
            key = (row["path"], row["signal_key"])
            if key not in signals:
                signals[key] = np.asarray(loadmat(row["path"], variable_names=[row["signal_key"]])[row["signal_key"]]).squeeze()
            raw = np.asarray(signals[key][int(row["raw_start"]):int(row["raw_end"])], dtype=np.float32)
            wave = resample_poly(raw, 16, 3).astype(np.float32)
            if len(raw) != 1536 or len(wave) != 8192: raise RuntimeError("CWRU resampling contract failed")
            wave = (wave - wave.mean()) / max(float(wave.std()), 1e-6); waves.append(wave)
        x = torch.from_numpy(np.stack(waves)).to(device)
        rpm = torch.tensor([float(x["rpm"]) for x in rows], dtype=torch.float32, device=device)
        with torch.no_grad(): tokens.append(model(x, rpm)["tokens"].half().cpu())
    result = torch.cat(tokens)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": result, "window_ids": ids, "checkpoint_sha256": CHECKPOINT_SHA256,
                "split_sha256": split_hash}, cache)
    return result


def record_metric(rows, logits, truth):
    grouped = defaultdict(list)
    for index, row in enumerate(rows): grouped[row["record_id"]].append(index)
    y, pred = [], []
    for record, indices in sorted(grouped.items()):
        y.append(int(truth[indices[0]])); pred.append(int(logits[indices].mean(0).argmax()))
    return metric(np.asarray(y), np.asarray(pred), 10)


def aggregate(episode_rows, method):
    chosen = [x for x in episode_rows if x["method"] == method]
    output = {"episodes": len(chosen)}
    for key in ("accuracy", "macro_f1", "coarse_accuracy", "coarse_macro_f1", "record_accuracy"):
        values = np.asarray([x[key] for x in chosen]); output[key + "_mean"] = float(values.mean()); output[key + "_std"] = float(values.std(ddof=1))
    output["ten_class_recall_mean"] = np.mean([x["recall"] for x in chosen], axis=0).tolist()
    output["confusion_matrix_sum"] = np.sum([x["confusion_matrix"] for x in chosen], axis=0).tolist()
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("outputs/staged/CWRU_10CLASS_BASELINE"))
    parser.add_argument("--protocol", type=Path, default=Path("CWRU_BASELINE_PROTOCOL.md")); parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "evaluation.lock").exists(): raise RuntimeError("CWRU baseline already locked")
    frozen = args.output / "protocol_frozen.md"
    if frozen.exists() and digest_text(frozen) != digest_text(args.protocol): raise RuntimeError("Protocol mismatch")
    if not frozen.exists(): shutil.copy2(args.protocol, frozen)
    split_path = Path("cwru_10class_splits.json"); pool_path = Path("cwru_10class_window_pool.csv")
    split_hash = sha256(split_path); payload = json.loads(split_path.read_text())
    if payload["protocol"] != "CWRU_10CLASS_CROSS_CONDITION_V1": raise RuntimeError("Split protocol mismatch")
    with pool_path.open(encoding="utf-8", newline="") as stream: pool = list(csv.DictReader(stream))
    index = {x["window_id"]: i for i, x in enumerate(pool)}
    checkpoint = Path("frozen/M2_LOCKED_v1_20260715/model/checkpoints/best.pt"); config = Path("frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")
    if sha256(checkpoint) != CHECKPOINT_SHA256: raise RuntimeError("M2 mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, _ = load_model(config, checkpoint, device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    features = extract_features(pool, model, device, args.output / "m2_tokens.pt", split_hash, args.batch_size)
    labels = np.asarray([int(x["label"]) for x in pool]); episode_rows, prediction_rows = [], []
    for episode in payload["episodes"]:
        si = np.asarray([index[x] for x in episode["source_window_ids"]]); ti = np.asarray([index[x] for x in episode["support_window_ids"]]); qi = np.asarray([index[x] for x in episode["query_window_ids"]])
        sy, ty, qy = labels[si], labels[ti], labels[qi]
        logits = {"target_euclidean_proto": prototype_logits(features[ti], ty, features[qi], False),
                  "target_diagonal_proto": prototype_logits(features[ti], ty, features[qi], True)}
        train = np.concatenate([si, ti]); train_y = labels[train]
        flat = features.flatten(1).float().numpy(); scaler = StandardScaler().fit(flat[train])
        linear = LogisticRegression(C=.1, max_iter=4000, solver="lbfgs", random_state=20260821 + int(episode["seed"]))
        weights = np.concatenate([np.ones(len(si)), np.full(len(ti), 10.)])
        linear.fit(scaler.transform(flat[train]), train_y, sample_weight=weights)
        logits["source_support_linear"] = linear.decision_function(scaler.transform(flat[qi]))
        query_rows = [pool[int(i)] for i in qi]
        for method, values in logits.items():
            pred = values.argmax(1); fine = metric(qy, pred); coarse = coarse_metric(qy, pred); record = record_metric(query_rows, values, qy)
            episode_rows.append({"episode_id": episode["episode_id"], "target_load": episode["target_load"],
                "seed": episode["seed"], "method": method, "accuracy": fine["accuracy"],
                "macro_f1": fine["macro_f1"], "recall": fine["recall"],
                "confusion_matrix": fine["confusion_matrix"], "coarse_accuracy": coarse["accuracy"],
                "coarse_macro_f1": coarse["macro_f1"], "record_accuracy": record["accuracy"]})
            for row, truth, guess in zip(query_rows, qy, pred):
                prediction_rows.append({"episode_id": episode["episode_id"], "target_load": episode["target_load"],
                    "seed": episode["seed"], "method": method, "window_id": row["window_id"],
                    "record_id": row["record_id"], "class_name": row["class_name"],
                    "truth": int(truth), "prediction": int(guess), "correct": int(truth == guess)})
    summaries = {method: aggregate(episode_rows, method) for method in METHODS}
    per_target = {}
    for target in range(4):
        per_target[str(target)] = {method: aggregate([x for x in episode_rows if int(x["target_load"]) == target], method) for method in METHODS}
    result = {"experiment": "CWRU_10CLASS_BASELINE", "split_sha256": split_hash,
        "pool_sha256": sha256(pool_path), "checkpoint_sha256": CHECKPOINT_SHA256,
        "query_used_for_selection": False, "summaries": summaries, "per_target": per_target,
        "epistemic_status": "frozen first-pass representation audit"}
    write_csv(args.output / "episode_metrics.csv", episode_rows); write_csv(args.output / "predictions.csv", prediction_rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "evaluation.lock").write_text(json.dumps({"completed": True, "split_sha256": split_hash,
        "result_sha256": sha256(args.output / "result.json")}, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__": main()
