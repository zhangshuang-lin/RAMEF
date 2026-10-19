"""T2: frozen post-hoc MCC5 equal-input method benchmark plus ProtoNet."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fair_baseline_models import WDCNNEncoder, make_model
from mcc5_confirmatory_data import sha256
from run_mcc5_ramef_confirmatory import SEEDS, balanced_source_weights, load_npz


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "staged" / "MCC5_RAMEF_CONFIRMATORY"
OUT = ROOT / "outputs" / "staged" / "MCC5_TRANSFERRED_EVIDENCE_PACKAGE" / "T2_benchmark"
DEVELOPMENT = BASE / "cache" / "development.npz"
QUERY = BASE / "cache" / "query_after_selection.npz"
PROTOCOL = ROOT / "MCC5_TRANSFERRED_EVIDENCE_PACKAGE_PROTOCOL.md"
MODELS = ("wdcnn", "resnet1d", "stft_cnn", "wdcnn_dann")
DISPLAY = {"wdcnn": "WDCNN-DM8", "resnet1d": "ResNet1D-DM8",
           "stft_cnn": "STFT-CNN-DM8", "wdcnn_dann": "WDCNN-DANN-DM8"}
WINDOW = 3072


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def standardise(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)


def metric(y, pred):
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(y == pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_NRF": float(recall[0]), "recall_IR": float(recall[1]), "recall_OR": float(recall[2]),
            "errors": int(np.sum(y != pred)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


@torch.no_grad()
def infer_model(model, waves: np.ndarray, device, batch=64):
    model.eval(); flat = torch.from_numpy(waves.reshape(-1, WINDOW)); embeddings, logits = [], []
    start_time = time.perf_counter()
    for start in range(0, len(flat), batch):
        out = model(standardise(flat[start:start + batch]).to(device))
        embeddings.append(out["embedding"].cpu()); logits.append(out["class_logits"].cpu())
    elapsed = time.perf_counter() - start_time
    n = len(waves); return (torch.cat(embeddings).reshape(n, 8, -1).mean(1).numpy(),
                            torch.cat(logits).reshape(n, 8, -1).mean(1).numpy(), elapsed)


def train_conventional(cache, name, seed, checkpoint, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    source = np.flatnonzero(cache["roles"] == "source")
    validation = np.flatnonzero(cache["roles"] == "source_validation")
    waves = torch.from_numpy(cache["distributed"][source].reshape(-1, WINDOW))
    labels = torch.tensor(np.repeat(cache["labels"][source], 8), dtype=torch.long)
    torque = np.repeat(cache["torques"][source], 8)
    domains = torch.tensor((torque == 40).astype(np.int64))
    counts = Counter(labels.tolist())
    weights = torch.tensor([1.0 / counts[int(label)] for label in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=max(counts.values()) * 3, replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(waves, labels, domains), batch_size=12, sampler=sampler, num_workers=0)
    model = make_model(name).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1.; best_state = None; history = []
    for epoch in range(12):
        model.train(); losses = []
        for step, (x, y, domain) in enumerate(loader):
            optimiser.zero_grad(set_to_none=True)
            progress = (epoch * len(loader) + step + 1) / (12 * len(loader))
            grl = .2 * progress if name == "wdcnn_dann" else 0.
            out = model(standardise(x).to(device), grl=grl)
            loss = F.cross_entropy(out["class_logits"], y.to(device))
            if name == "wdcnn_dann":
                loss = loss + .1 * F.cross_entropy(out["domain_logits"], domain.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
            losses.append(float(loss.detach()))
        _, logits, _ = infer_model(model, cache["distributed"][validation], device)
        pred = logits.argmax(1); value = metric(cache["labels"][validation], pred)
        history.append({"seed": seed, "model": name, "epoch": epoch,
                        "train_loss": float(np.mean(losses)), "validation_macro_f1": value["macro_f1"]})
        if value["macro_f1"] > best + 1e-12:
            best = value["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({"state": best_state, "seed": seed, "model": name, "best_val_macro_f1": best}, checkpoint)
    return best, history, sum(p.numel() for p in model.parameters())


def embed_runs(encoder, waves: np.ndarray, device, batch=64):
    encoder.eval(); flat = torch.from_numpy(waves.reshape(-1, WINDOW)); rows = []
    for start in range(0, len(flat), batch):
        rows.append(encoder(standardise(flat[start:start + batch]).to(device)))
    return torch.cat(rows).reshape(len(waves), 8, -1).mean(1)


def proto_logits(z, support_idx, query_idx, labels):
    prototypes = torch.stack([z[support_idx][torch.as_tensor(labels[support_idx] == c)].mean(0) for c in (0, 1, 2)])
    return -((z[query_idx, None] - prototypes[None]) ** 2).mean(2)


def deterministic_task(indices, labels, seed, shots=5):
    rng = np.random.default_rng(seed); support, query = [], []
    for c in (0, 1, 2):
        candidates = np.asarray(indices)[labels[indices] == c].copy(); rng.shuffle(candidates)
        support.extend(candidates[:shots]); query.extend(candidates[shots:])
    return np.asarray(support), np.asarray(query)


def train_protonet(cache, seed, checkpoint, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    source = np.flatnonzero(cache["roles"] == "source")
    validation = np.flatnonzero(cache["roles"] == "source_validation")
    labels = cache["labels"]
    encoder = WDCNNEncoder().to(device)
    optimiser = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    val_support, val_query = deterministic_task(validation, labels, seed + 9000, shots=5)
    best = -1.; best_state = None; history = []
    for epoch in range(12):
        encoder.train(); losses = []
        for episode in range(20):
            rng = np.random.default_rng(seed + epoch * 100 + episode)
            sup, qry = [], []
            for c in (0, 1, 2):
                candidates = source[labels[source] == c].copy(); rng.shuffle(candidates)
                sup.extend(candidates[:5]); qry.extend(candidates[5:8])
            sup = np.asarray(sup); qry = np.asarray(qry); chosen = np.concatenate([sup, qry])
            z = embed_runs(encoder, cache["distributed"][chosen], device)
            local_labels = labels[chosen]; local_sup = np.arange(len(sup)); local_qry = np.arange(len(sup), len(chosen))
            logits = proto_logits(z, local_sup, local_qry, local_labels)
            target = torch.as_tensor(local_labels[local_qry], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
            optimiser.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0); optimiser.step(); losses.append(float(loss.detach()))
        encoder.eval()
        with torch.no_grad():
            chosen = np.concatenate([val_support, val_query])
            z = embed_runs(encoder, cache["distributed"][chosen], device)
            local_labels = labels[chosen]; ls = np.arange(len(val_support)); lq = np.arange(len(val_support), len(chosen))
            pred = proto_logits(z, ls, lq, local_labels).argmax(1).cpu().numpy()
            score = metric(local_labels[lq], pred)["macro_f1"]
        history.append({"seed": seed, "model": "protonet", "epoch": epoch,
                        "train_loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best + 1e-12:
            best = score; best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    torch.save({"state": best_state, "seed": seed, "model": "protonet", "best_val_macro_f1": best}, checkpoint)
    return best, history, sum(p.numel() for p in encoder.parameters())


def train_stage():
    OUT.mkdir(parents=True, exist_ok=True); checkpoints = OUT / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    cache = load_npz(DEVELOPMENT)
    if bool(cache["query_cache"]) or np.any(cache["roles"] == "target_query"):
        raise RuntimeError("Training cache contains target query")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, history = [], []
    for seed in SEEDS:
        for name in MODELS:
            path = checkpoints / f"{name}_seed{seed}.pt"
            best, local, params = train_conventional(cache, name, seed, path, device)
            rows.append({"seed": seed, "model": name, "best_validation_macro_f1": best,
                         "parameters": params, "checkpoint": str(path), "sha256": sha256(path)})
            history.extend(local); print(f"T2 trained {name} seed={seed} val={best:.4f}", flush=True)
        path = checkpoints / f"protonet_seed{seed}.pt"
        best, local, params = train_protonet(cache, seed, path, device)
        rows.append({"seed": seed, "model": "protonet", "best_validation_macro_f1": best,
                     "parameters": params, "checkpoint": str(path), "sha256": sha256(path)})
        history.extend(local); print(f"T2 trained protonet seed={seed} val={best:.4f}", flush=True)
    write_csv(OUT / "checkpoints.csv", rows); write_csv(OUT / "training_history.csv", history)
    selection = {"experiment": "MCC5_T2_FAIR_METHOD_BENCHMARK", "status": "SELECTION_FROZEN",
                 "post_hoc": True, "query_cache_opened": False, "seeds": list(SEEDS),
                 "models": list(MODELS) + ["protonet"], "protocol_sha256": sha256(PROTOCOL),
                 "development_cache_sha256": sha256(DEVELOPMENT), "checkpoints": rows}
    path = OUT / "selection_complete.json"; path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"selection": str(path), "sha256": sha256(path)}, indent=2))


def load_conventional(name, checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(name); model.load_state_dict(payload["state"]); return model.to(device)


def load_proto(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = WDCNNEncoder(); model.load_state_dict(payload["state"]); return model.to(device)


def calibrated_predict(sx, sy, tx, ty, qx, seed):
    x = np.vstack([sx, tx]); y = np.concatenate([sy, ty])
    weights = np.concatenate([balanced_source_weights(sy), np.full(len(ty), 10.0)])
    scaler = StandardScaler().fit(x); head = LogisticRegression(C=.1, solver="lbfgs", max_iter=4000, random_state=seed)
    head.fit(scaler.transform(x), y, sample_weight=weights); return head.predict(scaler.transform(qx))


def evaluate_stage():
    selection_path = OUT / "selection_complete.json"
    if not selection_path.exists(): raise RuntimeError("T2 source selection not frozen")
    if (OUT / "target_evaluation.lock").exists(): raise RuntimeError("T2 evaluation already executed")
    development = load_npz(DEVELOPMENT); query = load_npz(QUERY)
    lock = {"status": "POST_HOC_TARGET_SCORING_STARTED", "selection_sha256": sha256(selection_path),
            "protocol_sha256": sha256(PROTOCOL), "query_cache_sha256": sha256(QUERY)}
    (OUT / "target_evaluation.lock").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    source = np.flatnonzero(development["roles"] == "source")
    support = np.flatnonzero(development["roles"] == "target_support")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows, prediction_rows = [], []
    for seed in SEEDS:
        for name in MODELS:
            model = load_conventional(name, OUT / "checkpoints" / f"{name}_seed{seed}.pt", device)
            sx, _, source_time = infer_model(model, development["distributed"][source], device)
            tx, _, support_time = infer_model(model, development["distributed"][support], device)
            qx, _, query_time = infer_model(model, query["distributed"], device)
            pred = calibrated_predict(sx, development["labels"][source], tx, development["labels"][support], qx, seed)
            value = metric(query["labels"], pred); method = DISPLAY[name]
            metric_rows.append({"seed": seed, "method": method, **{k: value[k] for k in value if k != "confusion_matrix"},
                                "confusion_matrix": json.dumps(value["confusion_matrix"]),
                                "parameters": sum(p.numel() for p in model.parameters()),
                                "query_inference_seconds": query_time})
            for i, prediction in enumerate(pred):
                prediction_rows.append({"seed": seed, "method": method, "run_id": str(query["ids"][i]),
                                        "context": str(query["contexts"][i]), "severity": str(query["severities"][i]),
                                        "torque": int(query["torques"][i]), "true": int(query["labels"][i]),
                                        "prediction": int(prediction), "correct": int(prediction == query["labels"][i])})
            del model
        encoder = load_proto(OUT / "checkpoints" / f"protonet_seed{seed}.pt", device)
        support_z = embed_runs(encoder, development["distributed"][support], device)
        query_z = embed_runs(encoder, query["distributed"], device)
        sy = development["labels"][support]
        prototypes = torch.stack([support_z[torch.as_tensor(sy == c)].mean(0) for c in (0, 1, 2)])
        pred = (-((query_z[:, None] - prototypes[None]) ** 2).mean(2)).argmax(1).cpu().numpy()
        value = metric(query["labels"], pred); method = "ProtoNet-DM8"
        metric_rows.append({"seed": seed, "method": method, **{k: value[k] for k in value if k != "confusion_matrix"},
                            "confusion_matrix": json.dumps(value["confusion_matrix"]),
                            "parameters": sum(p.numel() for p in encoder.parameters()), "query_inference_seconds": np.nan})
        for i, prediction in enumerate(pred):
            prediction_rows.append({"seed": seed, "method": method, "run_id": str(query["ids"][i]),
                                    "context": str(query["contexts"][i]), "severity": str(query["severities"][i]),
                                    "torque": int(query["torques"][i]), "true": int(query["labels"][i]),
                                    "prediction": int(prediction), "correct": int(prediction == query["labels"][i])})
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"T2 target scored seed={seed}", flush=True)

    # Append original frozen U2-Hz-DM8 and RAMEF results without recomputation.
    original_metrics = list(csv.DictReader((BASE / "run" / "seed_metrics.csv").open(encoding="utf-8-sig")))
    original_predictions = list(csv.DictReader((BASE / "run" / "predictions.csv").open(encoding="utf-8-sig")))
    mapping = {"U2-Hz-Distributed-M8": "U2-Hz-DM8", "RAMEF-F": "RAMEF"}
    for row in original_metrics:
        if row["variant"] not in mapping: continue
        metric_rows.append({"seed": int(row["seed"]), "method": mapping[row["variant"]],
                            "accuracy": float(row["accuracy"]), "macro_f1": float(row["macro_f1"]),
                            "recall_NRF": float(row["recall_N"]), "recall_IR": float(row["recall_IR"]),
                            "recall_OR": float(row["recall_OR"]), "errors": int(round(27 * (1-float(row["accuracy"])))),
                            "confusion_matrix": row["confusion_matrix"], "parameters": "", "query_inference_seconds": ""})
    for row in original_predictions:
        if row["variant"] not in mapping: continue
        prediction_rows.append({"seed": int(row["seed"]), "method": mapping[row["variant"]], "run_id": row["run_id"],
                                "context": row["context"], "severity": row["severity"], "torque": int(row["torque"]),
                                "true": int(row["true"]), "prediction": int(row["prediction"]), "correct": int(row["correct"])})
    write_csv(OUT / "seed_metrics.csv", metric_rows); write_csv(OUT / "predictions.csv", prediction_rows)
    methods = sorted(set(row["method"] for row in metric_rows)); summaries = {}
    for method in methods:
        rows = [row for row in metric_rows if row["method"] == method]; summaries[method] = {}
        for key in ("accuracy", "macro_f1", "recall_NRF", "recall_IR", "recall_OR"):
            vals = np.asarray([float(row[key]) for row in rows]); summaries[method][key] = {
                "mean": float(vals.mean()), "std": float(vals.std(ddof=1)), "min": float(vals.min()), "max": float(vals.max())}
    result = {"experiment": "MCC5_T2_FAIR_METHOD_BENCHMARK", "status": "COMPLETE_POST_HOC_FROZEN",
              "changes_ramef": False, "methods": methods, "seeds": list(SEEDS), "summaries": summaries,
              "protocol_sha256": sha256(PROTOCOL), "selection_sha256": sha256(selection_path),
              "query_cache_sha256": sha256(QUERY)}
    result_path = OUT / "result.json"; result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lock.update({"status": "COMPLETE_POST_HOC_FROZEN", "result_sha256": sha256(result_path)})
    (OUT / "target_evaluation.lock").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "train": train_stage()
    else: evaluate_stage()


if __name__ == "__main__": main()
