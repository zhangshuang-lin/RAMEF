"""Preregistered MCC5 RAMEF confirmatory experiment.

Use ``--stage train`` first. It cannot read target-query vibration. Only after
all five source-selected checkpoints are frozen may ``--stage evaluate`` create
the query cache and perform the single target evaluation event.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from mcc5_confirmatory_data import FS, WINDOW, build_cache, sha256
from u2_stft_model import U2STFT


SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
VARIANTS = ("U2-Hz-K1", "U2-Hz-Local-M8", "U2-Hz-Distributed-M8", "RAMEF-F", "Random63")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def standardise(waves: torch.Tensor) -> torch.Tensor:
    mean = waves.mean(-1, keepdim=True)
    std = waves.std(-1, keepdim=True).clamp_min(1e-6)
    return (waves - mean) / std


def make_model() -> U2STFT:
    return U2STFT(
        num_classes=3, lite=True, use_speed_adapter=False,
        use_iror_physics_branch=False, use_low_order_branch=False,
        support_conditioned_gate=False, late_fusion=False,
        sample_rate=FS, n_fft=1024, hop_length=256, win_length=1024,
        order_min=.5, order_max=256., order_bins=256, out_size=(128, 64),
        frontend_ablation="hz",
    )


def load_npz(path: Path) -> dict:
    cache = np.load(path, allow_pickle=False)
    return {key: cache[key] for key in cache.files}


def measurement_logits(model: U2STFT, waves: np.ndarray, rpms: np.ndarray,
                       device: torch.device, batch: int = 24) -> np.ndarray:
    model.eval(); x = torch.from_numpy(waves.reshape(-1, WINDOW)); r = torch.from_numpy(rpms.reshape(-1))
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            output.append(model(standardise(x[start:start + batch]).to(device),
                                r[start:start + batch].to(device))["logits"].cpu())
    return torch.cat(output).reshape(len(waves), 8, 3).mean(1).numpy()


def measurement_embeddings(model: U2STFT, waves: np.ndarray, rpms: np.ndarray,
                           device: torch.device, batch: int = 24) -> np.ndarray:
    model.eval(); x = torch.from_numpy(waves.reshape(-1, WINDOW)); r = torch.from_numpy(rpms.reshape(-1))
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            output.append(model(standardise(x[start:start + batch]).to(device),
                                r[start:start + batch].to(device))["embedding"].cpu())
    return torch.cat(output).reshape(len(waves), 8, -1).numpy()


def train_seed(cache: dict, seed: int, checkpoint: Path,
               device: torch.device) -> tuple[float, list[dict]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    source = np.flatnonzero(cache["roles"] == "source")
    validation = np.flatnonzero(cache["roles"] == "source_validation")
    model = make_model().to(device)
    waves = torch.from_numpy(cache["distributed"][source].reshape(-1, WINDOW))
    rpms = torch.from_numpy(cache["rpm_distributed"][source].reshape(-1))
    labels = torch.tensor(np.repeat(cache["labels"][source], 8), dtype=torch.long)
    counts = Counter(labels.tolist())
    weights = torch.tensor([1.0 / counts[int(label)] for label in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=max(counts.values()) * 3,
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(waves, labels, rpms), batch_size=12,
                        sampler=sampler, num_workers=0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1.0; best_state = None; history = []
    for epoch in range(12):
        model.train(); losses = []
        for x, y, rpm in loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(standardise(x).to(device), rpm.to(device))["logits"]
            loss = F.cross_entropy(logits, y.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step(); losses.append(float(loss.detach()))
        logits = measurement_logits(model, cache["distributed"][validation],
                                    cache["rpm_distributed"][validation], device)
        truth = cache["labels"][validation]; prediction = logits.argmax(1)
        score = float(f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0))
        history.append({"seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)),
                        "source_validation_macro_f1": score})
        if score > best + 1e-12:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None: raise RuntimeError("No checkpoint selected")
    torch.save({"state": best_state, "seed": seed, "best_val_macro_f1": best,
                "history": history}, checkpoint)
    return best, history


def load_model(path: Path, device: torch.device) -> U2STFT:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(); model.load_state_dict(payload["state"])
    return model.to(device)


def balanced_source_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist()); total = len(labels)
    return np.asarray([total / (3.0 * counts[int(label)]) for label in labels], dtype=np.float64)


def random63(ids: np.ndarray) -> np.ndarray:
    rows = []
    for item in ids.tolist():
        seed = int(hashlib.sha256(f"MCC5_RANDOM63:{item}".encode()).hexdigest()[:16], 16) % (2**32)
        rows.append(np.random.default_rng(seed).normal(size=63))
    return np.asarray(rows, dtype=np.float32)


def representation(variant: str, embeddings: np.ndarray, physical: np.ndarray,
                   ids: np.ndarray) -> np.ndarray:
    if variant == "U2-Hz-K1": return embeddings[:, 3]
    if variant == "U2-Hz-Local-M8": return embeddings.mean(1)
    # Caller passes local or distributed embeddings as appropriate. For the
    # remaining variants it passes distributed embeddings.
    carrier = embeddings.mean(1)
    if variant == "U2-Hz-Distributed-M8": return carrier
    if variant == "RAMEF-F": return np.hstack([carrier, physical])
    if variant == "Random63": return np.hstack([carrier, random63(ids)])
    raise ValueError(variant)


def fit_head(source_x: np.ndarray, source_y: np.ndarray, support_x: np.ndarray,
             support_y: np.ndarray, query_x: np.ndarray, seed: int) -> np.ndarray:
    train_x = np.vstack([source_x, support_x]); train_y = np.concatenate([source_y, support_y])
    weights = np.concatenate([balanced_source_weights(source_y), np.full(len(support_y), 10.0)])
    scaler = StandardScaler().fit(train_x)
    head = LogisticRegression(C=.1, solver="lbfgs", max_iter=4000, random_state=seed)
    head.fit(scaler.transform(train_x), train_y, sample_weight=weights)
    return head.predict(scaler.transform(query_x))


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    recalls = recall_score(truth, prediction, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(truth == prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_N": float(recalls[0]), "recall_IR": float(recalls[1]), "recall_OR": float(recalls[2]),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist()}


def train_stage(args, output: Path) -> None:
    cache = load_npz(args.development_cache)
    if np.any(cache["roles"] == "target_query") or bool(cache["query_cache"]):
        raise RuntimeError("Development cache contains target query")
    checkpoints = output / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, history = [], []
    for seed in SEEDS:
        path = checkpoints / f"u2hz_seed{seed}.pt"
        best, local_history = train_seed(cache, seed, path, device)
        rows.append({"seed": seed, "best_source_validation_macro_f1": best,
                     "checkpoint": str(path), "sha256": sha256(path)})
        history.extend(local_history)
        print(f"source-selected seed={seed} macro_f1={best:.6f}", flush=True)
    write_csv(output / "checkpoints.csv", rows); write_csv(output / "training_history.csv", history)
    selection = {"experiment": "MCC5_RAMEF_CONFIRMATORY", "seeds": list(SEEDS),
                 "query_vibration_loaded": False, "checkpoints": rows,
                 "development_cache_sha256": sha256(args.development_cache),
                 "protocol_sha256": sha256(args.protocol)}
    (output / "selection_complete.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")


def evaluate_stage(args, output: Path) -> None:
    selection_path = output / "selection_complete.json"
    if not selection_path.exists(): raise RuntimeError("Source selection is incomplete")
    lock = output / "target_evaluation.lock"
    if lock.exists(): raise RuntimeError("Target evaluation lock already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    lock.write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                                "selection_sha256": sha256(selection_path),
                                "protocol_sha256": sha256(args.protocol)}, indent=2), encoding="utf-8")
    build_cache(args.data_root, args.manifest, args.keyphase_audit, args.query_cache, True)
    development = load_npz(args.development_cache); query = load_npz(args.query_cache)
    source = np.flatnonzero(development["roles"] == "source")
    support = np.flatnonzero(development["roles"] == "target_support")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows, metric_rows = [], []
    per_seed_variant: dict[tuple[int, str], dict] = {}
    context_scores: dict[tuple[int, str, str], float] = {}
    for seed in SEEDS:
        checkpoint = output / "checkpoints" / f"u2hz_seed{seed}.pt"
        model = load_model(checkpoint, device)
        src_local = measurement_embeddings(model, development["local"][source], development["rpm_local"][source], device)
        src_dist = measurement_embeddings(model, development["distributed"][source], development["rpm_distributed"][source], device)
        sup_local = measurement_embeddings(model, development["local"][support], development["rpm_local"][support], device)
        sup_dist = measurement_embeddings(model, development["distributed"][support], development["rpm_distributed"][support], device)
        qry_local = measurement_embeddings(model, query["local"], query["rpm_local"], device)
        qry_dist = measurement_embeddings(model, query["distributed"], query["rpm_distributed"], device)
        for variant in VARIANTS:
            local_variant = variant in {"U2-Hz-K1", "U2-Hz-Local-M8"}
            source_x = representation(variant, src_local if local_variant else src_dist,
                                      development["physical"][source], development["ids"][source])
            support_x = representation(variant, sup_local if local_variant else sup_dist,
                                       development["physical"][support], development["ids"][support])
            query_x = representation(variant, qry_local if local_variant else qry_dist,
                                     query["physical"], query["ids"])
            pred = fit_head(source_x, development["labels"][source], support_x,
                            development["labels"][support], query_x, seed)
            value = metrics(query["labels"], pred); per_seed_variant[(seed, variant)] = value
            metric_rows.append({"seed": seed, "variant": variant, **value,
                                "confusion_matrix": json.dumps(value["confusion_matrix"])})
            for index, prediction in enumerate(pred):
                prediction_rows.append({"seed": seed, "variant": variant,
                                        "run_id": query["ids"][index], "context": query["contexts"][index],
                                        "severity": query["severities"][index], "torque": int(query["torques"][index]),
                                        "true": int(query["labels"][index]), "prediction": int(prediction),
                                        "correct": int(prediction == query["labels"][index])})
            for context in np.unique(query["contexts"]):
                mask = query["contexts"] == context
                context_scores[(seed, variant, str(context))] = float(np.mean(pred[mask] == query["labels"][mask]))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"target evaluation seed={seed}", flush=True)
    write_csv(output / "seed_metrics.csv", metric_rows); write_csv(output / "predictions.csv", prediction_rows)

    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for key in ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR"):
            values = np.asarray([per_seed_variant[(seed, variant)][key] for seed in SEEDS])
            summary[variant][key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                                     "min": float(values.min()), "max": float(values.max())}
    delta = lambda a, b, key: np.asarray([per_seed_variant[(s, a)][key] - per_seed_variant[(s, b)][key] for s in SEEDS])
    ramef_local = delta("RAMEF-F", "U2-Hz-Local-M8", "macro_f1")
    dist_local = delta("U2-Hz-Distributed-M8", "U2-Hz-Local-M8", "macro_f1")
    ramef_dist = delta("RAMEF-F", "U2-Hz-Distributed-M8", "macro_f1")
    random_local = delta("Random63", "U2-Hz-Local-M8", "macro_f1")
    class_safe = all(summary["RAMEF-F"][key]["mean"] + 1e-12 >= summary["U2-Hz-Local-M8"][key]["mean"]
                     for key in ("recall_IR", "recall_OR"))
    contexts = sorted(set(key[2] for key in context_scores))
    context_safe = all(
        np.mean([context_scores[(s, "RAMEF-F", c)] for s in SEEDS]) + .10 >=
        np.mean([context_scores[(s, "U2-Hz-Local-M8", c)] for s in SEEDS])
        for c in contexts
    )
    gates = {
        "ramef_minus_local_macro_f1_ge_2pp": float(ramef_local.mean()) >= .02 - 1e-12,
        "ramef_accuracy_positive_at_least_4of5": int(np.sum(delta("RAMEF-F", "U2-Hz-Local-M8", "accuracy") > 0)) >= 4,
        "distributed_nonworse_and_positive_at_least_3of5": float(dist_local.mean()) >= -1e-12 and int(np.sum(dist_local > 0)) >= 3,
        "ramef_ir_or_recall_nonworse": class_safe,
        "ramef_minus_distributed_macro_f1_ge_minus_1pp": float(ramef_dist.mean()) >= -.01 - 1e-12,
        "random63_does_not_match_ramef": float(random_local.mean()) < float(ramef_local.mean()) - 1e-12,
        "no_context_catastrophic_decline": context_safe,
    }
    result = {"experiment": "MCC5_RAMEF_CONFIRMATORY", "status": "PASS" if all(gates.values()) else "FAIL",
              "summaries": summary, "deltas": {"ramef_minus_local_macro_f1": ramef_local.tolist(),
              "distributed_minus_local_macro_f1": dist_local.tolist(),
              "ramef_minus_distributed_macro_f1": ramef_dist.tolist(),
              "random_minus_local_macro_f1": random_local.tolist()}, "gates": gates,
              "query_runs": int(len(query["labels"])), "seeds": list(SEEDS),
              "query_loaded_before_source_selection": False,
              "manifest_sha256": sha256(args.manifest), "protocol_sha256": sha256(args.protocol)}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lock_payload = json.loads(lock.read_text(encoding="utf-8")); lock_payload.update(
        {"completed_utc": datetime.now(timezone.utc).isoformat(), "result_sha256": sha256(output / "result.json")})
    lock.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\MCC5-THU Motor\MCC5-THU Motor"))
    parser.add_argument("--development-cache", type=Path, default=Path("outputs/staged/MCC5_RAMEF_CONFIRMATORY/cache/development.npz"))
    parser.add_argument("--query-cache", type=Path, default=Path("outputs/staged/MCC5_RAMEF_CONFIRMATORY/cache/query_after_selection.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/staged/MCC5_RAMEF_CONFIRMATORY/protocol/protocol_run_manifest.csv"))
    parser.add_argument("--keyphase-audit", type=Path, default=Path("outputs/staged/MCC5_RAMEF_CONFIRMATORY/P1/speed_keyphase_audit.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("MCC5_RAMEF_CONFIRMATORY_PROTOCOL.md"))
    parser.add_argument("--output", type=Path, default=Path("outputs/staged/MCC5_RAMEF_CONFIRMATORY/run"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "train": train_stage(args, args.output)
    else: evaluate_stage(args, args.output)


if __name__ == "__main__":
    main()
