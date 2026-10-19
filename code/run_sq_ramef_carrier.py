"""Train and evaluate SQ-native U2-Hz under SQ_RAMEF_STRICT_V1.4.

Each outer fold is trained independently. Query-role waveforms are materialised
only after all five source-selected checkpoints for that fold are frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from bottleneck_deconvolution import sha256
from sq_data import MACRO, SQRecord, load_txt_channel, scan_sq
from u2_stft_model import U2STFT


SEEDS = (20261401, 20261402, 20261403, 20261404, 20261405)
VARIANTS = ("Single", "Local-M8", "Distributed-M8")
WINDOW = 6000
SAMPLE_RATE = 25600.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def standardise(waves: torch.Tensor) -> torch.Tensor:
    mean = waves.mean(-1, keepdim=True)
    std = waves.std(-1, keepdim=True).clamp_min(1e-6)
    return (waves - mean) / std


def make_model() -> U2STFT:
    return U2STFT(
        num_classes=3, lite=True, use_speed_adapter=False,
        use_iror_physics_branch=False, use_low_order_branch=False,
        support_conditioned_gate=False, late_fusion=False,
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256,
        win_length=1024, order_min=.5, order_max=256., order_bins=256,
        out_size=(128, 64), frontend_ablation="hz",
    )


def block_rpms(detail: dict, starts: list[int]) -> np.ndarray:
    indices = np.asarray(detail["speed_sample_indices"], dtype=np.float64)
    rpm = np.asarray(detail["instantaneous_rpm"], dtype=np.float64)
    centre = np.asarray(starts, dtype=np.float64) + WINDOW / 2
    return np.interp(centre, indices, rpm).astype(np.float32)


def load_views(rec_ids: list[str], records: dict[str, SQRecord], details: dict[str, dict],
               target_region: str, include_target: bool) -> dict:
    source_local, source_dist, source_rpm_local, source_rpm_dist = [], [], [], []
    target_local, target_dist, target_rpm_local, target_rpm_dist = [], [], [], []
    labels, subtypes = [], []
    for index, rec_id in enumerate(rec_ids):
        record = records[rec_id]; detail = details[rec_id]
        _, signal = load_txt_channel(record.vibration_path, record.header_rows, record.block_size)
        src = detail["regions"]["plateau_source"]
        src_local_starts = src["local_starts"]
        src_dist_starts = src["distributed_starts"]
        if len(src_local_starts) == 8 and len(src_dist_starts) == 8:
            source_local.append(np.stack([signal[start:start + WINDOW] for start in src_local_starts]))
            source_dist.append(np.stack([signal[start:start + WINDOW] for start in src_dist_starts]))
            source_rpm_local.append(block_rpms(detail, src_local_starts))
            source_rpm_dist.append(block_rpms(detail, src_dist_starts))
        else:
            source_local.append(np.zeros((8, WINDOW), dtype=np.float32))
            source_dist.append(np.zeros((8, WINDOW), dtype=np.float32))
            source_rpm_local.append(np.zeros(8, dtype=np.float32))
            source_rpm_dist.append(np.zeros(8, dtype=np.float32))
        if include_target:
            tgt = detail["regions"][target_region]
            local_starts = tgt["local_starts"]; dist_starts = tgt["distributed_starts"]
            if len(local_starts) != 8 or len(dist_starts) != 8:
                raise RuntimeError(f"Target budget missing for {rec_id}")
            target_local.append(np.stack([signal[start:start + WINDOW] for start in local_starts]))
            target_dist.append(np.stack([signal[start:start + WINDOW] for start in dist_starts]))
            target_rpm_local.append(block_rpms(detail, local_starts))
            target_rpm_dist.append(block_rpms(detail, dist_starts))
        labels.append(record.label); subtypes.append(record.subtype)
        print(f"loaded SQ views {index + 1}/{len(rec_ids)} {rec_id}", flush=True)
    payload = {
        "ids": rec_ids, "labels": np.asarray(labels, dtype=np.int64),
        "subtypes": subtypes,
        "source_local": np.asarray(source_local, dtype=np.float32),
        "source_distributed": np.asarray(source_dist, dtype=np.float32),
        "source_rpm_local": np.asarray(source_rpm_local, dtype=np.float32),
        "source_rpm_distributed": np.asarray(source_rpm_dist, dtype=np.float32),
    }
    if include_target:
        payload.update({
            "target_local": np.asarray(target_local, dtype=np.float32),
            "target_distributed": np.asarray(target_dist, dtype=np.float32),
            "target_rpm_local": np.asarray(target_rpm_local, dtype=np.float32),
            "target_rpm_distributed": np.asarray(target_rpm_dist, dtype=np.float32),
        })
    return payload


def save_cache(path: Path, payload: dict) -> None:
    arrays = {key: value for key, value in payload.items()
              if isinstance(value, np.ndarray)}
    arrays["ids"] = np.asarray(payload["ids"])
    arrays["subtypes"] = np.asarray(payload["subtypes"])
    np.savez_compressed(path, **arrays)


def load_cache(path: Path) -> dict:
    cache = np.load(path, allow_pickle=False)
    return {key: cache[key] for key in cache.files}


def split_train_val(rec_ids: list[str], records: dict[str, SQRecord], fold: int) -> tuple[np.ndarray, np.ndarray]:
    by_label_subtype: dict[tuple[int, str], list[int]] = {}
    for index, rec_id in enumerate(rec_ids):
        record = records[rec_id]
        by_label_subtype.setdefault((record.label, record.subtype), []).append(index)
    validation = []
    for key, indices in sorted(by_label_subtype.items()):
        indices = sorted(indices, key=lambda i: stable_int(f"{fold}:{rec_ids[i]}"))
        # N has one subtype; IR/OR retain one validation REC per severity.
        validation.append(indices[0])
    val = np.asarray(sorted(set(validation)), dtype=np.int64)
    train = np.asarray([i for i in range(len(rec_ids)) if i not in set(val)], dtype=np.int64)
    if not all(np.any([records[rec_ids[i]].label == label for i in train]) for label in (0, 1, 2)):
        raise RuntimeError("Source train split lacks a class")
    return train, val


def measurement_logits(model: U2STFT, waves: np.ndarray, rpms: np.ndarray,
                       device: torch.device, batch: int = 24) -> np.ndarray:
    model.eval(); x = torch.from_numpy(waves.reshape(-1, WINDOW)); r = torch.from_numpy(rpms.reshape(-1))
    rows = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            rows.append(model(standardise(x[start:start + batch]).to(device),
                              r[start:start + batch].to(device))["logits"].cpu())
    return torch.cat(rows).reshape(len(waves), 8, 3).mean(1).numpy()


def train_model(source: dict, train_idx: np.ndarray, val_idx: np.ndarray,
                seed: int, checkpoint: Path, device: torch.device) -> tuple[U2STFT, list[dict]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = make_model().to(device)
    waves = torch.from_numpy(source["source_distributed"][train_idx].reshape(-1, WINDOW))
    rpms = torch.from_numpy(source["source_rpm_distributed"][train_idx].reshape(-1))
    labels = torch.tensor(np.repeat(source["labels"][train_idx], 8), dtype=torch.long)
    counts = Counter(labels.tolist())
    sample_weights = torch.tensor([1.0 / counts[int(label)] for label in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=max(counts.values()) * 3,
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(waves, labels, rpms), batch_size=12,
                        sampler=sampler, num_workers=0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1.0; state = None; history = []
    for epoch in range(12):
        model.train(); losses = []
        for x, y, rpm in loader:
            optimiser.zero_grad(set_to_none=True)
            output = model(standardise(x).to(device), rpm.to(device))
            loss = F.cross_entropy(output["logits"], y.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step(); losses.append(float(loss.detach()))
        logits = measurement_logits(model, source["source_distributed"][val_idx],
                                    source["source_rpm_distributed"][val_idx], device)
        y_val = source["labels"][val_idx]; pred = logits.argmax(1)
        score = float(f1_score(y_val, pred, labels=[0, 1, 2], average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_macro_f1": score})
        if score > best + 1e-9:
            best = score
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if state is None: raise RuntimeError("No source checkpoint")
    model.load_state_dict(state)
    torch.save({"state": state, "best_val_macro_f1": best,
                "seed": seed, "history": history}, checkpoint)
    return model, history


def load_model(checkpoint: Path, device: torch.device) -> U2STFT:
    model = make_model(); payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state"]); return model.to(device)


def measurement_embeddings(model: U2STFT, waves: np.ndarray, rpms: np.ndarray,
                           device: torch.device, batch: int = 24) -> np.ndarray:
    model.eval(); x = torch.from_numpy(waves.reshape(-1, WINDOW)); r = torch.from_numpy(rpms.reshape(-1))
    rows = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            output = model(standardise(x[start:start + batch]).to(device),
                           r[start:start + batch].to(device))
            rows.append(output["embedding"].cpu())
    return torch.cat(rows).reshape(len(waves), 8, -1).numpy()


def balanced_source_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist()); total = len(labels)
    return np.asarray([total / (3.0 * counts[int(label)]) for label in labels], dtype=np.float64)


def fit_head(source_x: np.ndarray, source_y: np.ndarray,
             support_x: np.ndarray, support_y: np.ndarray,
             query_x: np.ndarray, seed: int) -> np.ndarray:
    train_x = np.vstack([source_x, support_x]); train_y = np.concatenate([source_y, support_y])
    weights = np.concatenate([balanced_source_weights(source_y), np.full(len(support_y), 10.0)])
    scaler = StandardScaler().fit(train_x)
    head = LogisticRegression(C=.1, solver="lbfgs", max_iter=3000, random_state=seed)
    head.fit(scaler.transform(train_x), train_y, sample_weight=weights)
    return head.predict(scaler.transform(query_x))


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(y == pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_N": float(recall[0]), "recall_IR": float(recall[1]),
            "recall_OR": float(recall[2]),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\SQ-variable-speed-bearing"))
    parser.add_argument("--p0", type=Path, default=Path("outputs/staged/SQ_RAMEF_P0_V1_3"))
    parser.add_argument("--protocol", type=Path, default=Path("SQ_RAMEF_STRICT_PROTOCOL_V1_4.md"))
    parser.add_argument("--output", type=Path, default=Path("outputs/staged/SQ_RAMEF_CARRIER"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    decision = json.loads((args.p0 / "p0_decision.json").read_text(encoding="utf-8"))
    if not decision["training_authorised"] or decision["raw_window_samples"] != WINDOW:
        raise RuntimeError("SQ P0 did not authorise this runner")
    frozen = args.output / "protocol_frozen.md"
    if frozen.exists() and sha256(frozen) != sha256(args.protocol):
        raise RuntimeError("Carrier protocol changed after training began")
    if not frozen.exists(): shutil.copy2(args.protocol, frozen)
    schedule = json.loads((args.p0 / "split_schedule.json").read_text(encoding="utf-8"))
    details_list = json.loads((args.p0 / "record_speed_metadata.json").read_text(encoding="utf-8"))
    details = {row["rec_id"]: row for row in details_list}
    records = {record.rec_id: record for record in scan_sq(args.data_root)}
    target_region = schedule["primary_target_region"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows, history_rows, checkpoint_rows = [], [], []

    for fold_item in schedule["folds"]:
        fold = int(fold_item["fold"]); fold_dir = args.output / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True); checkpoint_dir = fold_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
        source_ids = fold_item["roles"]["source"]
        source_cache_path = fold_dir / "source_views.npz"
        if not source_cache_path.exists():
            save_cache(source_cache_path, load_views(source_ids, records, details, target_region, False))
        source = load_cache(source_cache_path)
        train_idx, val_idx = split_train_val(source_ids, records, fold)
        for seed in SEEDS:
            checkpoint = checkpoint_dir / f"u2hz_seed{seed}.pt"
            if checkpoint.exists():
                model = load_model(checkpoint, device); local_history = []
            else:
                model, local_history = train_model(source, train_idx, val_idx, seed, checkpoint, device)
            for row in local_history:
                history_rows.append({"fold": fold, "seed": seed, **row})
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_rows.append({"fold": fold, "seed": seed,
                                    "best_val_macro_f1": payload["best_val_macro_f1"],
                                    "checkpoint": str(checkpoint), "sha256": sha256(checkpoint)})
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"source-selected checkpoint fold={fold} seed={seed}", flush=True)
        selection_path = fold_dir / "selection_complete.json"
        selection_path.write_text(json.dumps({"fold": fold,
                                              "query_loaded_before_selection": False,
                                              "checkpoints": [row for row in checkpoint_rows if row["fold"] == fold]},
                                             indent=2), encoding="utf-8")

        target_ids = fold_item["roles"]["support"] + fold_item["roles"]["query"]
        target_cache_path = fold_dir / "target_views_after_selection.npz"
        if not target_cache_path.exists():
            save_cache(target_cache_path, load_views(target_ids, records, details, target_region, True))
        target = load_cache(target_cache_path)
        support_count = len(fold_item["roles"]["support"])
        if support_count != 15 or len(fold_item["roles"]["query"]) != 3:
            raise RuntimeError("SQ target role count mismatch")
        support_y = target["labels"][:support_count]; query_y = target["labels"][support_count:]
        query_ids = target["ids"][support_count:].tolist(); query_subtypes = target["subtypes"][support_count:].tolist()

        for seed in SEEDS:
            checkpoint = checkpoint_dir / f"u2hz_seed{seed}.pt"; model = load_model(checkpoint, device)
            source_local_tokens = measurement_embeddings(model, source["source_local"], source["source_rpm_local"], device)
            source_dist_tokens = measurement_embeddings(model, source["source_distributed"], source["source_rpm_distributed"], device)
            target_local_tokens = measurement_embeddings(model, target["target_local"], target["target_rpm_local"], device)
            target_dist_tokens = measurement_embeddings(model, target["target_distributed"], target["target_rpm_distributed"], device)
            views = {
                "Single": (source_local_tokens[:, 3], target_local_tokens[:, 3]),
                "Local-M8": (source_local_tokens.mean(1), target_local_tokens.mean(1)),
                "Distributed-M8": (source_dist_tokens.mean(1), target_dist_tokens.mean(1)),
            }
            for variant, (source_x, target_x) in views.items():
                pred = fit_head(source_x, source["labels"], target_x[:support_count],
                                support_y, target_x[support_count:], seed + fold)
                for index, rec_id in enumerate(query_ids):
                    prediction_rows.append({"fold": fold, "seed": seed, "variant": variant,
                                            "rec_id": rec_id, "subtype": query_subtypes[index],
                                            "true": int(query_y[index]), "prediction": int(pred[index]),
                                            "correct": int(pred[index] == query_y[index])})
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"carrier evaluation fold={fold} seed={seed}", flush=True)

    write_csv(args.output / "training_history.csv", history_rows)
    write_csv(args.output / "checkpoints.csv", checkpoint_rows)
    write_csv(args.output / "predictions.csv", prediction_rows)
    summaries = {}; seed_metrics = []
    for variant in VARIANTS:
        local = [row for row in prediction_rows if row["variant"] == variant]
        for seed in SEEDS:
            selected = [row for row in local if row["seed"] == seed]
            value = metric(np.asarray([row["true"] for row in selected]),
                           np.asarray([row["prediction"] for row in selected]))
            seed_metrics.append({"variant": variant, "seed": seed, **value})
        values = [row for row in seed_metrics if row["variant"] == variant]
        summaries[variant] = {key: {"mean": float(np.mean([row[key] for row in values])),
                                    "std": float(np.std([row[key] for row in values], ddof=1))}
                              for key in ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR")}
    write_csv(args.output / "seed_metrics.csv", seed_metrics)

    paired = []
    for fold in range(6):
        for seed in SEEDS:
            row = {"fold": fold, "seed": seed}
            for variant in ("Local-M8", "Distributed-M8"):
                selected = [item for item in prediction_rows
                            if item["fold"] == fold and item["seed"] == seed and item["variant"] == variant]
                row[variant] = float(np.mean([item["correct"] for item in selected]))
            row["distributed_minus_local"] = row["Distributed-M8"] - row["Local-M8"]
            paired.append(row)
    write_csv(args.output / "paired_folds.csv", paired)
    mean_macro_gain = (summaries["Distributed-M8"]["macro_f1"]["mean"]
                       - summaries["Local-M8"]["macro_f1"]["mean"])
    positive_folds = 0
    for fold in range(6):
        values = [row["distributed_minus_local"] for row in paired if row["fold"] == fold]
        positive_folds += float(np.mean(values)) > 0
    positive_seeds = 0
    for seed in SEEDS:
        values = [row["distributed_minus_local"] for row in paired if row["seed"] == seed]
        positive_seeds += float(np.mean(values)) > 0
    gates = {"macro_f1_gain_ge_2pp": mean_macro_gain >= .02,
             "positive_at_least_5of6_folds": positive_folds >= 5,
             "positive_at_least_4of5_seeds": positive_seeds >= 4}
    result = {"experiment": "SQ_RAMEF_CARRIER", "status": "PASS" if all(gates.values()) else "FAIL",
              "physical_stage_authorised": bool(all(gates.values())),
              "summaries": summaries, "mean_macro_f1_gain": mean_macro_gain,
              "positive_folds": positive_folds, "positive_seeds": positive_seeds,
              "gates": gates, "folds": 6, "seeds": list(SEEDS),
              "query_loaded_before_source_selection": False,
              "p0_decision_sha256": sha256(args.p0 / "p0_decision.json"),
              "split_schedule_sha256": sha256(args.p0 / "split_schedule.json"),
              "protocol_sha256": sha256(frozen)}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
