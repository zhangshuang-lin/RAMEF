"""Five-seed PU equal-budget reproducibility audit.

Each invocation trains one source-selected encoder and then evaluates the fixed
DS development cohort. The `--summarize` mode aggregates all 25 preregistered
runs without selecting a target seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from bottleneck_deconvolution import sha256
from fair_baseline_models import make_model
from prepare_splits import scan_records
from run_core_component_ablation import make_u2
from run_fair_baseline_pu import centre_item, fixed_source_split, waveform
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_pu_eightblock_fair_baselines import wave_cache
from run_racp_measurement_final_once import load_query
from run_tser_hb_support_p0 import support_only


MODELS = ("wdcnn", "resnet1d", "stft_cnn", "wdcnn_dann", "u2_hz")
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
PROTOCOL = Path("PU_MULTISEED_STABILITY_PROTOCOL.md")
OUTPUT = Path("outputs/staged/PU_MULTISEED_STABILITY")
CONFIG = Path("frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall_N": float(recalls[0]),
        "recall_IR": float(recalls[1]),
        "recall_OR": float(recalls[2]),
        "errors": int(np.sum(y != pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def create_model(name: str, device: torch.device):
    if name == "u2_hz":
        return make_u2("hz", CONFIG).to(device)
    return make_model(name).to(device)


def forward_train(model, name, x, rpm, grl):
    if name == "u2_hz":
        out = model(x, rpm)
        return out["logits"], None
    out = model(x, float(grl))
    return out["class_logits"], out.get("domain_logits")


@torch.no_grad()
def validation_logits(model, name, waves, rpms, indices, device):
    model.eval(); rows = []
    batch = 16 if name == "u2_hz" else 64
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        x = waves[idx].to(device)
        if name == "u2_hz":
            rows.append(model(x, rpms[idx].to(device))["logits"].cpu())
        else:
            rows.append(model(x)["class_logits"].cpu())
    return torch.cat(rows).numpy()


@torch.no_grad()
def infer_eight_blocks(model, name, waves, rpms, device):
    model.eval(); flat = waves.reshape(-1, waves.shape[-1]); flat_rpm = rpms.repeat_interleave(8); rows = []
    batch = 16 if name == "u2_hz" else 64
    for start in range(0, len(flat), batch):
        x = flat[start:start + batch].to(device)
        if name == "u2_hz":
            z = model(x, flat_rpm[start:start + batch].to(device))["tokens"].flatten(1)
        else:
            z = model(x)["embedding"]
        rows.append(z.cpu())
    return torch.cat(rows).reshape(len(waves), 8, -1).mean(1).numpy()


def load_data():
    records = scan_records(Path("D:/Paderborn"))
    source = sorted([r for r in records if r["domain"] in SOURCE_DOMAINS], key=lambda r: r["measurement_id"])
    support, _ = support_only(Path("split_independent_measurement.json"))
    query = load_query(Path("split_independent_measurement.json"))
    # Centre crops are required only for source training/validation. Target
    # rows retain historical short-window bounds in the split, so they must not
    # enter this cache.
    centre_path = OUTPUT / "source_centre_waves.pt"
    source_ids = [r["measurement_id"] for r in source]
    if centre_path.exists():
        payload = torch.load(centre_path, map_location="cpu", weights_only=False)
        if payload["measurement_ids"] != source_ids:
            raise RuntimeError("Stale source centre-wave cache")
        centre_waves = payload["waveforms"].float()
    else:
        centre_waves = torch.from_numpy(
            np.stack([waveform(centre_item(row)) for row in source])
        ).half()
        torch.save({"measurement_ids": source_ids, "waveforms": centre_waves}, centre_path)
        centre_waves = centre_waves.float()
    rows = source + support + query
    eight = wave_cache(rows, Path("outputs/staged/PU_EIGHTBLOCK_FAIR_BASELINES/eightblock_waves.pt"))
    if len(eight) != len(rows):
        raise RuntimeError("Eight-block cache length mismatch")
    return source, support, query, centre_waves, eight


def train_one(name: str, seed: int) -> None:
    if name not in MODELS or seed not in SEEDS:
        raise ValueError("Model or seed is not preregistered")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / "protocol_frozen.md"
    if frozen.exists() and sha256(frozen) != sha256(PROTOCOL):
        raise RuntimeError("Frozen protocol mismatch")
    if not frozen.exists():
        shutil.copy2(PROTOCOL, frozen)
    run_path = OUTPUT / "runs" / f"{name}_seed{seed}.json"
    if run_path.exists():
        print(f"SKIP existing {run_path}", flush=True)
        return
    (OUTPUT / "runs").mkdir(exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT / "predictions").mkdir(exist_ok=True)

    source, support, query, centre, eight = load_data()
    labels = np.asarray([int(x["label"]) for x in source], dtype=int)
    domain_map = {d: i for i, d in enumerate(SOURCE_DOMAINS)}
    domains = np.asarray([domain_map[x["domain"]] for x in source], dtype=int)
    train_idx, val_idx = fixed_source_split(source)
    source_rpm = torch.tensor([float(x["rpm"]) for x in source], dtype=torch.float32)
    all_rows = source + support + query
    all_rpm = torch.tensor([float(x["rpm"]) for x in all_rows], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)
    model = create_model(name, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(
        centre[train_idx], torch.tensor(labels[train_idx]),
        source_rpm[train_idx], torch.tensor(domains[train_idx]),
    )
    batch_size = 12 if name == "u2_hz" else 32
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_score = -1.0; best_state = None; best_epoch = -1; history = []
    for epoch in range(12):
        model.train(); losses = []
        progress = epoch / 11; grl = 2 / (1 + np.exp(-10 * progress)) - 1
        for x, y, rpm, domain in loader:
            x, y, rpm, domain = x.to(device), y.to(device), rpm.to(device), domain.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, domain_logits = forward_train(model, name, x, rpm, grl)
                loss = F.cross_entropy(logits, y)
                if domain_logits is not None:
                    loss = loss + 0.2 * F.cross_entropy(domain_logits, domain)
            loss.backward()
            if name == "u2_hz":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        val_logits = validation_logits(model, name, centre, source_rpm, val_idx, device)
        val_pred = val_logits.argmax(1)
        val_metric = metric(labels[val_idx], val_pred)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **val_metric})
        print(name, seed, "epoch", epoch + 1, "val_f1", f"{val_metric['macro_f1']:.4f}", flush=True)
        if val_metric["macro_f1"] > best_score:
            best_score = val_metric["macro_f1"]; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    checkpoint = OUTPUT / "checkpoints" / f"{name}_seed{seed}.pt"
    torch.save({"state": best_state, "seed": seed, "model": name,
                "best_epoch": best_epoch, "best_source_val_macro_f1": best_score,
                "history": history}, checkpoint)

    embeddings = infer_eight_blocks(model, name, eight, all_rpm, device)
    ns, nt = len(source), len(support)
    train_x = embeddings[:ns + nt]
    train_y = np.asarray([int(x["label"]) for x in source + support], dtype=int)
    scaler = StandardScaler().fit(train_x)
    head = LogisticRegression(C=0.1, max_iter=4000, solver="lbfgs", random_state=seed)
    head.fit(scaler.transform(train_x), train_y,
             sample_weight=np.concatenate([np.ones(ns), np.full(nt, 10.0)]))
    query_pred = head.predict(scaler.transform(embeddings[ns + nt:]))
    query_y = np.asarray([int(x["label"]) for x in query], dtype=int)
    target_metric = metric(query_y, query_pred)
    pred_path = OUTPUT / "predictions" / f"{name}_seed{seed}.csv"
    with pred_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "measurement_id", "bearing_code", "true", "prediction", "correct"])
        writer.writeheader()
        for item, y, pred in zip(query, query_y, query_pred):
            writer.writerow({"model": name, "seed": seed, "measurement_id": item["measurement_id"],
                             "bearing_code": item["bearing_code"], "true": int(y),
                             "prediction": int(pred), "correct": int(y == pred)})
    run = {
        "model": name, "seed": seed, "source_train_measurements": int(len(train_idx)),
        "source_val_measurements": int(len(val_idx)), "best_epoch": best_epoch,
        "best_source_val_macro_f1": best_score,
        "source_val_metric_at_best": history[best_epoch - 1],
        "target_metric": target_metric,
        "target_query_used_for_checkpoint_or_selection": False,
        "checkpoint_sha256": sha256(checkpoint), "prediction_sha256": sha256(pred_path),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("COMPLETED", json.dumps({"model": name, "seed": seed, "best_val": best_score,
                                    "target_accuracy": target_metric["accuracy"],
                                    "target_OR": target_metric["recall_OR"]}), flush=True)


def summarize() -> None:
    runs = []
    missing = []
    for name in MODELS:
        for seed in SEEDS:
            path = OUTPUT / "runs" / f"{name}_seed{seed}.json"
            if not path.exists(): missing.append(str(path))
            else: runs.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise RuntimeError("Missing preregistered runs:\n" + "\n".join(missing))
    rows = []
    summary = {}
    metrics = ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR", "errors")
    for run in runs:
        rows.append({"model": run["model"], "seed": run["seed"],
                     "best_epoch": run["best_epoch"],
                     "best_source_val_macro_f1": run["best_source_val_macro_f1"],
                     **run["target_metric"]})
    frame = pd.DataFrame(rows)
    for name in MODELS:
        group = frame[frame.model == name]
        summary[name] = {
            key: {"mean": float(group[key].mean()), "std": float(group[key].std(ddof=1)),
                  "min": float(group[key].min()), "max": float(group[key].max())}
            for key in metrics
        }
        summary[name]["source_val_macro_f1"] = {
            "mean": float(group.best_source_val_macro_f1.mean()),
            "std": float(group.best_source_val_macro_f1.std(ddof=1)),
            "min": float(group.best_source_val_macro_f1.min()),
            "max": float(group.best_source_val_macro_f1.max()),
        }
        summary[name]["selected_epoch"] = {
            "mean": float(group.best_epoch.mean()), "min": int(group.best_epoch.min()),
            "max": int(group.best_epoch.max()),
        }
    frame.to_csv(OUTPUT / "seed_metrics.csv", index=False)
    result = {
        "experiment": "PU_MULTISEED_STABILITY", "status": "COMPLETE",
        "epistemic_status": "post-hoc source-selected multi-seed reproducibility audit",
        "seeds": list(SEEDS), "models": list(MODELS), "runs": len(runs),
        "source_split": {"train": 816, "validation": 204,
                         "measurement_level": True, "bearing_code_isolated": False},
        "target_query_used_for_selection": False,
        "summary": summary, "protocol_sha256": sha256(OUTPUT / "protocol_frozen.md"),
    }
    result_path = OUTPUT / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "evaluation.lock").write_text(json.dumps({"complete": True,
                                                          "result_sha256": sha256(result_path)}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(); return
    if args.model is None or args.seed is None:
        raise SystemExit("Specify --model and --seed, or --summarize")
    train_one(args.model, args.seed)


if __name__ == "__main__":
    main()
