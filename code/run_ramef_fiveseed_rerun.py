"""Strict two-stage five-seed RAMEF reproducibility rerun.

Stage A fixes and trains seeds 20260911--20260915 using source data only.
Stage B verifies the frozen checkpoint ledger and evaluates every seed on the
historically accessed but fixed PU development cohort.  No target-ranked seed
selection is implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from bottleneck_deconvolution import sha256
from prepare_splits import scan_records
from run_core_component_ablation import make_u2
from run_fair_baseline_pu import fixed_source_split
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_pu_eightblock_fair_baselines import wave_cache
from run_pu_multiseed_stability import infer_eight_blocks, validation_logits
from run_racp_measurement_final_once import load_query
from run_tser_hb_support_p0 import support_only


SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)
PROTOCOL = Path("RAMEF_FIVESEED_RERUN_PROTOCOL.md")
OUTPUT = Path("outputs/staged/RAMEF_FIVESEED_RERUN_20260921")
CONFIG = Path("frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")
SPLIT = Path("split_independent_measurement.json")
SOURCE_WAVES = Path("outputs/staged/PU_MULTISEED_STABILITY/source_centre_waves.pt")
EIGHTBLOCK_WAVES = Path("outputs/staged/PU_EIGHTBLOCK_FAIR_BASELINES/eightblock_waves.pt")
SOURCE_PHYSICAL = Path("outputs/staged/RACP_OR_INFORMATION_P0/authorised_source_support_features.npz")
QUERY_PHYSICAL = Path("outputs/staged/RACP_MEASUREMENT_FINAL/query_features/authorised_source_support_features.npz")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(labels: np.ndarray, prediction: np.ndarray) -> dict:
    recalls = recall_score(labels, prediction, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(labels == prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall_N": float(recalls[0]),
        "recall_IR": float(recalls[1]),
        "recall_OR": float(recalls[2]),
        "errors": int(np.sum(labels != prediction)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist(),
    }


def model_on(device: torch.device):
    return make_u2("hz", CONFIG).to(device)


def checkpoint_path(seed: int) -> Path:
    return OUTPUT / "checkpoints" / f"u2_hz_seed{seed}.pt"


def source_records_and_waves() -> tuple[list[dict], torch.Tensor]:
    records = scan_records(Path("D:/Paderborn"))
    source = sorted([row for row in records if row["domain"] in SOURCE_DOMAINS],
                    key=lambda row: row["measurement_id"])
    payload = torch.load(SOURCE_WAVES, map_location="cpu", weights_only=False)
    if payload["measurement_ids"] != [row["measurement_id"] for row in source]:
        raise RuntimeError("Source centre-wave cache identity mismatch")
    return source, payload["waveforms"].float()


def train_seed(seed: int, source: list[dict], waves: torch.Tensor,
               device: torch.device) -> None:
    path = checkpoint_path(seed)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing checkpoint: {path}")
    labels = np.asarray([int(row["label"]) for row in source], dtype=int)
    train_idx, val_idx = fixed_source_split(source)
    rpms = torch.tensor([float(row["rpm"]) for row in source], dtype=torch.float32)
    seed_all(seed)
    model = model_on(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(waves[train_idx], torch.tensor(labels[train_idx]), rpms[train_idx])
    loader = DataLoader(dataset, batch_size=12, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    best_score = -1.0
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(12):
        model.train()
        losses = []
        for block, target, rpm in loader:
            block, target, rpm = block.to(device), target.to(device), rpm.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(block, rpm)["logits"]
                loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        val_logits = validation_logits(model, "u2_hz", waves, rpms, val_idx, device)
        val_pred = val_logits.argmax(1)
        val_f1 = float(f1_score(labels[val_idx], val_pred, labels=[0, 1, 2],
                                average="macro", zero_division=0))
        history.append({"seed": seed, "epoch": epoch + 1,
                        "train_loss": float(np.mean(losses)),
                        "source_val_macro_f1": val_f1})
        print(f"seed={seed} epoch={epoch + 1}/12 val_f1={val_f1:.6f}", flush=True)
        if val_f1 > best_score:
            best_score = val_f1
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    torch.save({"state": best_state, "seed": seed, "model": "u2_hz",
                "best_epoch": best_epoch,
                "best_source_val_macro_f1": best_score,
                "history": history}, path)


def stage_source_freeze() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "checkpoints").mkdir()
    shutil.copy2(PROTOCOL, OUTPUT / "protocol_frozen.md")
    registry = {
        "experiment": "RAMEF_FIVESEED_RERUN_20260921",
        "stage": "SEEDS_FIXED_BEFORE_QUERY",
        "seeds": list(SEEDS),
        "query_opened": False,
        "protocol_sha256": sha256(OUTPUT / "protocol_frozen.md"),
        "split_sha256": sha256(SPLIT),
    }
    registry_path = OUTPUT / "seed_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (OUTPUT / "seed_registry.lock").write_text(
        json.dumps({"seed_registry_sha256": sha256(registry_path),
                    "seeds": list(SEEDS)}, indent=2), encoding="utf-8")

    source, waves = source_records_and_waves()
    labels = np.asarray([int(row["label"]) for row in source], dtype=int)
    _, val_idx = fixed_source_split(source)
    rpms = torch.tensor([float(row["rpm"]) for row in source], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        train_seed(seed, source, waves, device)

    source_rows = []
    history_rows = []
    checkpoints = {}
    for seed in SEEDS:
        path = checkpoint_path(seed)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = model_on(device)
        model.load_state_dict(payload["state"])
        logits = validation_logits(model, "u2_hz", waves, rpms, val_idx, device)
        pred = logits.argmax(1)
        source_rows.append({
            "seed": seed,
            "best_epoch": int(payload["best_epoch"]),
            "source_val_accuracy": float(np.mean(pred == labels[val_idx])),
            "source_val_macro_f1": float(f1_score(labels[val_idx], pred, labels=[0, 1, 2],
                                                    average="macro", zero_division=0)),
            "checkpoint_sha256": sha256(path),
        })
        history_rows.extend(payload["history"])
        checkpoints[str(seed)] = sha256(path)
    write_csv(OUTPUT / "source_metrics.csv", source_rows)
    write_csv(OUTPUT / "training_history.csv", history_rows)
    source_freeze = {
        "experiment": "RAMEF_FIVESEED_RERUN_20260921",
        "stage": "SOURCE_TRAINING_COMPLETE",
        "query_opened": False,
        "seeds": list(SEEDS),
        "checkpoints": checkpoints,
        "source_metrics_sha256": sha256(OUTPUT / "source_metrics.csv"),
        "training_history_sha256": sha256(OUTPUT / "training_history.csv"),
    }
    freeze_path = OUTPUT / "source_freeze.json"
    freeze_path.write_text(json.dumps(source_freeze, indent=2), encoding="utf-8")
    (OUTPUT / "source_freeze.lock").write_text(
        json.dumps({"source_freeze_sha256": sha256(freeze_path),
                    "checkpoints": checkpoints}, indent=2), encoding="utf-8")
    print(json.dumps(source_freeze, indent=2), flush=True)


def load_physical(ids: list[str], path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    index = {value: idx for idx, value in enumerate(payload["ids"].tolist())}
    missing = [value for value in ids if value not in index]
    if missing:
        raise RuntimeError(f"Missing physical descriptors: {missing[:3]}")
    return payload["physical"][[index[value] for value in ids]].astype(np.float32)


def fit_predict(source_x, source_y, support_x, support_y, query_x, seed):
    features = np.vstack([source_x, support_x])
    labels = np.concatenate([source_y, support_y])
    scaler = StandardScaler().fit(features)
    head = LogisticRegression(C=0.1, solver="lbfgs", max_iter=4000,
                              random_state=seed)
    head.fit(scaler.transform(features), labels,
             sample_weight=np.concatenate([np.ones(len(source_x)),
                                           np.full(len(support_x), 10.0)]))
    return head.predict(scaler.transform(query_x))


def summarise(rows: list[dict], method: str) -> dict:
    group = [row for row in rows if row["method"] == method]
    fields = ["accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR", "errors"]
    result = {}
    for field in fields:
        values = np.asarray([row[field] for row in group], dtype=float)
        result[field] = {"mean": float(values.mean()),
                         "std": float(values.std(ddof=1)),
                         "min": float(values.min()),
                         "max": float(values.max())}
    return result


def stage_evaluate() -> None:
    if (OUTPUT / "target_evaluation.lock").exists():
        raise RuntimeError("Target evaluation is already locked")
    registry_path = OUTPUT / "seed_registry.json"
    registry_lock = json.loads((OUTPUT / "seed_registry.lock").read_text(encoding="utf-8"))
    if registry_lock["seed_registry_sha256"] != sha256(registry_path):
        raise RuntimeError("Seed registry changed after freeze")
    freeze_path = OUTPUT / "source_freeze.json"
    freeze_lock = json.loads((OUTPUT / "source_freeze.lock").read_text(encoding="utf-8"))
    if freeze_lock["source_freeze_sha256"] != sha256(freeze_path):
        raise RuntimeError("Source freeze changed after query gate")
    source_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if tuple(source_freeze["seeds"]) != SEEDS:
        raise RuntimeError("Frozen seed list mismatch")
    for seed in SEEDS:
        if source_freeze["checkpoints"][str(seed)] != sha256(checkpoint_path(seed)):
            raise RuntimeError(f"Checkpoint changed after source freeze: {seed}")

    source = sorted([row for row in scan_records(Path("D:/Paderborn"))
                     if row["domain"] in SOURCE_DOMAINS], key=lambda row: row["measurement_id"])
    support, _ = support_only(SPLIT)
    query = load_query(SPLIT)
    records = source + support + query
    waves = wave_cache(records, EIGHTBLOCK_WAVES)
    rpms = torch.tensor([float(row["rpm"]) for row in records], dtype=torch.float32)
    source_y = np.asarray([int(row["label"]) for row in source], dtype=int)
    support_y = np.asarray([int(row["label"]) for row in support], dtype=int)
    query_y = np.asarray([int(row["label"]) for row in query], dtype=int)
    physical_ss = load_physical([row["measurement_id"] for row in source + support], SOURCE_PHYSICAL)
    physical_q = load_physical([row["measurement_id"] for row in query], QUERY_PHYSICAL)
    ns, nt = len(source), len(support)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows = []
    prediction_rows = []
    for seed in SEEDS:
        payload = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
        model = model_on(device)
        model.load_state_dict(payload["state"])
        embeddings = infer_eight_blocks(model, "u2_hz", waves, rpms, device)
        source_embedding = embeddings[:ns]
        support_embedding = embeddings[ns:ns + nt]
        query_embedding = embeddings[ns + nt:]
        for method in ("U2_Hz_DM8", "RAMEF_F"):
            if method == "U2_Hz_DM8":
                source_x, support_x, query_x = source_embedding, support_embedding, query_embedding
            else:
                source_x = np.concatenate([source_embedding, physical_ss[:ns]], axis=1)
                support_x = np.concatenate([support_embedding, physical_ss[ns:]], axis=1)
                query_x = np.concatenate([query_embedding, physical_q], axis=1)
            prediction = fit_predict(source_x, source_y, support_x, support_y, query_x, seed)
            values = metric(query_y, prediction)
            metric_rows.append({"seed": seed, "method": method, **values})
            for row, truth, predicted in zip(query, query_y, prediction):
                prediction_rows.append({
                    "seed": seed,
                    "method": method,
                    "measurement_id": row["measurement_id"],
                    "bearing_code": row["bearing_code"],
                    "true": int(truth),
                    "prediction": int(predicted),
                    "correct": int(truth == predicted),
                })
        print(f"evaluated seed={seed}", flush=True)

    write_csv(OUTPUT / "target_seed_metrics.csv", metric_rows)
    write_csv(OUTPUT / "target_predictions.csv", prediction_rows)
    u2_summary = summarise(metric_rows, "U2_Hz_DM8")
    ramef_summary = summarise(metric_rows, "RAMEF_F")
    accuracy_by_seed = {}
    for row in metric_rows:
        accuracy_by_seed.setdefault(int(row["seed"]), {})[row["method"]] = float(row["accuracy"])
    deltas = {seed: values["RAMEF_F"] - values["U2_Hz_DM8"]
              for seed, values in accuracy_by_seed.items()}
    result = {
        "experiment": "RAMEF_FIVESEED_RERUN_20260921",
        "status": "COMPLETE_FIXED_FIVESEED_RERUN",
        "epistemic_status": "pre-fixed new seeds; historically accessed fixed PU development cohort",
        "seeds": list(SEEDS),
        "all_declared_seeds_reported": True,
        "query_used_for_seed_selection": False,
        "pu_query_historically_accessed_by_broader_project": True,
        "U2_Hz_DM8": u2_summary,
        "RAMEF_F": ramef_summary,
        "RAMEF_F_minus_U2_accuracy_by_seed": {str(key): float(value) for key, value in deltas.items()},
        "improved_seeds": int(sum(value > 0 for value in deltas.values())),
        "tied_seeds": int(sum(value == 0 for value in deltas.values())),
        "degraded_seeds": int(sum(value < 0 for value in deltas.values())),
        "protocol_sha256": sha256(OUTPUT / "protocol_frozen.md"),
        "seed_registry_sha256": sha256(registry_path),
        "source_freeze_sha256": sha256(freeze_path),
        "split_sha256": sha256(SPLIT),
    }
    result_path = OUTPUT / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = [
        "# RAMEF independent five-seed rerun",
        "",
        f"- Fixed seeds: {list(SEEDS)}.",
        f"- U2-Hz-DM8 accuracy: {100*u2_summary['accuracy']['mean']:.2f}% ± {100*u2_summary['accuracy']['std']:.2f}%.",
        f"- RAMEF-F accuracy: {100*ramef_summary['accuracy']['mean']:.2f}% ± {100*ramef_summary['accuracy']['std']:.2f}%.",
        f"- RAMEF-F Macro-F1: {100*ramef_summary['macro_f1']['mean']:.2f}% ± {100*ramef_summary['macro_f1']['std']:.2f}%.",
        f"- RAMEF-F versus U2-Hz-DM8: improved {result['improved_seeds']}, tied {result['tied_seeds']}, degraded {result['degraded_seeds']} seeds.",
        "",
        "All five seeds were fixed before this rerun opened query data and all five are reported. The fixed PU cohort had been accessed earlier by the broader project, so this remains a development-cohort reproducibility audit.",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "target_evaluation.lock").write_text(
        json.dumps({"completed": True,
                    "result_sha256": sha256(result_path),
                    "seed_registry_sha256": sha256(registry_path),
                    "source_freeze_sha256": sha256(freeze_path)}, indent=2),
        encoding="utf-8")
    artifacts = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.csv":
            artifacts.append({"relative_path": path.relative_to(OUTPUT).as_posix(),
                              "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(OUTPUT / "artifact_sha256.csv", artifacts)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("source-freeze", "evaluate"), required=True)
    args = parser.parse_args()
    if args.stage == "source-freeze":
        stage_source_freeze()
    else:
        stage_evaluate()


if __name__ == "__main__":
    main()
