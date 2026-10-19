"""Strict DS-support-only diagnostic of freezing versus U2-Hz fine-tuning.

The audit never loads a DS query row or waveform. Each fold holds out one of
the 15 authorised target-support measurements, adapts on the other 14 using all
eight registered blocks, and predicts the held measurement by mean block
logits. A fixed code-balanced source safety panel quantifies forgetting.

This is an encoder-adaptation diagnostic, not a replacement for RAMEF's
source+support external linear head. It directly tests whether scarce target
support provides a safe signal for neural parameter updates.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from bottleneck_deconvolution import load_vibration, sha256
from prepare_splits import scan_records
from racp_or import WINDOW, uniform_nonoverlap_starts
from run_a1_u2hz_confirmation import CHECKPOINT, CHECKPOINT_SHA
from run_core_component_ablation import make_u2
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_tser_hb_support_p0 import support_only


STAGES = ("frozen", "head_only", "last_rsu", "full")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def standardise(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32).copy()
    return (block - block.mean()) / max(float(block.std()), 1e-6)


def measurement_blocks(rows: list[dict], cache: Path) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [row["measurement_id"] for row in rows]
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload["ids"] != ids: raise RuntimeError("Stale waveform cache")
        return payload["waves"], payload["rpms"]
    waves, rpms = [], []
    for index, row in enumerate(rows):
        signal = np.asarray(load_vibration(row["path"], row.get("signal_key", "signal")),
                            dtype=np.float32).reshape(-1)
        starts = uniform_nonoverlap_starts(len(signal))
        waves.append(np.stack([standardise(signal[start:start + WINDOW]) for start in starts]))
        rpms.append(float(row["rpm"]))
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            print(f"waveform cache {index + 1}/{len(rows)}", flush=True)
    wave_tensor = torch.tensor(np.stack(waves), dtype=torch.float32)
    rpm_tensor = torch.tensor(rpms, dtype=torch.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "waves": wave_tensor, "rpms": rpm_tensor}, cache)
    return wave_tensor, rpm_tensor


def build_model(device: torch.device):
    model = make_u2("hz", "frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state"])
    return model.to(device)


def set_trainable(model, stage: str) -> list[str]:
    for parameter in model.parameters(): parameter.requires_grad = False
    modules = []
    if stage in ("head_only", "last_rsu", "full"): modules.append(model.linear_classifier)
    if stage == "last_rsu": modules.append(model.encoder.encode_modules[3])
    if stage == "full": modules.append(model)
    for module in modules:
        for parameter in module.parameters(): parameter.requires_grad = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def train_fold(model, stage: str, waves: torch.Tensor, labels: np.ndarray,
               rpms: torch.Tensor, seed: int, device: torch.device) -> float:
    names = set_trainable(model, stage)
    if stage == "frozen": return 0.0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    x = waves.reshape(-1, WINDOW); y = torch.tensor(np.repeat(labels, 8), dtype=torch.long)
    rpm = rpms.repeat_interleave(8)
    loader = DataLoader(TensorDataset(x, y, rpm), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    head = [parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("linear_classifier")]
    encoder = [parameter for name, parameter in model.named_parameters()
               if parameter.requires_grad and not name.startswith("linear_classifier")]
    groups = [{"params": head, "lr": 2e-4}]
    if encoder:
        groups.append({"params": encoder, "lr": 2e-5 if stage == "last_rsu" else 5e-6})
    optimiser = torch.optim.AdamW(groups, weight_decay=5e-5)
    model.train()
    for _ in range(8):
        for batch_x, batch_y, batch_rpm in loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device), batch_rpm.to(device))["logits"]
            loss = F.cross_entropy(logits, batch_y.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
    numerator = 0.0; denominator = 0.0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in initial:
                numerator += float(torch.sum((parameter - initial[name]) ** 2))
                denominator += float(torch.sum(initial[name] ** 2))
    return float(np.sqrt(numerator / max(denominator, 1e-20)))


@torch.no_grad()
def predict_measurements(model, waves: torch.Tensor, rpms: torch.Tensor,
                         device: torch.device, batch: int = 32) -> np.ndarray:
    model.eval(); flat = waves.reshape(-1, WINDOW); flat_rpm = rpms.repeat_interleave(8)
    logits = []
    for start in range(0, len(flat), batch):
        logits.append(model(flat[start:start + batch].to(device),
                            flat_rpm[start:start + batch].to(device))["logits"].cpu())
    return torch.cat(logits).reshape(len(waves), 8, 3).mean(1).argmax(1).numpy()


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(y == pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro",
                                       zero_division=0)),
            "recall_N": float(recall[0]), "recall_IR": float(recall[1]),
            "recall_OR": float(recall[2]),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def source_safety_panel(source: list[dict]) -> list[dict]:
    """One deterministic measurement per domain/class/code where available."""
    selected = []
    seen = set()
    for row in source:
        key = (row["domain"], int(row["label"]), row["bearing_code"])
        if key not in seen:
            selected.append(row); seen.add(key)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/staged/U2HZ_FREEZE_SUPPORT_AUDIT"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if sha256(Path(CHECKPOINT)) != CHECKPOINT_SHA: raise RuntimeError("Checkpoint mismatch")
    support, query_count = support_only(Path("split_independent_measurement.json"))
    source = sorted([row for row in scan_records(Path("D:/Paderborn"))
                     if row["domain"] in SOURCE_DOMAINS], key=lambda row: row["measurement_id"])
    safety = source_safety_panel(source)
    all_rows = support + safety
    waves, rpms = measurement_blocks(all_rows, args.output / "support_source_safety_waves.pt")
    support_waves, safety_waves = waves[:15], waves[15:]
    support_rpms, safety_rpms = rpms[:15], rpms[15:]
    support_y = np.asarray([int(row["label"]) for row in support])
    safety_y = np.asarray([int(row["label"]) for row in safety])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows, fold_rows = [], []
    for held in range(15):
        keep = np.asarray([i for i in range(15) if i != held])
        for stage in STAGES:
            model = build_model(device); drift = train_fold(
                model, stage, support_waves[keep], support_y[keep], support_rpms[keep],
                20261300 + held, device)
            target_pred = int(predict_measurements(
                model, support_waves[held:held + 1], support_rpms[held:held + 1], device)[0])
            safety_pred = predict_measurements(model, safety_waves, safety_rpms, device)
            safety_value = metric(safety_y, safety_pred)
            row = support[held]
            prediction_rows.append({"fold": held, "stage": stage,
                                    "measurement_id": row["measurement_id"],
                                    "bearing_code": row["bearing_code"],
                                    "true": int(support_y[held]), "prediction": target_pred,
                                    "correct": int(target_pred == support_y[held]),
                                    "source_safety_accuracy": safety_value["accuracy"],
                                    "source_safety_N_recall": safety_value["recall_N"],
                                    "source_safety_OR_recall": safety_value["recall_OR"],
                                    "relative_parameter_drift": drift})
            fold_rows.append({"fold": held, "stage": stage, "trainable": stage,
                              "source_safety_accuracy": safety_value["accuracy"],
                              "source_safety_N_recall": safety_value["recall_N"],
                              "source_safety_IR_recall": safety_value["recall_IR"],
                              "source_safety_OR_recall": safety_value["recall_OR"],
                              "relative_parameter_drift": drift})
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"support fold {held + 1}/15", flush=True)
    summary = {}
    for stage in STAGES:
        local = [row for row in prediction_rows if row["stage"] == stage]
        y = np.asarray([row["true"] for row in local]); pred = np.asarray([row["prediction"] for row in local])
        value = metric(y, pred)
        value["source_safety_accuracy_mean"] = float(np.mean([row["source_safety_accuracy"] for row in local]))
        value["source_safety_N_recall_min"] = float(min(row["source_safety_N_recall"] for row in local))
        value["source_safety_OR_recall_min"] = float(min(row["source_safety_OR_recall"] for row in local))
        value["relative_parameter_drift_mean"] = float(np.mean([row["relative_parameter_drift"] for row in local]))
        summary[stage] = value
    result = {"experiment": "U2HZ_FREEZE_SUPPORT_AUDIT",
              "status": "COMPLETE_QUERY_FREE",
              "interpretation_scope": "direct U2-Hz neural-head adaptation diagnostic; not the external RAMEF classifier",
              "metrics": summary,
              "source_safety_measurements": len(safety),
              "firewall": {"ds_query_rows_in_split": query_count,
                           "ds_query_measurements_loaded": 0,
                           "ds_query_waveforms_loaded": False,
                           "ds_query_labels_used": False},
              "checkpoint_sha256": CHECKPOINT_SHA,
              "split_sha256": sha256(Path("split_independent_measurement.json"))}
    write_csv(args.output / "support_predictions.csv", prediction_rows)
    write_csv(args.output / "fold_safety.csv", fold_rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
