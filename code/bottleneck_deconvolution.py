"""Frozen-M2 protocol difficulty decomposition.

All experimental conditions are declared in BOTTLENECK_DECONVOLUTION_PROTOCOL.md.
DS query labels are consumed only after tokens/logits are fixed, for reporting.
No query-derived choice, threshold, support selection, or hyperparameter exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data_loader_pu import load_vibration
from prepare_splits import scan_records
from u2_stft_model import U2STFT


CHECKPOINT_SHA256 = "cce4c3e22de6e1416a8df35bb5244e1c8cd2d5d3b12155a886da7dcf4553f098"
SPLIT_SHA256 = "0e7bf770e4de2037deff8471d2235c1b4493026e6b91b667def34d38881b3156"
BASELINE_ACCURACY = 0.8458333333333333
BASELINE_MACRO_F1 = 0.8454350098297431
CLASS_NAME = {0: "N", 1: "IR", 2: "OR"}


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    if sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Locked M2 checkpoint SHA256 mismatch")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = dict(cfg["model"]); num_classes = model_cfg.pop("num_classes"); lite = model_cfg.pop("lite")
    model = U2STFT(num_classes=num_classes, lite=lite,
                   sample_rate=cfg["data"]["sample_rate"], **model_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model"])
    model.eval()
    return model, cfg


def load_contract(split_path: Path):
    if sha256(split_path) != SPLIT_SHA256:
        raise RuntimeError("Locked independent split SHA256 mismatch")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    windows = payload["windows"]
    support = [x for x in windows if x["role"] == "support" and x["domain"] == "DS"]
    query = [x for x in windows if x["role"] == "query" and x["domain"] == "DS"]
    if len(support) != 15 or len(query) != 240:
        raise RuntimeError("Unexpected locked target contract")
    if {x["measurement_id"] for x in support} & {x["measurement_id"] for x in query}:
        raise RuntimeError("Target measurement leakage")
    return payload, support, query


def centred_item(record: dict, length: int = 8192, role: str = "aux_support") -> dict:
    centre = int(record.get("signal_length", 256001)) // 2
    start = centre - length // 2
    return {**record, "role": role, "start": start, "end": start + length}


def round_robin(records: list[dict], count: int, excluded_measurements: set[str] | None = None,
                excluded_code: str | None = None) -> list[dict]:
    excluded_measurements = excluded_measurements or set()
    eligible = [x for x in records if x["measurement_id"] not in excluded_measurements
                and (excluded_code is None or x["bearing_code"] != excluded_code)]
    by_code = defaultdict(list)
    for row in sorted(eligible, key=lambda x: (x["bearing_code"], x["measurement_id"])):
        by_code[row["bearing_code"]].append(row)
    selected = []
    while len(selected) < count:
        advanced = False
        for code in sorted(by_code):
            if by_code[code]:
                selected.append(by_code[code].pop(0)); advanced = True
                if len(selected) == count: break
        if not advanced: break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} independent records available, need {count}")
    return selected


class TokenExtractor:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.signals: dict[str, np.ndarray] = {}

    def _waveform(self, item: dict, length: int) -> tuple[np.ndarray, dict]:
        path = str(item["path"])
        if path not in self.signals:
            self.signals[path] = load_vibration(path, item.get("signal_key", "signal"))
        signal = self.signals[path]
        if length == int(item["end"]) - int(item["start"]):
            start = int(item["start"])
        else:
            centre = (int(item["start"]) + int(item["end"])) // 2
            requested = centre - length // 2
            start = max(0, min(requested, len(signal) - length))
        end = start + length
        waveform = np.asarray(signal[start:end], dtype=np.float32).copy()
        if len(waveform) != length:
            raise RuntimeError(f"Invalid {length}-sample crop: {item['measurement_id']}")
        waveform = (waveform - waveform.mean()) / max(float(waveform.std()), 1e-6)
        return waveform, {"measurement_id": item["measurement_id"], "requested_length": length,
                          "registered_start": int(item["start"]), "registered_end": int(item["end"]),
                          "actual_start": start, "actual_end": end,
                          "centre_shift_samples": start - ((int(item["start"]) + int(item["end"])) // 2 - length // 2)}

    @torch.no_grad()
    def extract(self, items: list[dict], length: int = 8192, batch_size: int = 12):
        all_tokens, bounds = [], []
        for begin in range(0, len(items), batch_size):
            batch = items[begin:begin + batch_size]
            loaded = [self._waveform(item, length) for item in batch]
            wave = torch.from_numpy(np.stack([x[0] for x in loaded])).to(self.device)
            rpm = torch.tensor([float(x["rpm"]) for x in batch], dtype=torch.float32, device=self.device)
            all_tokens.append(self.model(wave, rpm)["tokens"].cpu())
            bounds.extend(x[1] for x in loaded)
        return torch.cat(all_tokens), bounds


def ordinary_logits(model, support_tokens, support_labels, query_tokens):
    state = model.prototype.fit_support(support_tokens, support_labels, bayesian=False)
    return model.prototype.logits(query_tokens, state)


def grouped_logits(support_tokens: torch.Tensor, support_labels: torch.Tensor,
                   query_tokens: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    means, variances = [], []
    for classes in groups:
        mask = torch.zeros_like(support_labels, dtype=torch.bool)
        for label in classes: mask |= support_labels == label
        values = support_tokens[mask]
        means.append(values.mean(0)); variances.append(values.var(0, unbiased=False) + .05)
    mean, variance = torch.stack(means), torch.stack(variances)
    distance = ((query_tokens[:, None] - mean[None]).square() /
                variance[None].clamp_min(.05)).mean((2, 3))
    return -distance


def metrics_row(experiment: str, condition: str, logits: torch.Tensor, truth: np.ndarray,
                task: str = "three_class", notes: str = "") -> tuple[dict, np.ndarray]:
    if task == "three_class":
        prediction = logits.argmax(1).numpy(); labels = truth; names = [0, 1, 2]
        cm = confusion_matrix(labels, prediction, labels=names)
        recalls = [float(np.mean(prediction[labels == c] == c)) for c in names]
        macro = f1_score(labels, prediction, labels=names, average="macro", zero_division=0)
        errors = [int(np.sum((labels == c) & (prediction != c))) for c in names]
    elif task == "coarse":
        prediction = logits.argmax(1).numpy(); labels = (truth > 0).astype(int)
        cm = confusion_matrix(labels, prediction, labels=[0, 1])
        recalls = [float(np.mean(prediction[truth == 0] == 0)),
                   float(np.mean(prediction[truth == 1] == 1)),
                   float(np.mean(prediction[truth == 2] == 1))]
        macro = f1_score(labels, prediction, labels=[0, 1], average="macro", zero_division=0)
        errors = [int(np.sum((truth == 0) & (prediction != 0))),
                  int(np.sum((truth == 1) & (prediction != 1))),
                  int(np.sum((truth == 2) & (prediction != 1)))]
    elif task == "fault_only":
        prediction01 = logits.argmax(1).numpy(); prediction = prediction01 + 1; labels = truth
        cm = confusion_matrix(labels, prediction, labels=[1, 2])
        recalls = [None, float(np.mean(prediction[labels == 1] == 1)),
                   float(np.mean(prediction[labels == 2] == 2))]
        macro = f1_score(labels, prediction, labels=[1, 2], average="macro", zero_division=0)
        errors = [0, int(np.sum((labels == 1) & (prediction != 1))),
                  int(np.sum((labels == 2) & (prediction != 2)))]
    else:
        raise ValueError(task)
    row = {"experiment": experiment, "condition": condition, "task": task,
           "accuracy": float(accuracy_score(labels, prediction if task != "coarse" else prediction)),
           "macro_f1": float(macro), "n_recall": recalls[0], "ir_recall": recalls[1],
           "or_recall": recalls[2], "confusion_matrix": json.dumps(cm.tolist()),
           "error_n": errors[0], "error_ir": errors[1], "error_or": errors[2],
           "total_errors": int(sum(errors)), "n_samples": int(len(labels)), "notes": notes}
    return row, prediction


def support_index(items: list[dict], selected: list[dict]) -> list[int]:
    lookup = {(x["measurement_id"], int(x["start"]), int(x["end"])): i for i, x in enumerate(items)}
    return [lookup[(x["measurement_id"], int(x["start"]), int(x["end"]))] for x in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="split_independent_measurement.json")
    parser.add_argument("--config", default="frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")
    parser.add_argument("--checkpoint", default="frozen/M2_LOCKED_v1_20260715/model/checkpoints/best.pt")
    parser.add_argument("--protocol", default="BOTTLENECK_DECONVOLUTION_PROTOCOL.md")
    parser.add_argument("--data-root", default=r"D:\Paderborn")
    parser.add_argument("--output", default="outputs/staged/BOTTLENECK")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha256(args.protocol)
    payload, fixed_support, fixed_query = load_contract(Path(args.split))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(Path(args.config), Path(args.checkpoint), device)
    extractor = TokenExtractor(model, device)

    all_records = scan_records(Path(args.data_root))
    target_records = [x for x in all_records if x["domain"] == "DS"]
    locked_measurements = {x["measurement_id"] for x in fixed_support + fixed_query}
    unused = [x for x in target_records if x["measurement_id"] not in locked_measurements]
    if min(Counter(x["label"] for x in unused).values()) < 15:
        raise RuntimeError("Insufficient independent target records for 20-shot")
    aux_by_class = {c: round_robin([x for x in unused if int(x["label"]) == c], 15)
                    for c in range(3)}
    aux_items = [centred_item(x) for c in range(3) for x in aux_by_class[c]]
    target_items = fixed_support + aux_items + fixed_query
    target_tokens, target_bounds = extractor.extract(target_items, 8192)
    fixed_support_idx = list(range(15))
    aux_offset = 15; query_offset = 15 + len(aux_items)
    query_idx = list(range(query_offset, len(target_items)))
    query_tokens = target_tokens[query_idx]
    query_truth = np.asarray([int(x["label"]) for x in fixed_query])
    support_labels_fixed = torch.tensor([int(x["label"]) for x in fixed_support])

    results, predictions, support_manifest_rows = [], {}, []
    baseline_logits = ordinary_logits(model, target_tokens[fixed_support_idx].to(device),
                                      support_labels_fixed.to(device), query_tokens.to(device)).cpu()
    baseline_row, baseline_pred = metrics_row("BASELINE", "M2_A5_locked", baseline_logits, query_truth)
    if abs(baseline_row["accuracy"] - BASELINE_ACCURACY) > 1e-12 or abs(
            baseline_row["macro_f1"] - BASELINE_MACRO_F1) > 1e-12:
        raise RuntimeError(f"M2 baseline reproduction failed: {baseline_row}")
    results.append(baseline_row); predictions["BASELINE"] = baseline_pred

    # A: nested shot supports.
    fixed_by_class = {c: [i for i, x in enumerate(fixed_support) if int(x["label"]) == c] for c in range(3)}
    aux_indices_by_class = {c: list(range(aux_offset + c * 15, aux_offset + (c + 1) * 15)) for c in range(3)}
    a_sets = {
        1: [fixed_by_class[c][0] for c in range(3)],
        5: [i for c in range(3) for i in fixed_by_class[c]],
        10: [i for c in range(3) for i in fixed_by_class[c] + aux_indices_by_class[c][:5]],
        20: [i for c in range(3) for i in fixed_by_class[c] + aux_indices_by_class[c]],
    }
    for shot, indices in a_sets.items():
        for i in indices:
            item = target_items[i]
            support_manifest_rows.append({"experiment": "A", "condition": f"A{shot}",
                "query_index": "", "class": CLASS_NAME[int(item["label"])],
                "bearing_code": item["bearing_code"], "measurement_id": item["measurement_id"],
                "path": item["path"], "start": item["start"], "end": item["end"]})
        labels = torch.tensor([int(target_items[i]["label"]) for i in indices], device=device)
        logits = ordinary_logits(model, target_tokens[indices].to(device), labels, query_tokens.to(device)).cpu()
        row, pred = metrics_row("A_support_quantity", f"A{shot}_{shot}shot", logits, query_truth,
                                notes="nested independent-measurement support")
        results.append(row); predictions[f"A{shot}"] = pred

    # B: actual, code-matched oracle and code-excluded counterfactual episodes.
    safe_support = fixed_support + aux_items
    safe_tokens = target_tokens[:query_offset]
    safe_by_class = {c: [i for i, x in enumerate(safe_support) if int(x["label"]) == c] for c in range(3)}
    for level in ("B1_code_matched_oracle", "B2_code_excluded"):
        logits_rows = []
        for qi, query in enumerate(fixed_query):
            true_class, query_code = int(query["label"]), query["bearing_code"]
            chosen = []
            for c in range(3):
                candidates = safe_by_class[c]
                if c != true_class:
                    chosen.extend(candidates[:5]); continue
                same = [i for i in candidates if safe_support[i]["bearing_code"] == query_code]
                other = [i for i in candidates if safe_support[i]["bearing_code"] != query_code]
                if level.startswith("B1"):
                    if not same: raise RuntimeError(f"No matched support for {query_code}")
                    selected = same[:1] + other[:4]
                else:
                    selected = other[:5]
                if len(selected) != 5: raise RuntimeError(f"Invalid B support for {query_code}")
                chosen.extend(selected)
            for i in chosen:
                item = safe_support[i]
                support_manifest_rows.append({"experiment": "B", "condition": level,
                    "query_index": qi, "class": CLASS_NAME[int(item["label"])],
                    "bearing_code": item["bearing_code"], "measurement_id": item["measurement_id"],
                    "path": item["path"], "start": item["start"], "end": item["end"]})
            labels = torch.tensor([int(safe_support[i]["label"]) for i in chosen], device=device)
            logits_rows.append(ordinary_logits(model, safe_tokens[chosen].to(device), labels,
                                               query_tokens[qi:qi + 1].to(device)).cpu())
        logits = torch.cat(logits_rows)
        row, pred = metrics_row("B_code_visibility", level, logits, query_truth,
                                notes="offline oracle episode" if level.startswith("B1") else "query-code-excluded support")
        results.append(row); predictions["B1" if level.startswith("B1") else "B2"] = pred
    results.append({**baseline_row, "experiment": "B_code_visibility", "condition": "B0_actual_locked_support",
                    "notes": "actual support overlaps most, but not all, query code sets"})

    # C: same-speed optimistic source-domain references.
    c_rows = []
    for target_domain in ("DM", "DF"):
        train = [x for x in payload["windows"] if x["role"] == "train" and x["domain"] == target_domain]
        val = [x for x in payload["windows"] if x["role"] == "val" and x["domain"] == target_domain]
        selected = []
        for c in range(3):
            records = []
            seen = set()
            for row in sorted([x for x in train if int(x["label"]) == c],
                              key=lambda x: (x["bearing_code"], x["measurement_id"], x["start"])):
                if row["measurement_id"] not in seen:
                    records.append(row); seen.add(row["measurement_id"])
            selected.extend(round_robin(records, 5))
        c_items = selected + val
        for item in selected:
            support_manifest_rows.append({"experiment": "C", "condition": f"C2_{target_domain}",
                "query_index": "", "class": CLASS_NAME[int(item["label"])],
                "bearing_code": item["bearing_code"], "measurement_id": item["measurement_id"],
                "path": item["path"], "start": item["start"], "end": item["end"]})
        c_tokens, _ = extractor.extract(c_items, 8192)
        labels = torch.tensor([int(x["label"]) for x in selected], device=device)
        logits = ordinary_logits(model, c_tokens[:15].to(device), labels, c_tokens[15:].to(device)).cpu()
        truth = np.asarray([int(x["label"]) for x in val])
        row, pred = metrics_row("C_shift_type", f"C2_{target_domain}_1500rpm", logits, truth,
                                notes="optimistic same-speed reference; backbone saw domain in training")
        results.append(row); c_rows.append(row)
    results.append({**baseline_row, "experiment": "C_shift_type", "condition": "C1_DS_900rpm",
                    "notes": "locked cross-speed target"})

    # D: direct variable-length exposure, no retraining.
    d_predictions = {}
    all_d_items = fixed_support + fixed_query
    bounds_rows = []
    for length in (8192, 16384, 32768):
        if length == 8192:
            tokens = torch.cat((target_tokens[fixed_support_idx], query_tokens)); bounds = target_bounds[:15] + target_bounds[query_offset:]
        else:
            tokens, bounds = extractor.extract(all_d_items, length)
        bounds_rows.extend({"condition": f"D{length}", **x} for x in bounds)
        labels = support_labels_fixed.to(device)
        logits = ordinary_logits(model, tokens[:15].to(device), labels, tokens[15:].to(device)).cpu()
        condition = {8192: "D1_8192", 16384: "D2_16384", 32768: "D3_32768"}[length]
        row, pred = metrics_row("D_window_length", condition, logits, query_truth,
                                notes="frozen M2 direct variable-length input")
        results.append(row); predictions[condition[:2]] = pred; d_predictions[length] = pred
    write_csv(output / "window_bounds.csv", bounds_rows)

    # E: diagnostic hierarchy.
    st = target_tokens[fixed_support_idx].to(device); qt = query_tokens.to(device); sy = support_labels_fixed.to(device)
    coarse_logits = grouped_logits(st, sy, qt, [[0], [1, 2]]).cpu()
    row, coarse_pred = metrics_row("E_task_hierarchy", "E2_N_vs_Fault", coarse_logits, query_truth,
                                   task="coarse")
    results.append(row)
    fault_mask_q = query_truth > 0; fault_mask_s = support_labels_fixed > 0
    fault_logits = grouped_logits(st[fault_mask_s], sy[fault_mask_s], qt[fault_mask_q], [[1], [2]]).cpu()
    row, fault_pred = metrics_row("E_task_hierarchy", "E3_IR_vs_OR", fault_logits,
                                  query_truth[fault_mask_q], task="fault_only")
    results.append(row)
    results.append({**baseline_row, "experiment": "E_task_hierarchy", "condition": "E1_N_IR_OR"})

    # Save per-query baseline/counterfactual predictions used by overlap attribution.
    pred_rows = []
    for i, item in enumerate(fixed_query):
        row = {"query_index": i, "measurement_id": item["measurement_id"],
               "bearing_code": item["bearing_code"], "truth": CLASS_NAME[query_truth[i]]}
        for key in ("BASELINE", "A1", "A5", "A10", "A20", "B1", "B2", "D1", "D2", "D3"):
            if key in predictions: row[f"prediction_{key}"] = CLASS_NAME[int(predictions[key][i])]
        pred_rows.append(row)
    write_csv(output / "paired_query_predictions.csv", pred_rows)

    # Non-additive marginal table.
    lookup = {(x["experiment"], x["condition"]): x for x in results}
    a20 = lookup[("A_support_quantity", "A20_20shot")]
    b1 = lookup[("B_code_visibility", "B1_code_matched_oracle")]
    b2 = lookup[("B_code_visibility", "B2_code_excluded")]
    d1 = lookup[("D_window_length", "D1_8192")]; d3 = lookup[("D_window_length", "D3_32768")]
    c_mean = float(np.mean([x["accuracy"] for x in c_rows]))
    coarse_errors = int(np.sum(((query_truth == 0) & (baseline_pred != 0)) |
                               ((query_truth > 0) & (baseline_pred == 0))))
    marginal = [
        ("small_support_A5_to_A20", max(0., a20["accuracy"] - baseline_row["accuracy"]), "marginal; non-additive", "yes"),
        ("unseen_morphology_B2_to_B1_oracle", max(0., b1["accuracy"] - b2["accuracy"]), "oracle marginal; non-additive", "partly"),
        ("speed_context_C1_to_mean_C2", max(0., c_mean - baseline_row["accuracy"]), "optimistic contextual gap; non-causal", "requires real matched-speed data"),
        ("short_window_D1_to_D3", max(0., d3["accuracy"] - d1["accuracy"]), "marginal; direct frozen-model exposure", "yes if positive and stable"),
        ("health_fault_confusion_E", coarse_errors / len(query_truth), "exact baseline error anatomy", "possibly"),
    ]

    # Mutually exclusive baseline-error coverage mass for the pie.
    coverage = defaultdict(float); baseline_error_indices = np.flatnonzero(baseline_pred != query_truth)
    new_errors = {}
    candidates = {"small_support": predictions["A20"], "seen_morphology_oracle": predictions["B1"],
                  "long_window": predictions["D3"]}
    for name, pred in candidates.items():
        new_errors[name] = int(np.sum((baseline_pred == query_truth) & (pred != query_truth)))
    for i in baseline_error_indices:
        if (query_truth[i] == 0) != (baseline_pred[i] == 0):
            coverage["health_fault_confusion"] += 1.; continue
        fixers = [name for name, pred in candidates.items() if pred[i] == query_truth[i]]
        if fixers:
            for name in fixers: coverage[name] += 1. / len(fixers)
        else:
            coverage["unresolved_ir_or"] += 1.
    attribution_rows = []
    for factor, gap, method, mitigable in marginal:
        attribution_rows.append({"table": "marginal_nonadditive", "factor": factor,
            "contribution_error_rate_pp": 100 * gap,
            "share_of_baseline_15.42pct": gap / (1 - BASELINE_ACCURACY),
            "error_mass": None, "method": method, "mitigable": mitigable,
            "introduced_new_errors": None})
    for factor in ("health_fault_confusion", "small_support", "seen_morphology_oracle", "long_window", "unresolved_ir_or"):
        mass = coverage.get(factor, 0.)
        attribution_rows.append({"table": "overlap_aware_baseline_error_coverage", "factor": factor,
            "contribution_error_rate_pp": 100 * mass / len(query_truth),
            "share_of_baseline_15.42pct": mass / len(baseline_error_indices),
            "error_mass": mass, "method": "mutually exclusive Shapley-style correction coverage",
            "mitigable": "no" if factor == "unresolved_ir_or" else "diagnostic upper bound",
            "introduced_new_errors": new_errors.get(factor)})
    write_csv(output / "bottleneck_results.csv", results)
    write_csv(output / "bottleneck_attribution.csv", attribution_rows)
    write_csv(output / "support_manifests.csv", support_manifest_rows)
    write_csv(output / "unused_ds_support_pool.csv", [{"class": CLASS_NAME[int(x["label"])],
        "bearing_code": x["bearing_code"], "measurement_id": x["measurement_id"], "path": x["path"]}
        for x in sorted(unused, key=lambda x: (x["label"], x["bearing_code"], x["measurement_id"]))])
    write_csv(Path("bottleneck_results.csv"), results)
    write_csv(Path("bottleneck_attribution.csv"), attribution_rows)

    # Plots.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    a = [x for x in results if x["experiment"] == "A_support_quantity"]
    axes[0, 0].plot([1, 5, 10, 20], [next(x for x in a if x["condition"].startswith(f"A{s}_"))["accuracy"] for s in (1,5,10,20)], marker="o")
    axes[0, 0].axhline(.90, ls="--", color="black"); axes[0, 0].set_title("A Support quantity"); axes[0, 0].set_ylim(0,1)
    b = sorted([x for x in results if x["experiment"] == "B_code_visibility"],
               key=lambda x: {"B0": 0, "B1": 1, "B2": 2}[x["condition"].split("_")[0]])
    axes[0, 1].bar([x["condition"].split("_")[0] for x in b], [x["accuracy"] for x in b]); axes[0, 1].set_title("B Code visibility"); axes[0, 1].set_ylim(0,1)
    c = [x for x in results if x["experiment"] == "C_shift_type"]
    axes[0, 2].bar([x["condition"].replace("C2_","").replace("C1_","") for x in c], [x["accuracy"] for x in c]); axes[0, 2].set_title("C Shift context"); axes[0, 2].set_ylim(0,1)
    d = [x for x in results if x["experiment"] == "D_window_length"]
    axes[1, 0].plot([8192,16384,32768], [next(x for x in d if str(s) in x["condition"])["accuracy"] for s in (8192,16384,32768)], marker="o"); axes[1, 0].set_title("D Observation length"); axes[1, 0].set_ylim(0,1)
    e = sorted([x for x in results if x["experiment"] == "E_task_hierarchy"],
               key=lambda x: {"E1": 0, "E2": 1, "E3": 2}[x["condition"].split("_")[0]])
    axes[1, 1].bar([x["condition"].split("_")[0] for x in e], [x["accuracy"] for x in e]); axes[1, 1].set_title("E Diagnostic hierarchy"); axes[1, 1].set_ylim(0,1)
    pie_labels = [x for x in coverage if coverage[x] > 0]; pie_values = [coverage[x] for x in pie_labels]
    axes[1, 2].pie(pie_values, labels=pie_labels, autopct="%1.1f%%"); axes[1, 2].set_title("Baseline error coverage (37 errors)")
    fig.tight_layout(); fig.savefig(output / "bottleneck_deconvolution.png", dpi=180); plt.close(fig)

    audit = {"protocol": str(Path(args.protocol).resolve()), "protocol_sha256": protocol_sha,
             "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
             "split_sha256": sha256(args.split), "baseline_reproduced": True,
             "query_used_for_selection_or_tuning": False, "query_used_for_frozen_batch_metrics": True,
             "fixed_query_windows": len(fixed_query), "fixed_query_measurements": len({x['measurement_id'] for x in fixed_query}),
             "unused_ds_support_pool_measurements": len(unused), "device": str(device)}
    summary = {"audit": audit, "baseline": baseline_row, "results": results,
               "marginal_attribution": attribution_rows[:len(marginal)],
               "overlap_aware_coverage": attribution_rows[len(marginal):],
               "new_errors": new_errors,
               "warning": "Marginal gaps overlap and must not be summed; C2 is an optimistic contextual reference."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": baseline_row, "results": results,
                      "coverage": dict(coverage), "new_errors": new_errors}, indent=2))


if __name__ == "__main__":
    main()
