"""Run the preregistered C2OF architecture experiment.

The runner extracts only source measurements and the 15 labelled DS supports
until both the source gate and the one-time support-only gate pass.  DS query
rows are not retained and query waveforms are not loaded before that point.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from audit_x0b_phase_topology import representative_measurements
from bottleneck_deconvolution import load_model, sha256
from coordinate_carrier_order import CoordinateCarrierOrderDescriptor, CarrierOrderTokenAdapter
from data_loader_pu import load_vibration


SOURCE_DOMAINS = ("D0", "DM", "DF")
CLASS_NAMES = ("N", "IR", "OR")
CHECKPOINT_SHA256 = "cce4c3e22de6e1416a8df35bb5244e1c8cd2d5d3b12155a886da7dcf4553f098"
SPLIT_SHA256 = "0e7bf770e4de2037deff8471d2235c1b4493026e6b91b667def34d38881b3156"
M2_PARAMETER_COUNT = 1_059_088


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def safe_crop(item: dict, length: int = 16384) -> np.ndarray:
    signal = load_vibration(item["path"], item.get("signal_key", "signal"))
    centre = (int(item["start"]) + int(item["end"])) // 2
    start = max(0, min(centre - length // 2, len(signal) - length))
    wave = np.asarray(signal[start:start + length], dtype=np.float32).copy()
    if len(wave) != length:
        raise RuntimeError(f"Invalid C2OF crop: {item['measurement_id']}")
    wave = (wave - wave.mean()) / max(float(wave.std()), 1e-6)
    return wave


def make_descriptor(cfg: dict, device: torch.device) -> CoordinateCarrierOrderDescriptor:
    model = cfg["model"]; data = cfg["data"]
    return CoordinateCarrierOrderDescriptor(
        sample_rate=data["sample_rate"], window_length=data["window_length"],
        carrier_bands=model["carrier_bands"],
        dense_order_min=model["dense_order_min"], dense_order_max=model["dense_order_max"],
        dense_order_step=model["dense_order_step"], anchor_orders=model["anchor_orders"]).to(device)


def make_adapter(cfg: dict, device: torch.device) -> CarrierOrderTokenAdapter:
    model = cfg["model"]
    return CarrierOrderTokenAdapter(
        num_bands=len(model["carrier_bands"]), num_anchors=len(model["anchor_orders"]),
        hidden_dim=model["hidden_dim"], token_dim=128,
        max_beta=model["max_beta"]).to(device)


@torch.no_grad()
def extract_items(items: list[dict], m2, descriptor, device: torch.device,
                  batch_size: int = 8) -> dict:
    base_rows = []; descriptor_rows = defaultdict(list); metadata = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        wave = torch.from_numpy(np.stack([safe_crop(x) for x in batch])).to(device)
        rpm = torch.tensor([float(x["rpm"]) for x in batch], device=device)
        base_rows.append(m2(wave, rpm)["tokens"].cpu())
        values = descriptor(wave, rpm)
        for key, value in values.items(): descriptor_rows[key].append(value.cpu())
        metadata.extend({
            "domain": x["domain"], "role": x["role"],
            "measurement_id": x["measurement_id"], "bearing_code": x["bearing_code"],
            "label": int(x["label"]), "rpm": float(x["rpm"]), "path": x["path"]
        } for x in batch)
        print(f"C2OF features {min(start + len(batch), len(items))}/{len(items)}", flush=True)
    return {"base": torch.cat(base_rows), "descriptor": {
        key: torch.cat(value) for key, value in descriptor_rows.items()}, "metadata": metadata}


def cache_allowed_features(cfg: dict, m2, descriptor, device: torch.device,
                           split_path: Path, output: Path, force: bool = False) -> dict:
    cache = output / "source_and_support_cache.pt"
    if cache.exists() and not force:
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload["split_sha256"] != SPLIT_SHA256 or payload["query_used"]:
            raise RuntimeError("Invalid C2OF pre-gate feature cache")
        return payload
    items = representative_measurements(split_path)
    if any(x["domain"] == "DS" and x["role"] == "query" for x in items):
        raise RuntimeError("DS query crossed the C2OF pre-gate firewall")
    extracted = extract_items(items, m2, descriptor, device,
                              cfg["training"]["batch_size_extract"])
    payload = {**extracted, "split_sha256": sha256(split_path),
               "checkpoint_sha256": CHECKPOINT_SHA256, "query_used": False}
    torch.save(payload, cache)
    return payload


def to_device_descriptor(descriptor: dict[str, torch.Tensor], indices,
                         device: torch.device) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index].to(device) for key, value in descriptor.items()}


def ordinary_logits(support: torch.Tensor, support_y: torch.Tensor,
                    query: torch.Tensor, variance_floor: float = .05) -> torch.Tensor:
    means = []; variances = []
    for label in range(3):
        rows = support[support_y == label]
        means.append(rows.mean(0))
        variances.append(rows.var(0, unbiased=False) + variance_floor)
    mean = torch.stack(means); variance = torch.stack(variances)
    difference = query[:, None] - mean[None]
    return -(difference.square() / variance[None].clamp_min(variance_floor)).mean((2, 3))


def select_round_robin(metadata: list[dict], candidates: list[int], count: int,
                       rng: random.Random) -> list[int]:
    by_code = defaultdict(list)
    shuffled = list(candidates); rng.shuffle(shuffled)
    for index in shuffled: by_code[metadata[index]["bearing_code"]].append(index)
    codes = sorted(by_code); rng.shuffle(codes); selected = []
    while len(selected) < count:
        advanced = False
        for code in codes:
            if by_code[code]:
                selected.append(by_code[code].pop()); advanced = True
                if len(selected) == count: break
        if not advanced: break
    if len(selected) != count:
        raise RuntimeError(f"C2OF requires {count} independent support measurements")
    return selected


def sample_episode(metadata: list[dict], allowed_domains: tuple[str, ...],
                   rng: random.Random, shot: int, query_per_class: int) -> tuple[list[int], list[int]]:
    if len(allowed_domains) < 2:
        raise ValueError("C2OF episodes require two real source conditions")
    query_domain = rng.choice(allowed_domains)
    support_domain = rng.choice([x for x in allowed_domains if x != query_domain])
    support, query = [], []
    for label in range(3):
        query_codes = sorted({x["bearing_code"] for x in metadata
                              if x["domain"] == query_domain and x["role"] == "train"
                              and x["label"] == label})
        query_code = rng.choice(query_codes)
        query_candidates = [i for i, x in enumerate(metadata)
                            if x["domain"] == query_domain and x["role"] == "train"
                            and x["label"] == label and x["bearing_code"] == query_code]
        rng.shuffle(query_candidates); query.extend(query_candidates[:query_per_class])
        support_candidates = [i for i, x in enumerate(metadata)
                              if x["domain"] == support_domain and x["role"] == "train"
                              and x["label"] == label and x["bearing_code"] != query_code]
        support.extend(select_round_robin(metadata, support_candidates, shot, rng))
    return support, query


def coarse_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.stack((logits[:, 0], torch.logsumexp(logits[:, 1:3], dim=1)), dim=1)


def train_adapter(cache: dict, cfg: dict, allowed_domains: tuple[str, ...],
                  seed: int, device: torch.device) -> CarrierOrderTokenAdapter:
    torch.manual_seed(seed); np.random.seed(seed); rng = random.Random(seed)
    adapter = make_adapter(cfg, device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    metadata = cache["metadata"]; base_all = cache["base"].to(device)
    descriptor_all = {key: value.to(device) for key, value in cache["descriptor"].items()}
    weights = cfg["training"]["loss_weights"]
    for step in range(cfg["training"]["episodes"]):
        support, query = sample_episode(metadata, allowed_domains, rng,
                                        cfg["training"]["shot"],
                                        cfg["training"]["query_per_class"])
        indices = support + query; ns = len(support)
        selected = torch.as_tensor(indices, dtype=torch.long, device=device)
        base = base_all[selected]
        desc = {key: value[selected] for key, value in descriptor_all.items()}
        fused, details = adapter.fuse(base, desc)
        labels = torch.tensor([metadata[i]["label"] for i in indices], device=device)
        base_logits = ordinary_logits(base[:ns], labels[:ns], base[ns:])
        logits = ordinary_logits(fused[:ns], labels[:ns], fused[ns:])
        y = labels[ns:]
        proto = F.cross_entropy(logits, y)
        distill = F.kl_div(F.log_softmax(logits, -1), F.softmax(base_logits.detach(), -1),
                           reduction="batchmean")
        coarse = F.kl_div(F.log_softmax(coarse_logits(logits), -1),
                          F.softmax(coarse_logits(base_logits).detach(), -1), reduction="batchmean")
        base_margin = base_logits.gather(1, y[:, None]).squeeze(1) - base_logits.masked_fill(
            F.one_hot(y, 3).bool(), float("-inf")).max(1).values
        margin = logits.gather(1, y[:, None]).squeeze(1) - logits.masked_fill(
            F.one_hot(y, 3).bool(), float("-inf")).max(1).values
        margin_guard = F.relu(base_margin.detach() - margin).mean()
        residual = (details["beta"] * details["delta"]).square().mean()
        loss = (weights["proto"] * proto + weights["distill_kl"] * distill
                + weights["coarse"] * coarse + weights["margin_guard"] * margin_guard
                + weights["residual"] * residual)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"seed={seed} domains={allowed_domains} step={step+1} "
                  f"loss={float(loss.detach()):.4f} beta={float(details['beta'].detach()):.4f}", flush=True)
    return adapter.eval()


def metrics(y: np.ndarray, pred: np.ndarray, codes: list[str]) -> dict:
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    per_code = defaultdict(list)
    for code, truth, guess in zip(codes, y, pred): per_code[code].append(int(truth == guess))
    fault_codes = {code: float(np.mean(value)) for code, value in per_code.items()
                   if any(CLASS_NAMES[int(t)] != "N" for c, t in zip(codes, y) if c == code)}
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall": [float(x) for x in recall],
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "worst_fault_code_accuracy": min(fault_codes.values()) if fault_codes else 0.0,
            "per_code_accuracy": {key: float(np.mean(value)) for key, value in per_code.items()}}


@torch.no_grad()
def evaluate_held_domain(adapter: CarrierOrderTokenAdapter, cache: dict,
                         held_domain: str, device: torch.device, seed: int) -> tuple[list[dict], dict, dict]:
    metadata = cache["metadata"]; base_all = cache["base"].to(device)
    desc_all = {key: value.to(device) for key, value in cache["descriptor"].items()}
    support_domains = tuple(x for x in SOURCE_DOMAINS if x != held_domain)
    query_indices = [i for i, x in enumerate(metadata)
                     if x["domain"] == held_domain and x["role"] == "val"]
    rows = []
    for qi in query_indices:
        query = metadata[qi]; support = []
        for label in range(3):
            candidates = [i for i, x in enumerate(metadata)
                          if x["domain"] in support_domains and x["role"] == "train"
                          and x["label"] == label
                          and not (label == query["label"] and x["bearing_code"] == query["bearing_code"])]
            support.extend(select_round_robin(metadata, candidates, 5,
                                              random.Random(seed * 100000 + qi * 10 + label)))
        indices = support + [qi]; ns = len(support)
        index = torch.as_tensor(indices, device=device)
        base = base_all[index]; desc = {key: value[index] for key, value in desc_all.items()}
        fused, _ = adapter.fuse(base, desc)
        sy = torch.tensor([metadata[i]["label"] for i in support], device=device)
        base_pred = int(ordinary_logits(base[:ns], sy, base[ns:]).argmax(1))
        fused_pred = int(ordinary_logits(fused[:ns], sy, fused[ns:]).argmax(1))
        rows.append({"held_domain": held_domain, "measurement_id": query["measurement_id"],
                     "bearing_code": query["bearing_code"], "true": query["label"],
                     "base_pred": base_pred, "c2of_pred": fused_pred})
    y = np.asarray([x["true"] for x in rows]); codes = [x["bearing_code"] for x in rows]
    return rows, metrics(y, np.asarray([x["base_pred"] for x in rows]), codes), \
        metrics(y, np.asarray([x["c2of_pred"] for x in rows]), codes)


def aggregate_prediction_rows(rows: list[dict], key: str) -> dict:
    y = np.asarray([x["true"] for x in rows]); pred = np.asarray([x[key] for x in rows])
    return metrics(y, pred, [x["bearing_code"] for x in rows])


def compare_predictions(rows: list[dict]) -> dict:
    corrected = sum(x["base_pred"] != x["true"] and x["c2of_pred"] == x["true"] for x in rows)
    new = sum(x["base_pred"] == x["true"] and x["c2of_pred"] != x["true"] for x in rows)
    return {"corrected_errors": corrected, "new_errors": new}


def source_seed_gate(base: dict, candidate: dict, changes: dict, max_or_drop: float) -> dict:
    return {
        "macro_f1_noninferior": candidate["macro_f1"] >= base["macro_f1"] - 1e-12,
        "ir_noninferior": candidate["recall"][1] >= base["recall"][1] - 1e-12,
        "or_safe": candidate["recall"][2] >= base["recall"][2] - max_or_drop,
        "worst_fault_code_noninferior": candidate["worst_fault_code_accuracy"] >=
                                         base["worst_fault_code_accuracy"] - 1e-12,
        "error_exchange_safe": changes["new_errors"] <= changes["corrected_errors"],
    }


def source_phase(cache: dict, cfg: dict, device: torch.device, output: Path) -> tuple[dict, int]:
    seed_results = []
    for seed in cfg["training"]["seeds"]:
        all_rows = []; fold_results = []
        for held in SOURCE_DOMAINS:
            train_domains = tuple(x for x in SOURCE_DOMAINS if x != held)
            adapter = train_adapter(cache, cfg, train_domains, seed, device)
            rows, base, candidate = evaluate_held_domain(adapter, cache, held, device, seed)
            beta = float((adapter.max_beta * torch.sigmoid(adapter.gate_logit)).detach())
            all_rows.extend(rows); fold_results.append({"held_domain": held, "base": base,
                                                        "candidate": candidate, "beta": beta,
                                                        "parameter_count": adapter.parameter_count()})
        pooled_base = aggregate_prediction_rows(all_rows, "base_pred")
        pooled_candidate = aggregate_prediction_rows(all_rows, "c2of_pred")
        changes = compare_predictions(all_rows)
        gate = source_seed_gate(pooled_base, pooled_candidate, changes,
                                cfg["evaluation"]["max_or_drop_pp"] / 100.)
        gate["beta_valid"] = all(np.isfinite(x["beta"]) and 0 <= x["beta"] <= .20
                                 for x in fold_results)
        gate["parameter_budget"] = all(x["parameter_count"] < .05 * M2_PARAMETER_COUNT
                                       for x in fold_results)
        score = (pooled_candidate["macro_f1"]
                 - .5 * abs(pooled_candidate["recall"][1] - pooled_candidate["recall"][2])
                 + .2 * pooled_candidate["worst_fault_code_accuracy"])
        seed_results.append({"seed": seed, "base": pooled_base, "candidate": pooled_candidate,
                             "changes": changes, "gates": gate, "passed": all(gate.values()),
                             "selection_score": score, "folds": fold_results,
                             "predictions": all_rows})
        write_csv(output / f"source_predictions_seed{seed}.csv", all_rows)
    passing = [x for x in seed_results if x["passed"]]
    selected = max(passing or seed_results, key=lambda x: (x["selection_score"], -x["seed"]))
    source = {"seeds": [{k: v for k, v in row.items() if k != "predictions"}
                        for row in seed_results],
              "passing_seeds": len(passing), "selected_seed": selected["seed"],
              "source_unlocked": len(passing) >= 2,
              "selected_seed_passed": selected["passed"]}
    (output / "source_gate_before_target_support.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8")
    return source, int(selected["seed"])


def prototype_drift(tokens: torch.Tensor, labels: torch.Tensor) -> float:
    values = []
    for label in (1, 2):
        rows = tokens[labels == label]; full = rows.mean(0)
        for index in range(len(rows)):
            loo = rows[torch.arange(len(rows), device=rows.device) != index].mean(0)
            values.append((loo - full).norm())
    return float(torch.stack(values).mean())


@torch.no_grad()
def target_support_gate(adapter, cache: dict, device: torch.device) -> dict:
    indices = [i for i, x in enumerate(cache["metadata"])
               if x["domain"] == "DS" and x["role"] == "support"]
    if len(indices) != 15:
        raise RuntimeError("Expected exactly 15 DS supports")
    base = cache["base"][indices].to(device)
    desc = to_device_descriptor(cache["descriptor"], indices, device)
    fused, details = adapter.fuse(base, desc)
    labels = torch.tensor([cache["metadata"][i]["label"] for i in indices], device=device)
    rows = []
    for held in range(15):
        keep = torch.arange(15, device=device) != held
        base_pred = int(ordinary_logits(base[keep], labels[keep], base[held:held + 1]).argmax(1))
        fused_pred = int(ordinary_logits(fused[keep], labels[keep], fused[held:held + 1]).argmax(1))
        meta = cache["metadata"][indices[held]]
        rows.append({"measurement_id": meta["measurement_id"], "bearing_code": meta["bearing_code"],
                     "true": int(labels[held]), "base_pred": base_pred, "c2of_pred": fused_pred})
    y = np.asarray([x["true"] for x in rows]); codes = [x["bearing_code"] for x in rows]
    base_metric = metrics(y, np.asarray([x["base_pred"] for x in rows]), codes)
    candidate = metrics(y, np.asarray([x["c2of_pred"] for x in rows]), codes)
    fault = y != 0
    base_fault = float(np.mean(np.asarray([x["base_pred"] for x in rows])[fault] == y[fault]))
    candidate_fault = float(np.mean(np.asarray([x["c2of_pred"] for x in rows])[fault] == y[fault]))
    changes = compare_predictions(rows)
    drift_base = prototype_drift(base, labels); drift_candidate = prototype_drift(fused, labels)
    gates = {
        "macro_f1_noninferior": candidate["macro_f1"] >= base_metric["macro_f1"] - 1e-12,
        "fault_accuracy_noninferior": candidate_fault >= base_fault - 1e-12,
        "ir_noninferior": candidate["recall"][1] >= base_metric["recall"][1] - 1e-12,
        "or_noninferior": candidate["recall"][2] >= base_metric["recall"][2] - 1e-12,
        "prototype_drift_noninferior": drift_candidate <= drift_base + 1e-8,
        "at_least_one_safe_correction": changes["corrected_errors"] >= 1 and changes["new_errors"] == 0,
    }
    return {"base": base_metric, "candidate": candidate, "base_fault_accuracy": base_fault,
            "candidate_fault_accuracy": candidate_fault, "base_prototype_drift": drift_base,
            "candidate_prototype_drift": drift_candidate, "changes": changes,
            "beta": float(details["beta"]), "gates": gates, "unlocked": all(gates.values()),
            "predictions": rows}


def load_query_items(split_path: Path) -> list[dict]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    query = [x for x in payload["windows"] if x["domain"] == "DS" and x["role"] == "query"]
    if len(query) != 240:
        raise RuntimeError("Unexpected locked DS query contract")
    return query


@torch.no_grad()
def target_query_phase(adapter, allowed_cache: dict, cfg: dict, m2, descriptor,
                       device: torch.device, split_path: Path, output: Path) -> dict:
    query_items = load_query_items(split_path)
    query = extract_items(query_items, m2, descriptor, device,
                          cfg["training"]["batch_size_extract"])
    support_indices = [i for i, x in enumerate(allowed_cache["metadata"])
                       if x["domain"] == "DS" and x["role"] == "support"]
    base_support = allowed_cache["base"][support_indices].to(device)
    support_desc = to_device_descriptor(allowed_cache["descriptor"], support_indices, device)
    base_query = query["base"].to(device)
    query_desc = {key: value.to(device) for key, value in query["descriptor"].items()}
    fused_support, _ = adapter.fuse(base_support, support_desc)
    fused_query, details = adapter.fuse(base_query, query_desc)
    sy = torch.tensor([allowed_cache["metadata"][i]["label"] for i in support_indices], device=device)
    base_logits = ordinary_logits(base_support, sy, base_query)
    candidate_logits = ordinary_logits(fused_support, sy, fused_query)
    y = np.asarray([x["label"] for x in query["metadata"]]); codes = [x["bearing_code"] for x in query["metadata"]]
    base_pred = base_logits.argmax(1).cpu().numpy(); candidate_pred = candidate_logits.argmax(1).cpu().numpy()
    base_metric = metrics(y, base_pred, codes); candidate = metrics(y, candidate_pred, codes)
    rows = [{"measurement_id": x["measurement_id"], "bearing_code": x["bearing_code"],
             "true": int(t), "base_pred": int(b), "c2of_pred": int(c)}
            for x, t, b, c in zip(query["metadata"], y, base_pred, candidate_pred)]
    write_csv(output / "target_predictions.csv", rows)
    success = (candidate["accuracy"] >= cfg["evaluation"]["min_target_accuracy"]
               and candidate["macro_f1"] >= cfg["evaluation"]["min_target_macro_f1"]
               and candidate["recall"][0] >= cfg["evaluation"]["min_target_n_recall"]
               and candidate["recall"][1] >= cfg["evaluation"]["min_target_ir_recall"]
               and candidate["recall"][2] >= cfg["evaluation"]["min_target_or_recall"]
               and sum(np.asarray(base_pred) != y) - sum(np.asarray(candidate_pred) != y) >= 3)
    return {"base": base_metric, "candidate": candidate, "changes": compare_predictions(rows),
            "beta": float(details["beta"]), "success": bool(success), "query_count": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_c2of.yaml")
    parser.add_argument("--protocol", default="C2OF_PROTOCOL.md")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(cfg["output"]["directory"]); output.mkdir(parents=True, exist_ok=True)
    split_path = Path(cfg["data"]["split"]); checkpoint = Path(cfg["model"]["checkpoint"])
    if sha256(checkpoint) != CHECKPOINT_SHA256 or sha256(split_path) != SPLIT_SHA256:
        raise RuntimeError("Locked M2 or split SHA256 mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m2, _ = load_model(Path(cfg["model"]["config"]), checkpoint, device)
    for parameter in m2.parameters(): parameter.requires_grad_(False)
    descriptor = make_descriptor(cfg, device).eval()
    cache = cache_allowed_features(cfg, m2, descriptor, device, split_path, output, args.force_cache)
    source, selected_seed = source_phase(cache, cfg, device, output)
    summary = {"protocol_sha256": sha256(args.protocol), "config_sha256": sha256(args.config),
               "checkpoint_sha256": sha256(checkpoint), "split_sha256": sha256(split_path),
               "target_query_extracted_before_gates": False, "source": source}
    if not source["source_unlocked"] or not source["selected_seed_passed"]:
        summary.update({"target_support_evaluated": False, "target_unlocked": False,
                        "success": False, "decision": "STOP_SOURCE_GATE"})
    else:
        adapter = train_adapter(cache, cfg, SOURCE_DOMAINS, selected_seed, device)
        torch.save({"adapter": adapter.state_dict(), "selected_seed": selected_seed,
                    "protocol_sha256": summary["protocol_sha256"]}, output / "selected_adapter.pt")
        trainable = adapter.parameter_count(); total_m2 = sum(p.numel() for p in m2.parameters())
        parameter_gate = trainable < .05 * total_m2
        support = target_support_gate(adapter, cache, device)
        support["parameter_count"] = trainable; support["m2_parameter_fraction"] = trainable / total_m2
        support["gates"]["parameter_budget"] = parameter_gate
        support["unlocked"] = support["unlocked"] and parameter_gate
        write_csv(output / "target_support_loo_predictions.csv", support.pop("predictions"))
        (output / "target_support_gate_before_query.json").write_text(
            json.dumps(support, indent=2), encoding="utf-8")
        summary["target_support_evaluated"] = True; summary["target_support"] = support
        if not support["unlocked"]:
            summary.update({"target_unlocked": False, "success": False,
                            "decision": "STOP_TARGET_SUPPORT_GATE"})
        else:
            target = target_query_phase(adapter, cache, cfg, m2, descriptor, device,
                                        split_path, output)
            summary.update({"target_unlocked": True, "target": target,
                            "success": target["success"],
                            "decision": "GO" if target["success"] else "NO_GO_TARGET"})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
