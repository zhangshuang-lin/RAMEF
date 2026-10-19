"""Query-free RAMEF design-choice sensitivity audit.

This script addresses reviewer-facing questions about the source/support
weight, linear-head capacity and regularisation without loading any DS query
identifier, waveform, feature or label. Hyperparameters are ranked on real PU
source pseudo-target episodes with bearing-code-isolated queries. The locked 15
DS support measurements are used only in leave-one-out confirmation.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from bottleneck_deconvolution import sha256
from prepare_splits import scan_records
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_tser_hb_support_p0 import support_only


WEIGHTS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
CS = (0.01, 0.1, 1.0)
SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metrics(y: np.ndarray, pred: np.ndarray, codes: list[str]) -> dict:
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    code_accuracy = {}
    for code in sorted(set(codes)):
        idx = np.asarray([i for i, value in enumerate(codes) if value == code])
        code_accuracy[code] = float(np.mean(pred[idx] == y[idx]))
    fault_codes = [value for code, value in code_accuracy.items()
                   if int(y[codes.index(code)]) in (1, 2)]
    return {
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro",
                                    zero_division=0)),
        "recall_N": float(recalls[0]),
        "recall_IR": float(recalls[1]),
        "recall_OR": float(recalls[2]),
        "recall_gap": float(abs(recalls[1] - recalls[2])),
        "worst_fault_code_accuracy": float(min(fault_codes)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def fit_lr(train_x: np.ndarray, train_y: np.ndarray, train_w: np.ndarray,
           test_x: np.ndarray, c_value: float, seed: int) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    head = LogisticRegression(C=c_value, max_iter=3000, solver="lbfgs",
                              random_state=seed)
    head.fit(scaler.transform(train_x), train_y, sample_weight=train_w)
    return head.predict(scaler.transform(test_x))


class SmallMLP(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, 64), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(train_x: np.ndarray, train_y: np.ndarray, train_w: np.ndarray,
            test_x: np.ndarray, seed: int, device: torch.device) -> np.ndarray:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    scaler = StandardScaler().fit(train_x)
    x = torch.tensor(scaler.transform(train_x), dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.long, device=device)
    w = torch.tensor(train_w, dtype=torch.float32, device=device)
    test = torch.tensor(scaler.transform(test_x), dtype=torch.float32, device=device)
    model = SmallMLP(train_x.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    model.train()
    for _ in range(80):
        optimiser.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(model(x), y, reduction="none") * w).sum() / w.sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
    model.eval()
    with torch.no_grad(): pred = model(test).argmax(1).cpu().numpy()
    return pred


def build_pseudo_episodes(rows: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    """Build real-load pseudo-target episodes with code-isolated queries."""
    episodes = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for domain in SOURCE_DOMAINS:
            support_idx, query_idx, query_codes = [], [], set()
            for label in (0, 1, 2):
                domain_codes = sorted({row["bearing_code"] for row in rows
                                       if row["domain"] == domain and int(row["label"]) == label})
                chosen = sorted(rng.choice(domain_codes, 3, replace=False).tolist())
                held_codes = sorted(set(domain_codes) - set(chosen)); query_codes.update(held_codes)
                allocation = (2, 2, 1)
                for code, count in zip(chosen, allocation):
                    candidates = np.asarray([i for i, row in enumerate(rows)
                                             if row["domain"] == domain
                                             and int(row["label"]) == label
                                             and row["bearing_code"] == code])
                    support_idx.extend(rng.choice(candidates, count, replace=False).tolist())
                query_idx.extend(i for i, row in enumerate(rows)
                                 if row["domain"] == domain
                                 and int(row["label"]) == label
                                 and row["bearing_code"] in held_codes)
            source_idx = [i for i, row in enumerate(rows)
                          if row["domain"] != domain and row["bearing_code"] not in query_codes]
            if len(support_idx) != 15 or not query_idx:
                raise RuntimeError("Invalid pseudo-target episode")
            episodes.append({"seed": seed, "domain": domain,
                             "source": np.asarray(source_idx, dtype=np.int64),
                             "support": np.asarray(support_idx, dtype=np.int64),
                             "query": np.asarray(query_idx, dtype=np.int64),
                             "query_codes": sorted(query_codes)})
    return episodes


def evaluate_lr_grid(rows: list[dict], x: np.ndarray, episodes: list[dict]) -> list[dict]:
    y = np.asarray([int(row["label"]) for row in rows]); out = []
    for c_value in CS:
        for support_weight in WEIGHTS:
            episode_values = []
            for episode in episodes:
                source, support, query = episode["source"], episode["support"], episode["query"]
                train = np.concatenate([source, support])
                weights = np.concatenate([np.ones(len(source)),
                                          np.full(len(support), support_weight)])
                pred = fit_lr(x[train], y[train], weights, x[query], c_value,
                              episode["seed"])
                value = metrics(y[query], pred, [rows[i]["bearing_code"] for i in query])
                episode_values.append(value)
            row = {"head": "linear", "C": c_value, "support_weight": support_weight}
            for key in ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR",
                        "recall_gap", "worst_fault_code_accuracy"):
                values = np.asarray([value[key] for value in episode_values])
                row[key + "_mean"] = float(values.mean())
                row[key + "_std"] = float(values.std(ddof=1))
            row["selection_score"] = (row["macro_f1_mean"]
                                      - 0.25 * row["recall_gap_mean"]
                                      + 0.10 * row["worst_fault_code_accuracy_mean"])
            out.append(row)
    return out


def support_loo(rows: list[dict], x: np.ndarray, source_count: int,
                head: str, c_value: float, support_weight: float,
                device: torch.device) -> dict:
    source_y = np.asarray([int(row["label"]) for row in rows[:source_count]])
    support_rows = rows[source_count:]; support_y = np.asarray([int(row["label"])
                                                                for row in support_rows])
    pred = []
    for held in range(len(support_rows)):
        keep = np.asarray([i for i in range(len(support_rows)) if i != held])
        train_x = np.vstack([x[:source_count], x[source_count:][keep]])
        train_y = np.concatenate([source_y, support_y[keep]])
        weights = np.concatenate([np.ones(source_count), np.full(len(keep), support_weight)])
        seed = 20261100 + held
        if head == "linear":
            local = fit_lr(train_x, train_y, weights, x[source_count + held:source_count + held + 1],
                           c_value, seed)
        else:
            local = fit_mlp(train_x, train_y, weights, x[source_count + held:source_count + held + 1],
                            seed, device)
        pred.append(int(local[0]))
    return metrics(support_y, np.asarray(pred), [row["bearing_code"] for row in support_rows])


def compare_heads(rows: list[dict], x: np.ndarray, episodes: list[dict],
                  c_value: float, support_weight: float,
                  device: torch.device) -> list[dict]:
    y = np.asarray([int(row["label"]) for row in rows]); output = []
    for head in ("linear", "mlp64"):
        values = []
        for episode in episodes:
            source, support, query = episode["source"], episode["support"], episode["query"]
            train = np.concatenate([source, support])
            weights = np.concatenate([np.ones(len(source)), np.full(len(support), support_weight)])
            if head == "linear":
                pred = fit_lr(x[train], y[train], weights, x[query], c_value, episode["seed"])
            else:
                pred = fit_mlp(x[train], y[train], weights, x[query], episode["seed"], device)
            values.append(metrics(y[query], pred, [rows[i]["bearing_code"] for i in query]))
        row = {"head": head, "C": c_value, "support_weight": support_weight}
        for key in ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR",
                    "recall_gap", "worst_fault_code_accuracy"):
            array = np.asarray([value[key] for value in values])
            row[key + "_mean"] = float(array.mean()); row[key + "_std"] = float(array.std(ddof=1))
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path("outputs/staged/RACP_OR_INFORMATION_P0/authorised_source_support_features.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/staged/RAMEF_DESIGN_SENSITIVITY"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = sorted([row for row in scan_records(Path("D:/Paderborn"))
                     if row["domain"] in SOURCE_DOMAINS], key=lambda row: row["measurement_id"])
    support, query_count = support_only(Path("split_independent_measurement.json"))
    rows = source + support
    cache = np.load(args.cache, allow_pickle=False)
    if cache["ids"].tolist() != [row["measurement_id"] for row in rows]:
        raise RuntimeError("Authorised cache mismatch")
    x = np.concatenate([cache["blocks"].astype(np.float32).mean(1),
                        cache["physical"].astype(np.float32)], axis=1)
    episodes = build_pseudo_episodes(source, SEEDS)
    grid = evaluate_lr_grid(source, x[:len(source)], episodes)
    selected = sorted(grid, key=lambda row: (-row["selection_score"],
                                             row["support_weight"], row["C"]))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support_surface = []
    for candidate in grid:
        value = support_loo(rows, x, len(source), "linear", candidate["C"],
                            candidate["support_weight"], device)
        support_surface.append({"C": candidate["C"],
                                "support_weight": candidate["support_weight"], **value})
    heads = compare_heads(source, x[:len(source)], episodes, selected["C"],
                          selected["support_weight"], device)
    registered_heads = compare_heads(source, x[:len(source)], episodes, 0.1, 10.0, device)
    registered_confirmations = {
        head: support_loo(rows, x, len(source), head, 0.1, 10.0, device)
        for head in ("linear", "mlp64")
    }
    confirmations = {
        head: support_loo(rows, x, len(source), head, selected["C"],
                          selected["support_weight"], device)
        for head in ("linear", "mlp64")
    }
    result = {
        "experiment": "RAMEF_DESIGN_SENSITIVITY",
        "status": "COMPLETE_QUERY_FREE",
        "selection": selected,
        "support_loo_linear_surface": support_surface,
        "head_comparison": heads,
        "registered_C0p1_weight10_head_comparison": registered_heads,
        "registered_C0p1_weight10_support_loo": registered_confirmations,
        "support_loo": confirmations,
        "episodes": len(episodes),
        "firewall": {"ds_query_rows_in_split": query_count,
                     "ds_query_measurements_loaded": 0,
                     "ds_query_features_loaded": False,
                     "ds_query_labels_used": False},
        "cache_sha256": sha256(args.cache),
        "split_sha256": sha256(Path("split_independent_measurement.json")),
        "mlp_definition": {"hidden": 64, "dropout": 0.2, "epochs": 80,
                           "learning_rate": 1e-3, "weight_decay": 1e-3},
    }
    write_csv(args.output / "linear_grid.csv", grid)
    write_csv(args.output / "support_loo_linear_surface.csv", support_surface)
    write_csv(args.output / "head_comparison.csv", heads)
    write_csv(args.output / "registered_head_comparison.csv", registered_heads)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
