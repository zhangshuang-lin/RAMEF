"""Execute the preregistered X0-B P0 revolution-topology audit.

DS query waveforms are never loaded.  The only target observations are the 15
labelled supports permitted by X0B_P0_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from angular_cycle_topology import extract_representations
from data_loader_pu import load_vibration


SOURCE_DOMAINS = ("D0", "DM", "DF")
CLASS_NAME = {0: "N", 1: "IR", 2: "OR"}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def representative_measurements(split_path: Path) -> list[dict]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    eligible = [x for x in payload["windows"]
                if (x["domain"] in SOURCE_DOMAINS and x["role"] in {"train", "val"})
                or (x["domain"] == "DS" and x["role"] == "support")]
    # Explicit firewall: target query rows are not retained past JSON decoding.
    assert not any(x["domain"] == "DS" and x["role"] == "query" for x in eligible)
    grouped = defaultdict(list)
    for row in eligible:
        grouped[(row["domain"], row["role"], row["measurement_id"])].append(row)
    representatives = []
    for _, rows in sorted(grouped.items()):
        signal_length = int(rows[0].get("signal_length", 256001))
        target = signal_length / 2
        representatives.append(min(rows, key=lambda x: abs((int(x["start"]) + int(x["end"])) / 2 - target)))
    return representatives


def real_long_crop(item: dict, cache: dict[str, np.ndarray], length: int = 32768) -> tuple[np.ndarray, dict]:
    path = str(item["path"])
    if path not in cache:
        cache[path] = load_vibration(path, item.get("signal_key", "signal"))
    signal = cache[path]
    registered_centre = (int(item["start"]) + int(item["end"])) // 2
    requested = registered_centre - length // 2
    start = max(0, min(requested, len(signal) - length)); end = start + length
    if start < 0 or end > len(signal) or end - start != length:
        raise RuntimeError(f"Invalid long crop for {item['measurement_id']}")
    return np.asarray(signal[start:end], dtype=np.float32).copy(), {
        "measurement_id": item["measurement_id"], "registered_centre": registered_centre,
        "actual_start": start, "actual_end": end, "centre_shift": start - requested,
    }


def extract_all(items: list[dict], output: Path, force: bool = False):
    cache_path = output / "feature_cache.npz"
    meta_path = output / "feature_metadata.csv"
    if cache_path.exists() and meta_path.exists() and not force:
        data = np.load(cache_path)
        with meta_path.open(encoding="utf-8-sig") as stream:
            metadata = list(csv.DictReader(stream))
        return metadata, data["order"], data["topology"]
    signal_cache, order_rows, topology_rows, metadata = {}, [], [], []
    for index, item in enumerate(items):
        wave, bounds = real_long_crop(item, signal_cache)
        order, topology = extract_representations(wave, float(item["rpm"]))
        order_rows.append(order); topology_rows.append(topology)
        metadata.append({
            "index": index, "domain": item["domain"], "role": item["role"],
            "measurement_id": item["measurement_id"], "bearing_code": item["bearing_code"],
            "label": int(item["label"]), "rpm": float(item["rpm"]), **bounds,
        })
        if (index + 1) % 50 == 0:
            print(f"features {index + 1}/{len(items)}", flush=True)
    order_array = np.stack(order_rows); topology_array = np.stack(topology_rows)
    np.savez_compressed(cache_path, order=order_array, topology=topology_array)
    write_csv(meta_path, metadata)
    return metadata, order_array, topology_array


def metric(y: np.ndarray, pred: np.ndarray, labels=(1, 2)) -> dict:
    recalls = recall_score(y, pred, labels=list(labels), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0)),
        "recalls": {str(c): float(v) for c, v in zip(labels, recalls)},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(labels)).tolist(),
    }


def representation_map(order: np.ndarray, topology: np.ndarray) -> dict[str, np.ndarray]:
    return {"ORDER_AVG": order, "RPT": topology, "ORDER_PLUS_RPT": np.concatenate((order, topology), axis=1)}


def source_folds(metadata: list[dict], representations: dict[str, np.ndarray]):
    rows, predictions = [], []
    for held in SOURCE_DOMAINS:
        test_idx = np.asarray([i for i, x in enumerate(metadata)
                               if x["domain"] == held and x["role"] == "val" and int(x["label"]) in (1, 2)])
        query_codes = {c: {metadata[i]["bearing_code"] for i in test_idx if int(metadata[i]["label"]) == c}
                       for c in (1, 2)}
        train_idx = np.asarray([i for i, x in enumerate(metadata)
                                if x["domain"] in SOURCE_DOMAINS and x["domain"] != held
                                and x["role"] == "train" and int(x["label"]) in (1, 2)
                                and x["bearing_code"] not in query_codes[int(x["label"])]] )
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"Empty code-isolated fold {held}")
        y_train = np.asarray([int(metadata[i]["label"]) for i in train_idx])
        y_test = np.asarray([int(metadata[i]["label"]) for i in test_idx])
        for name, values in representations.items():
            scaler = StandardScaler().fit(values[train_idx])
            classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                            random_state=20260716).fit(scaler.transform(values[train_idx]), y_train)
            pred = classifier.predict(scaler.transform(values[test_idx]))
            result = metric(y_test, pred)
            rows.append({"representation": name, "held_domain": held, **result,
                         "n_train_measurements": len(train_idx), "n_query_measurements": len(test_idx),
                         "excluded_codes": json.dumps({str(k): sorted(v) for k, v in query_codes.items()})})
            for idx, truth, guess in zip(test_idx, y_test, pred):
                predictions.append({"representation": name, "held_domain": held,
                                    "measurement_id": metadata[idx]["measurement_id"],
                                    "bearing_code": metadata[idx]["bearing_code"],
                                    "true": int(truth), "pred": int(guess), "correct": int(truth == guess)})
    return rows, predictions


def pooled_metrics(predictions: list[dict]) -> list[dict]:
    output = []
    for name in ("ORDER_AVG", "RPT", "ORDER_PLUS_RPT"):
        selected = [x for x in predictions if x["representation"] == name]
        y = np.asarray([x["true"] for x in selected]); p = np.asarray([x["pred"] for x in selected])
        output.append({"representation": name, **metric(y, p), "n_measurements": len(y)})
    return output


def paired_bootstrap(predictions: list[dict], repeats: int = 5000) -> dict:
    base = {(x["measurement_id"], x["held_domain"]): x for x in predictions if x["representation"] == "ORDER_AVG"}
    cand = {(x["measurement_id"], x["held_domain"]): x for x in predictions if x["representation"] == "ORDER_PLUS_RPT"}
    keys = sorted(base)
    strata = defaultdict(list)
    for key in keys:
        strata[(base[key]["held_domain"], base[key]["true"])].append(key)
    rng = np.random.default_rng(20260716); changes = []
    for _ in range(repeats):
        sample = []
        for values in strata.values():
            chosen = rng.integers(0, len(values), size=len(values))
            sample.extend(values[int(i)] for i in chosen)
        y = np.asarray([base[k]["true"] for k in sample])
        pb = np.asarray([base[k]["pred"] for k in sample]); pc = np.asarray([cand[k]["pred"] for k in sample])
        changes.append(f1_score(y, pc, labels=[1, 2], average="macro", zero_division=0)
                       - f1_score(y, pb, labels=[1, 2], average="macro", zero_division=0))
    changes = np.asarray(changes)
    return {"mean": float(changes.mean()), "ci_low": float(np.quantile(changes, .025)),
            "ci_high": float(np.quantile(changes, .975)), "repeats": repeats}


def code_comparison(predictions: list[dict]) -> list[dict]:
    output = []
    codes = sorted({x["bearing_code"] for x in predictions})
    for code in codes:
        row = {"bearing_code": code}
        for name in ("ORDER_AVG", "ORDER_PLUS_RPT"):
            values = [x["correct"] for x in predictions if x["representation"] == name and x["bearing_code"] == code]
            row[name.lower() + "_accuracy"] = float(np.mean(values)) if values else None
            row[name.lower() + "_n"] = len(values)
        row["gain"] = row["order_plus_rpt_accuracy"] - row["order_avg_accuracy"]
        output.append(row)
    return output


def support_loo(metadata: list[dict], representations: dict[str, np.ndarray]) -> list[dict]:
    source_train = np.asarray([i for i, x in enumerate(metadata)
                               if x["domain"] in SOURCE_DOMAINS and x["role"] == "train"])
    support = np.asarray([i for i, x in enumerate(metadata) if x["domain"] == "DS" and x["role"] == "support"])
    rows = []
    for name, values in representations.items():
        scaler = StandardScaler().fit(values[source_train])
        z_source = scaler.transform(values[source_train])
        n_components = min(12, z_source.shape[0] - 1, z_source.shape[1])
        pca = PCA(n_components=n_components, whiten=True, random_state=20260716).fit(z_source)
        z = pca.transform(scaler.transform(values[support]))
        y = np.asarray([int(metadata[i]["label"]) for i in support])
        for task, labels in (("three_class", (0, 1, 2)), ("fault_only", (1, 2))):
            used = np.where(np.isin(y, labels))[0]; pred, truth, identities = [], [], []
            for local in used:
                means = []
                for c in labels:
                    mask = (y == c); mask[local] = False
                    means.append(z[mask].mean(0))
                distance = [np.mean((z[local] - centre) ** 2) for centre in means]
                pred.append(labels[int(np.argmin(distance))]); truth.append(y[local]); identities.append(local)
            result = metric(np.asarray(truth), np.asarray(pred), labels=labels)
            rows.append({"representation": name, "task": task, **result,
                         "n_correct": int(np.sum(np.asarray(truth) == np.asarray(pred))),
                         "n_samples": len(truth),
                         "predictions": json.dumps([{"measurement_id": metadata[support[i]]["measurement_id"],
                                                      "bearing_code": metadata[support[i]]["bearing_code"],
                                                      "true": int(t), "pred": int(p)}
                                                     for i, t, p in zip(identities, truth, pred)])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="split_independent_measurement.json")
    parser.add_argument("--protocol", default="X0B_P0_PROTOCOL.md")
    parser.add_argument("--output", default="outputs/staged/X0B_P0")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    items = representative_measurements(Path(args.split))
    metadata, order, topology = extract_all(items, output, args.force_features)
    reps = representation_map(order, topology)
    folds, predictions = source_folds(metadata, reps)
    pooled = pooled_metrics(predictions); bootstrap = paired_bootstrap(predictions)
    codes = code_comparison(predictions); loo = support_loo(metadata, reps)
    write_csv(output / "source_fold_metrics.csv", folds)
    write_csv(output / "source_predictions.csv", predictions)
    write_csv(output / "source_pooled_metrics.csv", pooled)
    write_csv(output / "bearing_code_comparison.csv", codes)
    write_csv(output / "ds_support_loo.csv", loo)

    pooled_by = {x["representation"]: x for x in pooled}
    gain = pooled_by["ORDER_PLUS_RPT"]["macro_f1"] - pooled_by["ORDER_AVG"]["macro_f1"]
    positive_domains = sum(
        next(x for x in folds if x["representation"] == "ORDER_PLUS_RPT" and x["held_domain"] == d)["macro_f1"]
        > next(x for x in folds if x["representation"] == "ORDER_AVG" and x["held_domain"] == d)["macro_f1"]
        for d in SOURCE_DOMAINS)
    code_by = {x["bearing_code"]: x for x in codes}
    or_gain = any(code_by.get(c, {}).get("gain", -1) > 0 for c in ("KA22", "KA30"))
    ir_gain = any(code_by.get(c, {}).get("gain", -1) > 0 for c in ("KI04", "KI14"))
    family_safe = all(np.mean([code_by[c]["gain"] for c in group if c in code_by]) >= -.10
                      for group in (("KA22", "KA30"), ("KI04", "KI14")))
    candidate_loo = {(x["representation"], x["task"]): x for x in loo}
    three = candidate_loo[("ORDER_PLUS_RPT", "three_class")]
    fault = candidate_loo[("ORDER_PLUS_RPT", "fault_only")]
    gate = {
        "source_macro_f1_gain_ge_5pp": gain >= .05,
        "positive_in_two_domains": positive_domains >= 2,
        "difficult_or_and_ir_gain": or_gain and ir_gain,
        "difficult_families_safe": family_safe,
        "ds_three_class_loo": three["n_correct"] >= 10 and three["recalls"]["1"] >= .6 and three["recalls"]["2"] >= .6,
        "ds_fault_loo": fault["n_correct"] >= 7,
    }
    decision = "PASS_TO_RPT_NET" if all(gate.values()) else "STOP_RPT_NET"
    summary = {"decision": decision, "gates": gate, "source_macro_f1_gain": gain,
               "positive_domains": positive_domains, "bootstrap": bootstrap,
               "pooled": pooled, "ds_support_loo": loo,
               "protocol_sha256": sha256(args.protocol), "split_sha256": sha256(args.split),
               "ds_query_features_accessed": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    names = ["ORDER_AVG", "RPT", "ORDER_PLUS_RPT"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(names, [100 * pooled_by[n]["macro_f1"] for n in names], color=["#78849a", "#da8a49", "#3c8d7b"])
    axes[0].set_ylabel("Source fault-only Macro-F1 (%)"); axes[0].tick_params(axis="x", rotation=18)
    loo_by = {(x["representation"], x["task"]): x for x in loo}
    axes[1].bar(names, [100 * loo_by[(n, "three_class")]["accuracy"] for n in names], color=["#78849a", "#da8a49", "#3c8d7b"])
    axes[1].axhline(100 * 10 / 15, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("DS support three-class LOO Accuracy (%)"); axes[1].tick_params(axis="x", rotation=18)
    fig.suptitle(f"X0-B P0: {decision}"); fig.tight_layout()
    fig.savefig(output / "x0b_p0_summary.png", dpi=220); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
