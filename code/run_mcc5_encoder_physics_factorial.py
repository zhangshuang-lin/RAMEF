"""Frozen MCC5 encoder × physics 2×2 factorial.

Run `prepare` first. `score` is a separately locked, once-only *post-target*
event on an already accessed query cohort. It never fits on query data.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from capacity_matched_baselines import WidthMatchedWDCNN
from mcc5_confirmatory_data import sha256
from run_mcc5_capacity_dimension_p0 import infer_blocks, metric, validate_development_roles, write_csv
from run_mcc5_posthoc_fair_comparison_once import validate_query, vector_metrics, read_csv
from run_mcc5_ramef_confirmatory import SEEDS, balanced_source_weights, load_model

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/staged/MCC5_RAMEF_CONFIRMATORY"
P0 = ROOT / "outputs/staged/MCC5_CAPACITY_DIMENSION_P0"
FAIR = ROOT / "outputs/staged/MCC5_POSTHOC_FAIR_COMPARISON"
OUT = ROOT / "outputs/staged/MCC5_ENCODER_PHYSICS_FACTORIAL"
PROTOCOL = ROOT / "MCC5_ENCODER_PHYSICS_FACTORIAL_PROTOCOL.md"
VIEWS = ("u2_carrier", "u2_physical", "wdcnn_carrier", "wdcnn_physical")


def feature_views(u2: np.ndarray, wdcnn: np.ndarray, physical: np.ndarray):
    """The same physical coordinates enter both frozen carriers unchanged."""
    if u2.shape[1] != 128 or wdcnn.shape[1] != 128 or physical.shape[1] != 63:
        raise RuntimeError("2×2 feature dimension changed")
    return {"u2_carrier": u2, "u2_physical": np.hstack([u2, physical]),
            "wdcnn_carrier": wdcnn, "wdcnn_physical": np.hstack([wdcnn, physical])}


def existing_reference_hashes():
    paths = [BASE / "run/target_evaluation.lock", BASE / "run/result.json",
             BASE / "run/predictions.csv", FAIR / "target_evaluation.lock",
             FAIR / "result.json", FAIR / "predictions.csv"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def prepare():
    if OUT.exists():
        raise RuntimeError("Factorial archive already exists; no overwrite")
    OUT.mkdir(parents=True)
    (OUT / "protocol_frozen.md").write_bytes(PROTOCOL.read_bytes())
    selected = json.loads((BASE / "run/selection_complete.json").read_text())
    devpath = BASE / "cache/development.npz"
    if sha256(devpath) != selected["development_cache_sha256"]:
        raise RuntimeError("Development cache hash changed")
    with np.load(devpath, allow_pickle=False) as archive:
        ids, roles, y, phys = (archive[k] for k in ("ids", "roles", "labels", "physical"))
        validate_development_roles(roles, ids)
        if bool(archive["query_cache"]):
            raise RuntimeError("Query data entered development phase")
    source = np.flatnonzero(roles == "source")
    support = np.flatnonzero(roles == "target_support")
    fit = np.concatenate([source, support])
    weights = np.r_[balanced_source_weights(y[source]), np.full(len(support), 10.)]
    rows, checks, model_hashes = [], [], {}
    for seed in SEEDS:
        feature_path = P0 / f"development_embeddings_seed{seed}.npz"
        with np.load(feature_path, allow_pickle=False) as features:
            if not np.array_equal(features["ids"], ids):
                raise RuntimeError("Frozen feature order changed")
            u2 = features["U2_Hz_1024"]
            unique = u2[:, :128]
            if not np.array_equal(u2.reshape(99, 8, 128), np.repeat(unique[:, None, :], 8, axis=1)):
                raise RuntimeError("U2 carrier token repetition changed")
            x = feature_views(unique, features["WDCNN_widthMatched_128"], phys)
        for name, value in x.items():
            scaler = StandardScaler().fit(value[fit])
            head = LogisticRegression(C=.1, solver="lbfgs", max_iter=4000, random_state=seed)
            head.fit(scaler.transform(value[fit]), y[fit], sample_weight=weights)
            path = OUT / f"head_{seed}_{name}.joblib"
            joblib.dump({"seed": seed, "view": name, "scaler": scaler, "head": head,
                         "query_execution_enabled": False}, path)
            rows.append({"seed": seed, "view": name, "dimension": value.shape[1],
                         "C": .1, "file": path.name, "sha256": sha256(path),
                         "development_feature_sha256": sha256(feature_path)})
            # In-sample diagnostics are not model-selection scores.
            p = head.predict(scaler.transform(value[fit]))
            checks.append({"seed": seed, "view": name, "fit_accuracy": float(np.mean(p == y[fit])),
                           "fit_runs": len(fit)})
        for path in (BASE / f"run/checkpoints/u2hz_seed{seed}.pt",
                     P0 / f"checkpoints/wdcnn_widthMatched_seed{seed}.pt"):
            model_hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_csv(OUT / "development_checks.csv", checks)
    registry = {"status": "PREPARED_QUERY_UNREAD", "prepared_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                "development_cache_sha256": sha256(devpath), "head_rows": rows,
                "model_hashes": model_hashes, "reference_hashes": existing_reference_hashes(),
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "scikit_learn": sklearn.__version__},
                "target_query_read": False, "target_evaluation_performed": False}
    if len(rows) != 20:
        raise RuntimeError("Incomplete factorial preparation")
    (OUT / "preparation_complete.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print("PREPARED_QUERY_UNREAD: 20 frozen heads", flush=True)


def contrasts(values):
    """Return fixed factorial contrasts from four per-view metric values."""
    return {"physical_effect_u2": values["u2_physical"] - values["u2_carrier"],
            "physical_effect_wdcnn": values["wdcnn_physical"] - values["wdcnn_carrier"],
            "encoder_x_physics_interaction": (values["u2_physical"] - values["u2_carrier"])
                                             - (values["wdcnn_physical"] - values["wdcnn_carrier"]),
            "fused_u2_minus_fused_wdcnn": values["u2_physical"] - values["wdcnn_physical"]}


def score():
    registry_path = OUT / "preparation_complete.json"
    registry = json.loads(registry_path.read_text())
    if registry["status"] != "PREPARED_QUERY_UNREAD" or len(registry["head_rows"]) != 20:
        raise RuntimeError("Incomplete or changed preparation")
    if sha256(PROTOCOL) != registry["protocol_sha256"] or sha256(Path(__file__)) != registry["script_sha256"]:
        raise RuntimeError("Protocol or runner changed after preparation")
    if existing_reference_hashes() != registry["reference_hashes"]:
        raise RuntimeError("Earlier MCC5 result changed")
    for rel, expected in registry["model_hashes"].items():
        if sha256(ROOT / rel) != expected:
            raise RuntimeError(f"Frozen checkpoint changed: {rel}")
    packets = {}
    for row in registry["head_rows"]:
        path = OUT / row["file"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Frozen head changed: {path}")
        packets[(row["seed"], row["view"])] = joblib.load(path)
    if set(packets) != {(s, v) for s in SEEDS for v in VIEWS}:
        raise RuntimeError("Incomplete seed × condition grid")
    lock = OUT / "target_evaluation.lock"
    event = {"event": "POST_TARGET_FACTORIAL_SUPPLEMENT", "started_utc": datetime.now(timezone.utc).isoformat(),
             "preparation_sha256": sha256(registry_path), "protocol_sha256": sha256(PROTOCOL),
             "runner_sha256": sha256(Path(__file__)), "query_already_accessed_historically": True,
             "model_selection_allowed": False}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(event, stream, indent=2)
    query_path = BASE / "cache/query_after_selection.npz"
    with np.load(query_path, allow_pickle=False) as archive:
        ids, roles, y = (archive[k] for k in ("ids", "roles", "labels"))
        manifest = read_csv(BASE / "protocol/protocol_run_manifest.csv")
        validate_query(roles, ids, y, manifest)
        waves, rpm, phys = (archive[k] for k in ("distributed", "rpm_distributed", "physical"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    grid, metric_rows, pred_rows = {}, [], []
    for seed in SEEDS:
        u2_model = load_model(BASE / f"run/checkpoints/u2hz_seed{seed}.pt", device)
        u2_blocks, _ = infer_blocks(u2_model, waves, rpm, device, u2=True)
        del u2_model
        token = u2_blocks.mean(1)
        if not np.array_equal(token.reshape(27, 8, 128), np.repeat(token[:, :128, None], 8, axis=2).transpose(0, 2, 1)):
            # Simpler direct check below also catches nonidentical token contents.
            if np.max(np.abs(token.reshape(27, 8, 128) - token[:, None, :128])) > 1e-6:
                raise RuntimeError("U2 duplicate carrier tokens diverged")
        payload = torch.load(P0 / f"checkpoints/wdcnn_widthMatched_seed{seed}.pt",
                             map_location="cpu", weights_only=False)
        wd_model = WidthMatchedWDCNN(payload["width"]).to(device)
        wd_model.load_state_dict(payload["state"])
        wd_blocks, _ = infer_blocks(wd_model, waves, rpm, device)
        del wd_model
        features = feature_views(token[:, :128], wd_blocks.mean(1), phys)
        for name in VIEWS:
            packet = packets[(seed, name)]
            if packet["query_execution_enabled"] is not False:
                raise RuntimeError("Head was not frozen")
            proba = packet["head"].predict_proba(packet["scaler"].transform(features[name]))
            pred = packet["head"].classes_[proba.argmax(1)]
            grid[(seed, name)] = pred
            metric_rows.append({"seed": seed, "view": name, "n_query_runs": 27, **metric(y, pred)})
            for i, label in enumerate(pred):
                pred_rows.append({"seed": seed, "view": name, "run_id": str(ids[i]),
                                  "true": int(y[i]), "prediction": int(label),
                                  "correct": int(label == y[i])})
        print(f"scored seed {seed}: all four frozen cells", flush=True)
    write_csv(OUT / "seed_metrics.csv", metric_rows)
    write_csv(OUT / "predictions.csv", pred_rows)
    arrays = {v: np.stack([grid[(s, v)] for s in SEEDS]) for v in VIEWS}
    summaries = []
    for view in VIEWS:
        part = [r for r in metric_rows if r["view"] == view]
        row = {"view": view, "seeds": 5, "query_runs": 27}
        for key in ("accuracy", "macro_f1", "recall_NRF", "recall_IR", "recall_OR"):
            vals = 100 * np.asarray([r[key] for r in part])
            row[key + "_mean_percent"] = float(vals.mean())
            row[key + "_sample_sd_pp"] = float(vals.std(ddof=1))
        summaries.append(row)
    write_csv(OUT / "summary.csv", summaries)
    paired = []
    for seed in SEEDS:
        for name, a, b in (("physical_effect_u2", "u2_physical", "u2_carrier"),
                           ("physical_effect_wdcnn", "wdcnn_physical", "wdcnn_carrier"),
                           ("fused_u2_minus_fused_wdcnn", "u2_physical", "wdcnn_physical")):
            ca, cb = grid[(seed, a)] == y, grid[(seed, b)] == y
            paired.append({"seed": seed, "contrast": name,
                           "corrected_errors": int(np.sum(ca & ~cb)),
                           "introduced_errors": int(np.sum(~ca & cb))})
    write_csv(OUT / "paired_errors.csv", paired)
    observed = {}
    for score_key in ("accuracy", "macro_f1"):
        means = {v: 100 * np.mean([r[score_key] for r in metric_rows if r["view"] == v]) for v in VIEWS}
        observed[score_key] = contrasts(means)
    rng = np.random.default_rng(20261007)
    groups = [np.flatnonzero(y == c) for c in (0, 1, 2)]
    reps = {f"{m}_{c}": np.empty(10000) for m in ("accuracy", "macro_f1") for c in observed[m]}
    for b in range(10000):
        seed_idx = rng.integers(0, 5, 5)
        run_idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boot = {v: vector_metrics(arrays[v][np.ix_(seed_idx, run_idx)], y[run_idx]) for v in VIEWS}
        for k, metric_index in (("accuracy", 0), ("macro_f1", 1)):
            for name, value in contrasts({v: 100 * boot[v][metric_index] for v in VIEWS}).items():
                reps[f"{k}_{name}"][b] = value
    np.savez_compressed(OUT / "bootstrap_replicates.npz", **reps)
    effects = []
    for k in ("accuracy", "macro_f1"):
        for name, value in observed[k].items():
            dist = reps[f"{k}_{name}"]
            effects.append({"metric": k, "contrast": name, "observed_delta_pp": value,
                            "percentile_2p5_pp": float(np.quantile(dist, .025)),
                            "percentile_97p5_pp": float(np.quantile(dist, .975))})
    write_csv(OUT / "factorial_effects.csv", effects)
    if existing_reference_hashes() != registry["reference_hashes"]:
        raise RuntimeError("Original result changed during scoring")
    result = {"status": "COMPLETE_POST_TARGET_EXPLANATORY", "query_runs": 27,
              "prediction_rows": len(pred_rows), "summary": summaries, "effects": effects,
              "historical_full_RAMEF_accuracy_percent": 96.30,
              "historical_anchor_not_a_factorial_cell": True,
              "target_cohort_previously_accessed": True, "model_fitting_on_query": False,
              "original_target_artifacts_unchanged": True,
              "query_cache_sha256": sha256(query_path)}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    event.update({"completed_utc": datetime.now(timezone.utc).isoformat(),
                  "result_sha256": sha256(OUT / "result.json"),
                  "predictions_sha256": sha256(OUT / "predictions.csv")})
    lock.write_text(json.dumps(event, indent=2), encoding="utf-8")
    with (OUT / "MANIFEST.sha256").open("w", encoding="utf-8") as stream:
        for p in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
            stream.write(f"{sha256(p)} *{p.relative_to(OUT).as_posix()}\n")
    print("COMPLETE_POST_TARGET_EXPLANATORY", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "score"))
    phase = parser.parse_args().phase
    with threadpool_limits(limits=4):
        prepare() if phase == "prepare" else score()
