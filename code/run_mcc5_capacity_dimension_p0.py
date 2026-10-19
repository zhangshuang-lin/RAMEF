"""Run the frozen MCC5 dimension/capacity P0 on development data only.

The runner has no query evaluation stage. All reported accuracies are either
source-validation run accuracy or held-target-support accuracy, never query
test accuracy. Frozen source and support waveforms are sufficient for P0.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from capacity_matched_baselines import WidthMatchedWDCNN, nearest_width, parameter_count
from fair_baseline_models import make_model as make_wdcnn
from mcc5_confirmatory_data import sha256
from run_mcc5_ramef_confirmatory import SEEDS, balanced_source_weights, load_model, make_model, standardise

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/staged/MCC5_RAMEF_CONFIRMATORY"
T2 = ROOT / "outputs/staged/MCC5_TRANSFERRED_EVIDENCE_PACKAGE/T2_benchmark"
PROTOCOL = ROOT / "MCC5_CAPACITY_DIMENSION_P0_PROTOCOL.md"
VIEWS = ("U2-Hz-1024", "U2-Hz-sourcePCA128", "WDCNN-128", "WDCNN-widthMatched-128")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_development_roles(roles, ids):
    expected = {"source": 42, "source_validation": 42, "target_support": 15}
    if Counter(roles.tolist()) != expected or len(ids) != 99 or len(set(ids.tolist())) != 99:
        raise RuntimeError("Development identity/role firewall failed: query or changed roles")


def load_development():
    path = BASE / "cache/development.npz"
    selection = json.loads((BASE / "run/selection_complete.json").read_text())
    if sha256(path) != selection["development_cache_sha256"]:
        raise RuntimeError("Frozen development cache hash changed")
    with np.load(path, allow_pickle=False) as archive:
        roles, ids = archive["roles"], archive["ids"]
        validate_development_roles(roles, ids)
        if bool(archive["query_cache"]):
            raise RuntimeError("Query cache prohibited")
        return {k: archive[k] for k in ("roles", "ids", "labels", "distributed", "rpm_distributed")}


def metric(y, pred):
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"accuracy": float(np.mean(y == pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_NRF": float(recall[0]), "recall_IR": float(recall[1]), "recall_OR": float(recall[2]),
            "errors": int(np.sum(y != pred)), "confusion_matrix": json.dumps(confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())}


@torch.no_grad()
def infer_blocks(model, waves, rpms, device, u2=False):
    model.eval(); z, logits = [], []
    x = torch.from_numpy(waves.reshape(-1, 3072))
    rpm = torch.from_numpy(rpms.reshape(-1))
    batch = 24 if u2 else 64
    for start in range(0, len(x), batch):
        value = standardise(x[start:start+batch]).to(device)
        out = model(value, rpm[start:start+batch].to(device)) if u2 else model(value)
        z.append(out["embedding"].cpu()); logits.append(out["logits" if u2 else "class_logits"].cpu())
    return (torch.cat(z).reshape(len(waves), 8, -1).numpy(),
            torch.cat(logits).reshape(len(waves), 8, 3).mean(1).numpy())


def fit_predict(train_x, train_y, test_x, weights, seed):
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=.1, solver="lbfgs", max_iter=4000, random_state=seed)
    model.fit(scaler.transform(train_x), train_y, sample_weight=weights)
    return model.predict(scaler.transform(test_x))


def train_width_model(cache, width, seed, checkpoint, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    src = np.flatnonzero(cache["roles"] == "source")
    val = np.flatnonzero(cache["roles"] == "source_validation")
    x = torch.from_numpy(cache["distributed"][src].reshape(-1, 3072))
    y = torch.tensor(np.repeat(cache["labels"][src], 8), dtype=torch.long)
    counts = Counter(y.tolist())
    weights = torch.tensor([1/counts[int(v)] for v in y], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, max(counts.values())*3, replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(x, y), batch_size=12, sampler=sampler, num_workers=0)
    model = WidthMatchedWDCNN(width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history, best, best_epoch, state = [], -1., None, None
    for epoch in range(12):
        model.train(); losses = []
        for wave, truth in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(standardise(wave).to(device))["class_logits"]
            loss = F.cross_entropy(logits, truth.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step(); losses.append(float(loss.detach()))
        _, logits = infer_blocks(model, cache["distributed"][val], cache["rpm_distributed"][val], device)
        value = metric(cache["labels"][val], logits.argmax(1))
        history.append({"seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)),
                        "source_validation_macro_f1": value["macro_f1"]})
        if value["macro_f1"] > best + 1e-12:
            best, best_epoch = value["macro_f1"], epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"seed={seed} width epoch={epoch+1}/12 source-val-F1={value['macro_f1']:.4f}", flush=True)
    torch.save({"state": state, "seed": seed, "width": width, "best_epoch": best_epoch,
                "best_source_validation_macro_f1": best}, checkpoint)
    model.load_state_dict(state)
    return model, history, best_epoch, best


def run(output):
    if (output / "source_audit.lock").exists():
        raise RuntimeError("Audit already started; existing archive is preserved")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    target_params = parameter_count(make_model())
    width = nearest_width(target_params)
    matched_params = parameter_count(WidthMatchedWDCNN(width))
    relative = abs(matched_params-target_params)/target_params
    if relative > .01:
        raise RuntimeError("Width control exceeds frozen 1% capacity tolerance")
    original_files = [BASE / f"run/checkpoints/u2hz_seed{s}.pt" for s in SEEDS]
    original_files += [T2 / f"checkpoints/wdcnn_seed{s}.pt" for s in SEEDS]
    originals = {str(p.relative_to(ROOT)): sha256(p) for p in original_files}
    selected = json.loads((BASE/"run/selection_complete.json").read_text())
    for record in selected["checkpoints"]:
        path = BASE/f"run/checkpoints/u2hz_seed{record['seed']}.pt"
        if sha256(path) != record["sha256"]:
            raise RuntimeError("Original U2-Hz selection hash mismatch")
    with (T2/"checkpoints.csv").open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            if record["model"] == "wdcnn":
                path = T2/f"checkpoints/wdcnn_seed{record['seed']}.pt"
                if sha256(path) != record["sha256"]:
                    raise RuntimeError("Original WDCNN selection hash mismatch")
    registry = {"started_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
                "protocol_sha256": sha256(PROTOCOL), "runner_sha256": sha256(Path(__file__)),
                "model_code_sha256": sha256(ROOT/"capacity_matched_baselines.py"),
                "development_cache_sha256": sha256(BASE/"cache/development.npz"),
                "u2_parameters": target_params, "width_matched_parameters": matched_params,
                "relative_parameter_difference": relative, "base_channels": width,
                "original_checkpoints": originals, "query_evaluation_authorized": False}
    (output/"source_audit.lock").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    shutil.copyfile(PROTOCOL, output/"protocol_frozen.md")
    cache = load_development()
    source = np.flatnonzero(cache["roles"] == "source")
    validation = np.flatnonzero(cache["roles"] == "source_validation")
    support = np.flatnonzero(cache["roles"] == "target_support")
    y = cache["labels"]
    source_weights = balanced_source_weights(y[source])
    rows, predictions, histories, details = [], [], [], []
    checkpoint_dir = output/"checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    for seed in SEEDS:
        u2 = load_model(BASE/f"run/checkpoints/u2hz_seed{seed}.pt", device)
        z2, log2 = infer_blocks(u2, cache["distributed"], cache["rpm_distributed"], device, u2=True)
        del u2
        pca = PCA(n_components=128, svd_solver="full", whiten=False)
        pca.fit(z2[source].reshape(-1, 1024))
        compressed = pca.transform(z2.reshape(-1, 1024)).reshape(99, 8, 128)
        baseline = make_wdcnn("wdcnn").to(device)
        baseline.load_state_dict(torch.load(T2/f"checkpoints/wdcnn_seed{seed}.pt", map_location="cpu", weights_only=False)["state"])
        zw, logw = infer_blocks(baseline, cache["distributed"], cache["rpm_distributed"], device)
        del baseline
        wide_path = checkpoint_dir/f"wdcnn_widthMatched_seed{seed}.pt"
        wide, history, best_epoch, best_f1 = train_width_model(cache, width, seed, wide_path, device)
        histories.extend(history)
        zm, logm = infer_blocks(wide, cache["distributed"], cache["rpm_distributed"], device)
        del wide
        views = dict(zip(VIEWS, [z2.mean(1), compressed.mean(1), zw.mean(1), zm.mean(1)]))
        details.append({"seed": seed, "source_block_rows_for_pca": len(source)*8,
                        "pca_retained_variance": float(pca.explained_variance_ratio_.sum()),
                        "width_best_epoch_zero_based": best_epoch, "width_source_validation_best_f1": best_f1,
                        "width_checkpoint_sha256": sha256(wide_path)})
        np.savez_compressed(output/f"development_embeddings_seed{seed}.npz", ids=cache["ids"],
                            roles=cache["roles"], labels=y, **{k.replace("-", "_"): v for k,v in views.items()},
                            pca_components=pca.components_, pca_mean=pca.mean_)
        for name, x in views.items():
            conditions = {}
            conditions["source_validation_source_only"] = (validation, fit_predict(x[source], y[source], x[validation], source_weights, seed))
            indices = np.concatenate([source, support])
            weights = np.concatenate([source_weights, np.full(len(support), 10.)])
            conditions["source_validation_source_plus_support"] = (validation, fit_predict(x[indices], y[indices], x[validation], weights, seed))
            loo = np.empty(len(support), dtype=int)
            for held, held_id in enumerate(support):
                keep = np.delete(support, held)
                indices = np.concatenate([source, keep])
                weights = np.concatenate([source_weights, np.full(len(keep), 10.)])
                loo[held] = fit_predict(x[indices], y[indices], x[[held_id]], weights, seed)[0]
            conditions["held_target_support_loo"] = (support, loo)
            for condition, (indices, pred) in conditions.items():
                rows.append({"seed": seed, "view": name, "condition": condition,
                             "n_runs": len(indices), "embedding_dim": x.shape[1], **metric(y[indices], pred)})
                for index, prediction in zip(indices, pred):
                    predictions.append({"seed": seed, "view": name, "condition": condition,
                                        "run_id": str(cache["ids"][index]), "role": str(cache["roles"][index]),
                                        "true": int(y[index]), "prediction": int(prediction)})
        for name, logits in [(VIEWS[0], log2), (VIEWS[2], logw), (VIEWS[3], logm)]:
            rows.append({"seed": seed, "view": name, "condition": "source_validation_neural_head",
                         "n_runs": len(validation), "embedding_dim": 1024 if name == VIEWS[0] else 128,
                         **metric(y[validation], logits[validation].argmax(1))})
        write_csv(output/"seed_metrics.csv", rows)
        write_csv(output/"training_history.csv", histories)
        print(f"seed={seed} COMPLETE; PCA variance={details[-1]['pca_retained_variance']:.4f}", flush=True)
    summaries = []
    for view, condition in sorted({(r["view"], r["condition"]) for r in rows}):
        subset = [r for r in rows if (r["view"], r["condition"]) == (view, condition)]
        summary = {"view": view, "condition": condition, "seeds": len(subset), "n_runs_per_seed": subset[0]["n_runs"]}
        for key in ("accuracy", "macro_f1", "recall_NRF", "recall_IR", "recall_OR"):
            values = [r[key]*100 for r in subset]
            summary[key+"_mean_percent"] = float(np.mean(values))
            summary[key+"_sample_sd_pp"] = float(np.std(values, ddof=1))
        summaries.append(summary)
    if any(sha256(ROOT/path) != value for path,value in originals.items()):
        raise RuntimeError("An original checkpoint was modified")
    write_csv(output/"predictions.csv", predictions)
    write_csv(output/"summary.csv", summaries)
    write_csv(output/"model_details.csv", details)
    result = {"status": "COMPLETE_QUERY_FREE_P0", "registry": registry, "summaries": summaries,
              "details": details, "target_query_loaded": False, "original_checkpoint_hashes_unchanged": True,
              "interpretation": "Source-validation and held-support readouts only; no target-query accuracy conclusion",
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    (output/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output/"MANIFEST.sha256").open("w", encoding="utf-8") as stream:
        for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
            stream.write(f"{sha256(path)} *{path.relative_to(output).as_posix()}\n")
    print("COMPLETE_QUERY_FREE_P0", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/staged/MCC5_CAPACITY_DIMENSION_P0")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.output.resolve())
