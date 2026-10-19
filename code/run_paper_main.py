"""One-shot, read-only PAPER_MAIN_V1 evaluation of the frozen B2 method.

This runner performs no training, tuning, threshold selection, or checkpoint
selection.  The method and ten support episodes must already be frozen.  The
same 90-measurement query cohort is evaluated under each support episode to
quantify support-set variability.  Episode 0 is required to reproduce the P2
ordinary-prototype baseline exactly before the remaining results are accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from bottleneck_deconvolution import TokenExtractor, load_model, ordinary_logits
from run_c2of import metrics


CHECKPOINT_SHA256 = "cce4c3e22de6e1416a8df35bb5244e1c8cd2d5d3b12155a886da7dcf4553f098"
PROTOCOL_MANIFEST_SHA256 = "f3fd66b7723534007ed4b44ee708f6597eeb1a24565aacdee295f2102fa8963f"
EPISODES = 10
WINDOW_LENGTH = 16384


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_contract(protocol_dir: Path) -> tuple[list[dict], list[list[dict]], list[str]]:
    manifest = protocol_dir / "MANIFEST.sha256"
    if sha256(manifest) != PROTOCOL_MANIFEST_SHA256:
        raise RuntimeError("PAPER_MAIN_V1 protocol manifest mismatch")
    queries, supports, split_hashes = [], [], []
    fixed_ids = None
    for episode in range(EPISODES):
        split_path = protocol_dir / f"split_episode_{episode:02d}.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        current_query = [row for row in payload["windows"] if row["role"] == "query"]
        current_support = [row for row in payload["windows"] if row["role"] == "support"]
        ids = [row["measurement_id"] for row in current_query]
        fixed_ids = ids if fixed_ids is None else fixed_ids
        if ids != fixed_ids or len(current_query) != 90 or len(current_support) != 30:
            raise RuntimeError(f"Episode {episode} violates the frozen cohort")
        if set(ids) & {row["measurement_id"] for row in current_support}:
            raise RuntimeError(f"Episode {episode} has measurement leakage")
        if Counter(int(row["label"]) for row in current_support) != Counter({0: 10, 1: 10, 2: 10}):
            raise RuntimeError(f"Episode {episode} support class mismatch")
        if episode == 0:
            queries = current_query
        supports.append(current_support); split_hashes.append(sha256(split_path))
    return queries, supports, split_hashes


def hierarchical_bootstrap(correct: np.ndarray, repeats: int = 5000,
                           seed: int = 20260811) -> dict:
    """Bootstrap support episodes and measurement identities hierarchically."""
    rng = np.random.default_rng(seed)
    episode_count, query_count = correct.shape
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        episode_indices = rng.integers(0, episode_count, episode_count)
        query_indices = rng.integers(0, query_count, query_count)
        values[index] = correct[np.ix_(episode_indices, query_indices)].mean()
    low, high = np.quantile(values, [0.025, 0.975])
    return {"repeats": repeats, "mean": float(values.mean()),
            "ci_low": float(low), "ci_high": float(high), "seed": seed}


def reproduce_p2_episode0(rows: list[dict]) -> dict:
    path = Path("outputs/staged/MPC_LR_P2/query_predictions.csv")
    expected = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[row["measurement_id"]] = int(row["base_pred"])
    mismatches = [row["measurement_id"] for row in rows
                  if expected.get(row["measurement_id"]) != int(row["prediction"])]
    return {"reference": str(path), "expected_measurements": len(expected),
            "mismatches": mismatches, "passed": not mismatches and len(expected) == 90}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path,
                        default=Path("paper/evidence/protocol/PAPER_MAIN_V1"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/staged/PAPER_MAIN_V1_B2"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()

    checkpoint = Path("frozen/M2_LOCKED_v1_20260715/model/checkpoints/best.pt")
    config = Path("frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen B2 checkpoint mismatch")
    query, episode_supports, split_hashes = load_contract(args.protocol_dir)
    args.output.mkdir(parents=True, exist_ok=True)

    # Extract each physical measurement once.  This cache is created only after
    # method/protocol freeze and is never consulted for a training decision.
    support_by_id = {}
    for rows in episode_supports:
        for row in rows:
            support_by_id.setdefault(row["measurement_id"], row)
    unique_support = sorted(support_by_id.values(), key=lambda row: row["measurement_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(config, checkpoint, device)
    extractor = TokenExtractor(model, device)
    query_tokens, query_bounds = extractor.extract(query, WINDOW_LENGTH, args.batch_size)
    support_tokens, support_bounds = extractor.extract(unique_support, WINDOW_LENGTH, args.batch_size)
    token_by_id = {row["measurement_id"]: support_tokens[index]
                   for index, row in enumerate(unique_support)}

    truth = np.asarray([int(row["label"]) for row in query])
    codes = [row["bearing_code"] for row in query]
    episode_metrics, prediction_rows, prediction_matrix = [], [], []
    for episode, support in enumerate(episode_supports):
        tokens = torch.stack([token_by_id[row["measurement_id"]] for row in support])
        labels = torch.tensor([int(row["label"]) for row in support], dtype=torch.long)
        with torch.inference_mode():
            logits = ordinary_logits(model, tokens, labels, query_tokens)
        prediction = logits.argmax(1).cpu().numpy()
        result = metrics(truth, prediction, codes)
        result.update({"episode": episode, "split_sha256": split_hashes[episode],
                       "support_measurements": len(support)})
        episode_metrics.append(result); prediction_matrix.append(prediction)
        rows = [{"episode": episode, "measurement_id": item["measurement_id"],
                 "bearing_code": item["bearing_code"], "true": int(target),
                 "prediction": int(guess), "correct": int(target == guess)}
                for item, target, guess in zip(query, truth, prediction)]
        prediction_rows.extend(rows)
        write_csv(args.output / f"predictions_episode_{episode:02d}.csv", rows)

    reproduction = reproduce_p2_episode0(
        [row for row in prediction_rows if int(row["episode"]) == 0])
    if not reproduction["passed"]:
        raise RuntimeError(f"Episode 0 failed P2 reproduction: {reproduction}")

    prediction_matrix_np = np.stack(prediction_matrix)
    correct = prediction_matrix_np == truth[None, :]
    numeric = {
        "accuracy": np.asarray([row["accuracy"] for row in episode_metrics]),
        "macro_f1": np.asarray([row["macro_f1"] for row in episode_metrics]),
        "recall_N": np.asarray([row["recall"][0] for row in episode_metrics]),
        "recall_IR": np.asarray([row["recall"][1] for row in episode_metrics]),
        "recall_OR": np.asarray([row["recall"][2] for row in episode_metrics]),
        "worst_fault_code_accuracy": np.asarray(
            [row["worst_fault_code_accuracy"] for row in episode_metrics]),
    }
    aggregate = {name: {"mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "min": float(values.min()), "max": float(values.max())}
                 for name, values in numeric.items()}
    stability = []
    for index, item in enumerate(query):
        guesses = prediction_matrix_np[:, index]
        majority = int(np.bincount(guesses, minlength=3).argmax())
        stability.append({"measurement_id": item["measurement_id"],
                          "bearing_code": item["bearing_code"], "true": int(truth[index]),
                          "majority_prediction": majority,
                          "episode_agreement": float(np.mean(guesses == majority)),
                          "correct_episodes": int(np.sum(guesses == truth[index]))})
    write_csv(args.output / "episode_metrics.csv", [
        {"episode": row["episode"], "accuracy": row["accuracy"],
         "macro_f1": row["macro_f1"], "recall_N": row["recall"][0],
         "recall_IR": row["recall"][1], "recall_OR": row["recall"][2],
         "worst_fault_code_accuracy": row["worst_fault_code_accuracy"],
         "confusion_matrix": json.dumps(row["confusion_matrix"]),
         "split_sha256": row["split_sha256"]}
        for row in episode_metrics])
    write_csv(args.output / "prediction_stability.csv", stability)

    summary = {
        "status": "FROZEN_EVALUATION_COMPLETE",
        "method": "H2_M2_CROSS_SPEED_MARGIN",
        "selection_or_tuning_performed": False,
        "checkpoint_sha256": sha256(checkpoint),
        "protocol_manifest_sha256": sha256(args.protocol_dir / "MANIFEST.sha256"),
        "device": str(device),
        "input_window_length": WINDOW_LENGTH,
        "episodes": EPISODES,
        "query_measurements": len(query),
        "unique_support_measurements": len(unique_support),
        "episode_0_p2_reproduction": reproduction,
        "aggregate": aggregate,
        "hierarchical_bootstrap_accuracy": hierarchical_bootstrap(
            correct, args.bootstrap_repeats),
        "episode_metrics": episode_metrics,
        "query_feature_extracted_after_freeze": True,
        "query_used_for_model_selection": False,
        "query_bounds_audit_count": len(query_bounds),
        "support_bounds_audit_count": len(support_bounds),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artefacts = sorted(path for path in args.output.iterdir()
                       if path.is_file() and path.name != "RESULTS_FREEZE.sha256")
    with (args.output / "RESULTS_FREEZE.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for path in artefacts:
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps({"status": summary["status"], "reproduction": reproduction["passed"],
                      "aggregate": aggregate,
                      "bootstrap": summary["hierarchical_bootstrap_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
