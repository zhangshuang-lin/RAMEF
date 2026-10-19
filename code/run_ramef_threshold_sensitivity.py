"""Query-free sensitivity of the RAMEF occurrence threshold/statistics.

The registered physical signal operations are re-evaluated once for all PU
source measurements and the 15 authorised DS supports. Multiple occurrence
thresholds and a threshold-free descriptor are then compared with the frozen
U2-Hz-M8 embeddings. No DS query row, waveform, feature or label is loaded.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis

from bottleneck_deconvolution import load_vibration, sha256
from prepare_splits import scan_records
from racp_or import BANDS, WINDOW, _soft_harmonic_ratio, uniform_nonoverlap_starts
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_ramef_design_sensitivity import (SEEDS, build_pseudo_episodes,
                                          fit_lr, metrics, write_csv)
from run_tser_hb_support_p0 import support_only


THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


def extract_evidence(rows: list[dict], cache_path: Path) -> np.ndarray:
    ids = [row["measurement_id"] for row in rows]
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        if cache["ids"].tolist() != ids:
            raise RuntimeError("Stale threshold evidence cache")
        return cache["evidence"].astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_evidence = []
    sample_rate = 64000.0; nyquist = sample_rate / 2.0
    filters = [butter(4, (low / nyquist, high / nyquist), btype="bandpass", output="sos")
               for low, high in BANDS]
    for index, row in enumerate(rows):
        signal = np.asarray(load_vibration(row["path"], row.get("signal_key", "signal")),
                            dtype=np.float32).reshape(-1)
        starts = uniform_nonoverlap_starts(len(signal))
        blocks = np.stack([signal[start:start + WINDOW] for start in starts])
        per_band = []
        for sos in filters:
            filtered = sosfiltfilt(sos, blocks, axis=-1)
            envelopes = np.abs(hilbert(filtered, axis=-1))
            band_rows = []
            for block, envelope in zip(filtered, envelopes):
                band_rows.append([
                    _soft_harmonic_ratio(envelope, sample_rate, float(row["rpm"]), 3.05),
                    _soft_harmonic_ratio(envelope, sample_rate, float(row["rpm"]), 4.95),
                    float(kurtosis(block, fisher=True, bias=False)),
                    float(np.log1p(np.sqrt(np.mean(block ** 2)))),
                ])
            per_band.append(band_rows)
        all_evidence.append(per_band)
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f"physical evidence {index + 1}/{len(rows)}", flush=True)
    evidence = np.asarray(all_evidence, dtype=np.float32)
    np.savez_compressed(cache_path, ids=np.asarray(ids), evidence=evidence)
    return evidence


def descriptor(evidence: np.ndarray, threshold: float | None) -> np.ndarray:
    output = []
    for measurement in evidence:
        features = []
        for band in measurement:
            for column in range(band.shape[1]):
                values = band[:, column]
                features.extend([float(values.mean()), float(np.median(values)),
                                 float(values.std())])
                if threshold is not None:
                    features.append(float(np.mean(values > threshold)))
        best_or = measurement[:, :, 0].max(axis=0)
        best_ir = measurement[:, :, 1].max(axis=0)
        for values in (best_or, best_ir, best_or - best_ir):
            features.extend([float(values.mean()), float(np.median(values)),
                             float(values.std())])
            if threshold is not None:
                features.append(float(np.mean(values > threshold)))
            features.append(float(np.mean(np.sort(values)[-2:])))
        output.append(features)
    result = np.asarray(output, dtype=np.float32)
    if not np.isfinite(result).all(): raise RuntimeError("Non-finite descriptor")
    return result


def pseudo_target_summary(rows: list[dict], x: np.ndarray, episodes: list[dict]) -> dict:
    y = np.asarray([int(row["label"]) for row in rows]); values = []
    for episode in episodes:
        source, support, query = episode["source"], episode["support"], episode["query"]
        train = np.concatenate([source, support])
        weights = np.concatenate([np.ones(len(source)), np.full(len(support), 10.0)])
        pred = fit_lr(x[train], y[train], weights, x[query], 0.1, episode["seed"])
        values.append(metrics(y[query], pred, [rows[i]["bearing_code"] for i in query]))
    summary = {}
    for key in ("accuracy", "macro_f1", "recall_N", "recall_IR", "recall_OR",
                "recall_gap", "worst_fault_code_accuracy"):
        array = np.asarray([value[key] for value in values])
        summary[key + "_mean"] = float(array.mean())
        summary[key + "_std"] = float(array.std(ddof=1))
    return summary


def support_leave_two(rows: list[dict], x: np.ndarray, source_count: int) -> dict:
    source_y = np.asarray([int(row["label"]) for row in rows[:source_count]])
    support_rows = rows[source_count:]; support_y = np.asarray([int(row["label"])
                                                                for row in support_rows])
    truth, pred, codes = [], [], []
    for fold, pair in enumerate(itertools.combinations(range(15), 2)):
        held = np.asarray(pair); keep = np.asarray([i for i in range(15) if i not in pair])
        train_x = np.vstack([x[:source_count], x[source_count:][keep]])
        train_y = np.concatenate([source_y, support_y[keep]])
        weights = np.concatenate([np.ones(source_count), np.full(len(keep), 10.0)])
        local = fit_lr(train_x, train_y, weights, x[source_count:][held], 0.1,
                       20261200 + fold)
        truth.extend(support_y[held]); pred.extend(local)
        codes.extend([support_rows[i]["bearing_code"] for i in held])
    return metrics(np.asarray(truth), np.asarray(pred), codes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-cache", type=Path,
                        default=Path("outputs/staged/RACP_OR_INFORMATION_P0/authorised_source_support_features.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/staged/RAMEF_THRESHOLD_SENSITIVITY"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = sorted([row for row in scan_records(Path("D:/Paderborn"))
                     if row["domain"] in SOURCE_DOMAINS], key=lambda row: row["measurement_id"])
    support, query_count = support_only(Path("split_independent_measurement.json"))
    rows = source + support
    carrier_cache = np.load(args.carrier_cache, allow_pickle=False)
    if carrier_cache["ids"].tolist() != [row["measurement_id"] for row in rows]:
        raise RuntimeError("Carrier cache mismatch")
    mean8 = carrier_cache["blocks"].astype(np.float32).mean(1)
    evidence_cache = args.output / "source_support_raw_evidence.npz"
    evidence = extract_evidence(rows, evidence_cache)
    episodes = build_pseudo_episodes(source, SEEDS)
    result_rows = []
    candidates = [(f"threshold_{value:g}", value) for value in THRESHOLDS]
    candidates.append(("no_occurrence_fraction", None))
    for name, threshold in candidates:
        physical = descriptor(evidence, threshold)
        x = np.concatenate([mean8, physical], axis=1)
        pseudo = pseudo_target_summary(source, x[:len(source)], episodes)
        support_metric = support_leave_two(rows, x, len(source))
        row = {"variant": name, "threshold": threshold,
               "physical_dimensions": int(physical.shape[1]), **pseudo}
        row.update({"support_l2o_" + key: value for key, value in support_metric.items()
                    if key != "confusion_matrix"})
        row["support_l2o_confusion_matrix"] = json.dumps(support_metric["confusion_matrix"])
        result_rows.append(row)
        print(name, pseudo, support_metric, flush=True)
    registered = next(row for row in result_rows if row["variant"] == "threshold_0.5")
    result = {
        "experiment": "RAMEF_THRESHOLD_SENSITIVITY",
        "status": "COMPLETE_QUERY_FREE",
        "registered_variant": registered,
        "variants": result_rows,
        "firewall": {"ds_query_rows_in_split": query_count,
                     "ds_query_measurements_loaded": 0,
                     "ds_query_waveforms_loaded": False,
                     "ds_query_features_loaded": False,
                     "ds_query_labels_used": False},
        "carrier_cache_sha256": sha256(args.carrier_cache),
        "evidence_cache_sha256": sha256(evidence_cache),
        "split_sha256": sha256(Path("split_independent_measurement.json")),
    }
    write_csv(args.output / "sensitivity.csv", result_rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
