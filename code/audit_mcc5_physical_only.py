"""Post-target descriptive physical-only audit for MCC5.

This control was not preregistered and cannot alter the confirmatory decision.
It is added solely to determine whether the 63-D view is independently
sufficient or complementary to the carrier on the already evaluated cohort.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_mcc5_ramef_confirmatory import fit_head, metrics, write_csv


def load(path: Path) -> dict:
    cache = np.load(path, allow_pickle=False)
    return {key: cache[key] for key in cache.files}


def main() -> None:
    base = Path("outputs/staged/MCC5_RAMEF_CONFIRMATORY")
    development = load(base / "cache" / "development.npz")
    query = load(base / "cache" / "query_after_selection.npz")
    source = np.flatnonzero(development["roles"] == "source")
    support = np.flatnonzero(development["roles"] == "target_support")
    pred = fit_head(development["physical"][source], development["labels"][source],
                    development["physical"][support], development["labels"][support],
                    query["physical"], 20261199)
    value = metrics(query["labels"], pred)
    rows = [{"run_id": query["ids"][i], "context": query["contexts"][i],
             "severity": query["severities"][i], "true": int(query["labels"][i]),
             "prediction": int(pred[i]), "correct": int(pred[i] == query["labels"][i])}
            for i in range(len(pred))]
    output = base / "posthoc_physical_only"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", rows)
    result = {"audit": "POSTHOC_PHYSICAL_ONLY", "confirmatory_decision_unchanged": True,
              "target_already_evaluated": True, **value}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
