"""Synthetic unit tests only; no synthetic result is a paper reference result."""
from __future__ import annotations

import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENTRY = HERE / "replay_paper_results.py"
if not ENTRY.exists():
    ENTRY = HERE.parent / "scripts/replay_paper_results.py"
SPEC = importlib.util.spec_from_file_location("replay_paper_results_under_test", ENTRY)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class MetricTests(unittest.TestCase):
    def test_confusion_accuracy_macro_f1_and_recalls(self):
        result = REPLAY.compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [0, 0, 2]])
        self.assertAlmostEqual(result["accuracy"], 400 / 6)
        self.assertAlmostEqual(result["macro_f1"], (200 / 3 + 50 + 80) / 3)
        self.assertEqual([result[f"recall_{c}"] for c in range(3)], [50, 50, 100])
        self.assertEqual(result["errors"], 2)

    def test_missing_predicted_class_is_zero_not_nan(self):
        result = REPLAY.compute_metrics([0, 1, 2], [0, 0, 0])
        self.assertAlmostEqual(result["macro_f1"], 50 / 3)
        self.assertTrue(math.isfinite(result["macro_f1"]))

    def test_invalid_metrics_input(self):
        for truth, prediction in (([], []), ([0], []), ([3], [0]), ([0], [-1])):
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises(ValueError):
                    REPLAY.compute_metrics(truth, prediction)

    def test_seed_summary_uses_sample_sd(self):
        rows = [dict(dataset="pu", cohort="synthetic", method="toy", seed=str(seed),
                     **{field: score for field in REPLAY.METRICS})
                for seed, score in ((1, 80.0), (2, 100.0))]
        result = REPLAY.summarize_seeds(rows)[0]
        self.assertEqual(result["accuracy_mean"], 90.0)
        self.assertAlmostEqual(result["accuracy_sample_sd"], math.sqrt(200))


class IntegrityTests(unittest.TestCase):
    def test_identical_duplicate_collapses(self):
        row = dict(measurement_id="toy", true=1, prediction=2)
        unique = REPLAY.deduplicate_records([row, dict(row)], "measurement_id")
        self.assertEqual(unique, [row])

    def test_conflicting_duplicate_raises(self):
        a = dict(measurement_id="toy", true=1, prediction=2)
        for b in (dict(a, true=2), dict(a, prediction=1)):
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                REPLAY.deduplicate_records([a, b], "measurement_id")

    def test_expected_counts_and_shared_identifiers(self):
        rows = []
        for label, count in REPLAY.PU_COUNTS.items():
            for index in range(count):
                rows.append(dict(cohort="toy", seed="1", method="toy",
                                 measurement_id=f"class{label}_{index}", bearing_code=f"C{label}",
                                 true=label, prediction=label))
        rows.append(dict(rows[0]))
        scores = REPLAY.score_groups(rows, "pu")
        self.assertEqual(scores[0]["unique_count"], 171)
        self.assertEqual(scores[0]["duplicate_count"], 1)
        other = [dict(row, method="other") for row in rows[:-1]]
        other[0]["measurement_id"] = "different_identity"
        with self.assertRaisesRegex(ValueError, "identities or truths differ"):
            REPLAY.score_groups(rows + other, "pu")
        with self.assertRaisesRegex(ValueError, "class counts"):
            REPLAY.score_groups(rows[1:-1], "pu")

    def test_output_existing_directory_is_always_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                REPLAY.create_output_dir(root)
            target = root / "new"
            REPLAY.create_output_dir(target)
            self.assertTrue(target.is_dir())
            with self.assertRaises(FileExistsError):
                REPLAY.create_output_dir(target)

    def test_csv_requires_columns_and_valid_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(REPLAY.REQUIRED_PU))
                writer.writeheader()
                writer.writerow(dict(cohort="toy", seed="1", method="toy", measurement_id="toy",
                                     bearing_code="toy", true="3", prediction="0"))
            with self.assertRaisesRegex(ValueError, "must be 0, 1 or 2"):
                REPLAY.load_predictions(path, "pu")
            path.write_text("cohort,seed\ntoy,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                REPLAY.load_predictions(path, "pu")

    def test_missing_headline_is_not_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, "exactly one locked PU group"):
            REPLAY.validate_headlines([], [])


if __name__ == "__main__":
    unittest.main()
