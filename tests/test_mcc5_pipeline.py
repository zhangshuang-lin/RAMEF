"""Standard-library launcher tests: no ML, inference, or waveform loading."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


_PATH = Path(__file__).with_name("mcc5_pipeline.py")
if not _PATH.exists():
    _PATH = Path(__file__).resolve().parents[1] / "scripts" / "mcc5_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("mcc5_public_pipeline_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
pipeline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pipeline
_SPEC.loader.exec_module(pipeline)


class MCC5PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ramef_mcc5_launcher_")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.output = patch("builtins.print")
        self.output.start()
        self.addCleanup(self.output.stop)
        self.root = self.base / "repo"
        for relative in (
            "code/mcc5_confirmatory_data.py", "code/run_mcc5_ramef_confirmatory.py",
            "code/u2_stft_model.py", "protocols/mcc5/protocol_run_manifest.csv",
            "protocols/mcc5/P1/speed_keyphase_audit.csv",
            "protocols/mcc5/MCC5_RAMEF_CONFIRMATORY_PROTOCOL.md",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        self.data = self.base / "official_data"
        self.data.mkdir()
        self.work = self.root / "runs" / "fresh"

    def ready(self):
        self.work.mkdir(parents=True)
        (self.work / "development.npz").write_bytes(b"not-a-cache; never read")
        (self.work / "run").mkdir()
        (self.work / "run/selection_complete.json").write_text("{}", encoding="utf-8")

    def test_prepare_command_and_no_side_effect_dry_run(self):
        plan = pipeline.build_plan("prepare", self.root, self.work, self.data)
        self.assertEqual(plan.command[0], sys.executable)
        self.assertEqual(plan.command[1], str(self.root / "code/mcc5_confirmatory_data.py"))
        self.assertEqual(plan.command[plan.command.index("--root") + 1], str(self.data))
        self.assertEqual(plan.command[plan.command.index("--output") + 1], str(self.work / "development.npz"))
        self.assertNotIn("--only-query", plan.command)
        with patch.object(pipeline.subprocess, "run") as run:
            self.assertEqual(pipeline.execute_plan(plan, dry_run=True), 0)
            run.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_existing_release_outside_repository_is_protected(self):
        with self.assertRaisesRegex(ValueError, "release"):
            pipeline.build_plan("prepare", self.root, self.base / "release/old/new", self.data)

    def test_repository_with_release_ancestor_allows_its_runs(self):
        nested_root = self.base / "release" / "public_repository"
        work = nested_root / "runs" / "fresh"
        self.assertEqual(pipeline.validate_work_dir(nested_root, work), work.resolve())
        self.assertFalse(work.exists())

    def test_existing_development_and_training_outputs_are_not_overwritten(self):
        self.ready()
        with self.assertRaisesRegex(ValueError, "already exists"):
            pipeline.build_plan("prepare", self.root, self.work, self.data)
        with self.assertRaisesRegex(ValueError, "not empty"):
            pipeline.build_plan("train", self.root, self.work)

    def test_train_explicitly_overrides_paths_and_subprocess_is_mocked(self):
        self.work.mkdir(parents=True)
        (self.work / "development.npz").write_bytes(b"never read")
        plan = pipeline.build_plan("train", self.root, self.work)
        for option in ("--data-root", "--development-cache", "--query-cache", "--manifest",
                       "--keyphase-audit", "--protocol", "--output"):
            self.assertIn(option, plan.command)
        with patch.object(pipeline.subprocess, "run") as run:
            pipeline.execute_plan(plan)
            run.assert_called_once()
            self.assertIs(run.call_args.kwargs["check"], True)
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"].split(pipeline.os.pathsep)[0],
                             str(self.root / "code"))

    def test_evaluate_dry_run_creates_neither_lock_nor_query(self):
        self.ready()
        plan = pipeline.build_plan("evaluate", self.root, self.work, self.data)
        with patch.object(pipeline.subprocess, "run") as run:
            pipeline.execute_plan(plan, dry_run=True)
            run.assert_not_called()
        self.assertFalse((self.work / "run/.public_evaluation_started.lock").exists())
        self.assertFalse((self.work / "query_after_selection.npz").exists())

    def test_evaluation_guard_is_permanent_when_subprocess_fails(self):
        self.ready()
        plan = pipeline.build_plan("evaluate", self.root, self.work, self.data)
        with patch.object(pipeline.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["mock"])):
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.execute_plan(plan)
        self.assertTrue((self.work / "run/.public_evaluation_started.lock").is_file())
        with self.assertRaisesRegex(ValueError, "already started"):
            pipeline.build_plan("evaluate", self.root, self.work, self.data)

    def test_original_target_lock_prevents_execution(self):
        self.ready()
        (self.work / "run/target_evaluation.lock").write_text("original", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "original target evaluation lock"):
            pipeline.build_plan("evaluate", self.root, self.work, self.data)

    def test_root_and_raw_dataset_cannot_be_work_directories(self):
        with self.assertRaisesRegex(ValueError, "repository root"):
            pipeline.build_plan("prepare", self.root, self.root, self.data)
        with self.assertRaisesRegex(ValueError, "raw dataset"):
            pipeline.build_plan("prepare", self.root, self.data / "outputs", self.data)


def _data_root_test(stage):
    def test(self):
        with self.assertRaisesRegex(ValueError, "--data-root"):
            pipeline.build_plan(stage, self.root, self.work)
    return test


def _protected_path_test(protected):
    def test(self):
        with self.assertRaisesRegex(ValueError, "protected|release"):
            pipeline.build_plan("prepare", self.root, self.root / protected / "new", self.data)
    return test


# Separate methods retain the original 22 cases plus the release-ancestor fix.
for _stage in ("prepare", "evaluate"):
    setattr(MCC5PipelineTests, f"test_{_stage}_requires_explicit_data_root", _data_root_test(_stage))
for _protected in (
    "code", "evidence", "protocols", "reference", "configs", "archive",
    "figures", "docs", "release", "frozen", "FROZEN_ARCHIVE", "outputs/staged",
):
    setattr(MCC5PipelineTests, "test_protected_" + _protected.replace("/", "_"),
            _protected_path_test(_protected))


if __name__ == "__main__":
    unittest.main()
