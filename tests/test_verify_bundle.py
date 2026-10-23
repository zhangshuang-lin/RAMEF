"""Synthetic packaging fixtures; never paper evidence or model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ENTRY = Path(__file__).resolve().parents[1] / "scripts/verify_bundle.py"
SPEC = importlib.util.spec_from_file_location("verify_bundle_under_test", ENTRY)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ramef_bundle_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "code").mkdir()
        (self.root / "code/example.py").write_bytes(b"value = 1\r\n")
        (self.root / "docs").mkdir()
        (self.root / "docs/dependency_graph.json").write_text(json.dumps({"local_imports": {}}), encoding="utf-8")
        (self.root / ".gitattributes").write_text("* -text\n", encoding="utf-8")
        self.make_manifest()

    def make_manifest(self):
        files = sorted(p for p in self.root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
        text = "\n".join(hashlib.sha256(p.read_bytes()).hexdigest() + " *" + p.relative_to(self.root).as_posix() for p in files)
        (self.root / "MANIFEST.sha256").write_text(text + "\n", encoding="utf-8")

    def test_exact_mixed_line_endings_pass(self):
        report = VERIFY.verify(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["manifest_coverage_complete"])
        self.assertFalse(report["models_loaded"])

    def test_changed_line_endings_are_not_silently_accepted(self):
        (self.root / "code/example.py").write_bytes(b"value = 1\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            VERIFY.verify(self.root)

    def test_unlisted_file_is_rejected(self):
        (self.root / "unexpected.txt").write_text("fixture", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            VERIFY.verify(self.root)

    def test_missing_file_is_rejected(self):
        (self.root / "code/example.py").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/unsafe"):
            VERIFY.verify(self.root)

    def test_duplicate_manifest_record_is_rejected(self):
        path = self.root / "MANIFEST.sha256"
        data = path.read_text(encoding="utf-8")
        path.write_text(data + data.splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate manifest"):
            VERIFY.verify(self.root)

    def test_parent_escape_is_rejected(self):
        (self.root / "MANIFEST.sha256").write_text("0" * 64 + " *../outside.txt\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid manifest"):
            VERIFY.verify(self.root)

    def test_forbidden_model_file_is_rejected_even_if_listed(self):
        (self.root / "fixture.pt").write_bytes(b"synthetic packaging fixture, not a model")
        self.make_manifest()
        with self.assertRaisesRegex(ValueError, "Non-code/private/heavy"):
            VERIFY.verify(self.root)

    def test_generated_runs_do_not_change_distribution(self):
        (self.root / "runs").mkdir()
        (self.root / "runs/generated.csv").write_text("fixture\n", encoding="utf-8")
        self.assertEqual(VERIFY.verify(self.root)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
