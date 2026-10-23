"""Check exact distribution bytes without importing models or reading waveforms."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import json
import re


IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}
GENERATED_ROOTS = {"runs", "tmp"}
FORBIDDEN_SUFFIXES = {".mat", ".pt", ".pth", ".npz", ".npy", ".joblib",
                      ".pkl", ".pickle", ".docx", ".pdf", ".7z", ".zip"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distribution_files(root):
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(p in IGNORED_PARTS for p in relative.parts):
            continue
        if relative.parts[0] in GENERATED_ROOTS:
            continue
        if path.is_symlink():
            raise ValueError("Unexpected symbolic link: " + relative.as_posix())
        if path.is_file():
            yield path


def verify(root):
    root = root.resolve()
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError("Missing MANIFEST.sha256")
    names = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if " *" not in line:
            raise ValueError(f"Malformed manifest line {number}")
        expected, relative = line.split(" *", 1)
        pure = PurePosixPath(relative)
        if (not re.fullmatch(r"[0-9a-f]{64}", expected) or not relative or
                pure.is_absolute() or ".." in pure.parts or "\\" in relative or
                relative == "MANIFEST.sha256"):
            raise ValueError(f"Invalid manifest entry at line {number}")
        if relative in names:
            raise ValueError("Duplicate manifest entry: " + relative)
        names.add(relative)
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Missing/unsafe manifest file: " + relative)
        if digest(path) != expected:
            raise ValueError("SHA-256 mismatch: " + relative +
                             ". Preserve exact bytes; use a fresh checkout without line-ending conversion, not rewritten historical hashes.")
    files = list(distribution_files(root))
    actual = {p.relative_to(root).as_posix() for p in files if p != manifest}
    if actual != names:
        raise ValueError("Manifest coverage mismatch: unlisted=" + str(sorted(actual - names)) +
                         "; absent=" + str(sorted(names - actual)))
    bad = [p.relative_to(root).as_posix() for p in files if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    if bad:
        raise ValueError("Non-code/private/heavy artifacts found: " + ", ".join(bad))
    raw_names = {"raw_local_detail.csv", "overview_bin_ranges.csv"}
    if any(p.name in raw_names for p in files):
        raise ValueError("External waveform assets must not be included in this lightweight distribution")
    code = list((root / "code").glob("*.py"))
    names_in_code = {p.stem for p in code}
    graph_file = root / "docs/dependency_graph.json"
    with graph_file.open(encoding="utf-8") as stream:
        graph = json.load(stream)
    missing = []
    for module, dependencies in graph["local_imports"].items():
        for dependency in dependencies:
            if dependency not in names_in_code:
                missing.append(f"{module} -> {dependency}")
    if missing:
        raise ValueError("Missing local modules: " + ", ".join(missing))
    parsed = 0
    patterns = [r"gh[pousr]_[A-Za-z0-9]{25,}", r"sk-[A-Za-z0-9]{25,}",
                r"C:[\\/]Users[\\/](?:Charlotte|dev-admin)"]
    for path in files:
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path))
        parsed += 1
        if any(re.search(pattern, source) for pattern in patterns):
            raise ValueError("Potential credential or personal path: " + path.relative_to(root).as_posix())
    return {"status": "PASS", "manifest_files": len(names), "manifest_coverage_complete": True,
            "code_modules": len(code), "python_sources_parsed": parsed,
            "line_endings": "Exact byte checks; no implicit normalisation",
            "high_confidence_privacy_scan_hits": 0, "forbidden_binary_artifacts": [],
            "imports_executed": False, "models_loaded": False, "target_inference_performed": False,
            "excluded_runtime_locations": [".git", "__pycache__", ".pytest_cache", "runs/", "tmp/"],
            "scope": "Distribution integrity/static checks, not scientific validity, licensing or external-registration certification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo_root), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")
