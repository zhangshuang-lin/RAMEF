"""Portable launcher for the archived, full-63D MCC5 RAMEF workflow.

Install this file as ``scripts/mcc5_pipeline.py`` in the public repository.
Only the original runner performs training or inference.  This launcher makes
all paths explicit and adds a persistent, exclusive public evaluation guard.
Use a fresh work directory; archived results must never be an output target.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class Plan:
    stage: str
    repo_root: Path
    work_dir: Path
    data_root: Path | None
    command: tuple[str, ...]


def _inside(path: Path, parent: Path) -> bool:
    """Case-insensitive containment also protects Windows paths in tests."""
    child = tuple(part.casefold() for part in path.parts)
    base = tuple(part.casefold() for part in parent.parts)
    return len(child) >= len(base) and child[:len(base)] == base


def validate_work_dir(repo_root: Path, work_dir: Path) -> Path:
    root, work = repo_root.resolve(), work_dir.expanduser().resolve()
    if work == root:
        raise ValueError("Use a dedicated work directory, not the repository root.")
    protected = (
        "code", "evidence", "protocols", "reference", "configs", "archive",
        "figures", "docs", "release", "frozen", "FROZEN_ARCHIVE", "outputs/staged",
    )
    if any(_inside(work, root / name) for name in protected):
        raise ValueError("Work directory is inside a protected source or archive directory.")
    if not _inside(work, root) and any(part.casefold() == "release" for part in work.parts):
        raise ValueError("A release archive must not be used as a work directory.")
    if work.exists() and not work.is_dir():
        raise ValueError("Work directory points to an existing file.")
    return work


def _validate_stage(stage: str, work: Path) -> None:
    development, run = work / "development.npz", work / "run"
    if stage == "prepare":
        if development.exists():
            raise ValueError("Development cache already exists; use a fresh work directory.")
    elif stage == "train":
        if not development.is_file():
            raise ValueError("Prepare the development cache before training.")
        if run.exists() and (not run.is_dir() or any(run.iterdir())):
            raise ValueError("Training output is not empty; existing runs are never overwritten.")
    elif stage == "evaluate":
        if not development.is_file():
            raise ValueError("Development cache is missing.")
        if not (run / "selection_complete.json").is_file():
            raise ValueError("Source-selected checkpoints have not been frozen.")
        if (run / "target_evaluation.lock").exists():
            raise ValueError("The original target evaluation lock already exists.")
        if (run / ".public_evaluation_started.lock").exists():
            raise ValueError("A public evaluation was already started; its guard is permanent.")
    else:
        raise ValueError(f"Unknown stage: {stage}")


def build_plan(stage: str, repo_root: Path, work_dir: Path,
               data_root: Path | None = None) -> Plan:
    root = repo_root.expanduser().resolve()
    work = validate_work_dir(root, work_dir)
    data = data_root.expanduser().resolve() if data_root is not None else None
    if stage in {"prepare", "evaluate"} and data is None:
        raise ValueError(f"--data-root is required for {stage}.")
    if data is not None and not data.is_dir():
        raise ValueError("Data root must be an existing official MCC5 data directory.")
    if data is not None and _inside(work, data):
        raise ValueError("Keep generated outputs outside the raw dataset directory.")
    code = root / "code"
    protocol = root / "protocols" / "mcc5"
    files = (
        code / "mcc5_confirmatory_data.py",
        code / "run_mcc5_ramef_confirmatory.py",
        code / "u2_stft_model.py",
        protocol / "protocol_run_manifest.csv",
        protocol / "P1" / "speed_keyphase_audit.csv",
        protocol / "MCC5_RAMEF_CONFIRMATORY_PROTOCOL.md",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError("Missing required package files: " + ", ".join(missing))
    _validate_stage(stage, work)
    manifest = protocol / "protocol_run_manifest.csv"
    keyphase = protocol / "P1" / "speed_keyphase_audit.csv"
    development = work / "development.npz"
    if stage == "prepare":
        command = (
            sys.executable, str(code / "mcc5_confirmatory_data.py"),
            "--root", str(data), "--manifest", str(manifest),
            "--keyphase-audit", str(keyphase), "--output", str(development),
        )
    else:
        # train_stage never reads data_root.  The explicit work-dir value when
        # omitted prevents inheriting the historical machine-specific default.
        command = (
            sys.executable, str(code / "run_mcc5_ramef_confirmatory.py"),
            "--stage", stage, "--data-root", str(data if data is not None else work),
            "--development-cache", str(development),
            "--query-cache", str(work / "query_after_selection.npz"),
            "--manifest", str(manifest), "--keyphase-audit", str(keyphase),
            "--protocol", str(protocol / "MCC5_RAMEF_CONFIRMATORY_PROTOCOL.md"),
            "--output", str(work / "run"),
        )
    return Plan(stage, root, work, data, command)


def execute_plan(plan: Plan, dry_run: bool = False) -> int:
    # Validation and dry-run never import ML libraries or load either cache.
    validate_work_dir(plan.repo_root, plan.work_dir)
    _validate_stage(plan.stage, plan.work_dir)
    print(json.dumps({"stage": plan.stage, "command": list(plan.command),
                      "cwd": str(plan.repo_root),
                      "PYTHONPATH": str(plan.repo_root / "code"),
                      "dry_run": dry_run}, indent=2), flush=True)
    if dry_run:
        return 0
    plan.work_dir.mkdir(parents=True, exist_ok=True)
    if plan.stage == "evaluate":
        guard = plan.work_dir / "run" / ".public_evaluation_started.lock"
        # Never delete this guard, including on subprocess/import/data failures.
        with guard.open("x", encoding="utf-8") as stream:
            json.dump({"stage": "evaluate", "started_utc": datetime.now(timezone.utc).isoformat(),
                       "command": list(plan.command), "retry_allowed": False}, stream, indent=2)
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(plan.repo_root / "code") + (os.pathsep + previous if previous else "")
    subprocess.run(list(plan.command), cwd=str(plan.repo_root), env=env, check=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--data-root", type=Path,
                        help="Parent containing MCC5-THU Motor_speed_circulation; required for prepare/evaluate.")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Dedicated fresh output directory, for example <repo>/runs/mcc5_reproduction.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print validated command only; create no files and load no ML libraries or query.")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.stage, Path(__file__).resolve().parents[1],
                          args.work_dir, args.data_root)
        return execute_plan(plan, args.dry_run)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
