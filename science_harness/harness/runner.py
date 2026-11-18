"""Study orchestration over existing engines; no scientific reasoning loop."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from ..benchmarks import get
from .config import StudyConfig


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    revision, dirty, reason = None, None, "installed package, not a source checkout"
    git = shutil.which("git")
    if (root / "pyproject.toml").is_file():
        reason = "Git unavailable" if git is None else "not a Git checkout"
        if git is not None:
            top = subprocess.run([git, "-C", str(root), "rev-parse", "--show-toplevel"],
                                 capture_output=True, text=True)
            if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
                git = None
                reason = "not a Git checkout root"
        if git is not None:
            result = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                revision, reason = result.stdout.strip(), None
                status = subprocess.run(
                    [git, "-C", str(root), "status", "--porcelain", "--", "science_harness", "pyproject.toml"],
                    capture_output=True, text=True, check=True)
                dirty = bool(status.stdout.strip())
    return {
        "revision": revision, "dirty": dirty, "revision_unavailable_reason": reason,
        "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((root / "science_harness").rglob("*")) if p.suffix in (".py", ".mjs")},
    }


def run_study(config, output, approved_credits=None):
    if not isinstance(config, StudyConfig):
        raise ValueError("Validate a StudyConfig before running")
    cap = Decimal(str(config.value["limits"]["credits"]))
    try:
        approved = Decimal(str(approved_credits)) if approved_credits is not None else None
    except InvalidOperation as exc:
        raise ValueError("Approval credits must be a finite number") from exc
    if approved is not None and (not approved.is_finite() or approved < 0):
        raise ValueError("Approval credits must be finite and nonnegative")
    if cap and approved != cap:
        raise ValueError("Paid studies require --approve-credits equal to the config's NEW study credit cap")
    _, loader = get(config.value["benchmark"])
    adapter = loader()()
    directory = Path(output)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "preparing", "config": config.value,
                "source": source_manifest(), "started_at": datetime.now(timezone.utc).isoformat(),
                "cells": config.cells(), "rows": []}
    write_json(directory / "study.json", manifest)
    try:
        adapter.prepare(config, directory, approved_credits)
        manifest["status"] = "running"
        write_json(directory / "study.json", manifest)
        for cell in manifest["cells"]:
            episode = directory / "episodes" / cell["id"]
            episode.mkdir(parents=True)
            row = adapter.run_episode(cell, episode)
            if not isinstance(row, dict) or not {"status", "final_pass", "requests", "observations", "metrics"} <= row.keys():
                raise ValueError("Adapter returned an incomplete episode result")
            ungraded = row["final_pass"] is None and row.get("evaluation_status") == "failed"
            if (not isinstance(row["status"], str) or
                    (type(row["final_pass"]) is not bool and not ungraded) or
                    not isinstance(row["metrics"], dict)):
                raise ValueError("Adapter status, verdict and metric fields have invalid types")
            if ungraded and (row.get("censored") is not True or row.get("stop_study") is not True
                             or not isinstance(row["metrics"].get("error"), str)
                             or not row["metrics"]["error"]):
                raise ValueError("An unavailable evaluation must preserve its error and stop the study")
            for field in ("requests", "observations"):
                if type(row[field]) is not int or not 0 <= row[field] <= config.value["limits"][field]:
                    raise ValueError(f"Adapter exceeded or misreported its {field} budget")
            manifest["rows"].append({**row, "id": cell["id"], "world": cell["world"],
                                     "seed": cell["seed"], "arm": cell["arm"]["name"]})
            write_json(directory / "study.json", manifest)
            if row.get("stop_study"):
                manifest["status"] = "failed" if ungraded else "stopped"
                if ungraded:
                    manifest["error"] = row["metrics"]["error"]
                break
        if manifest["status"] not in ("stopped", "failed"):
            manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "study.json", manifest)
        adapter.close()
    from ..reporting import report_study
    report_study(directory)
    if manifest["status"] == "failed":
        raise RuntimeError(manifest["error"])
    return {"output": str(directory), "status": manifest["status"], "episodes": len(manifest["rows"])}
