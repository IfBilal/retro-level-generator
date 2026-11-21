"""Gate bookkeeping (CLAUDE.md §3): every phase ends in reports/gates/PN/RESULT.json and the next phase checks it.

Explain-me:
  What: write_result() stores {pass, metrics, timestamp, git SHA}; require() raises unless a gate passed.
  In/Out: scripts/gate_pN.py call write_result; later scripts call require("P1") before they run.
"""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class GateNotPassedError(RuntimeError):
    pass


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):         # no git or not a repository: the provenance field says so
        return "unknown"


def gate_dir(phase: str) -> Path:
    d = ROOT / "reports" / "gates" / phase
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_result(phase: str, passed: bool, metrics: dict, checks: list[dict] | None = None, *, out_dir: Path | None = None,
                 **extra) -> Path:
    """RESULT.json (doc 24 §0.2).  `extra` carries diagnostics / warnings / artifacts / inputs."""
    out = (Path(out_dir) if out_dir else gate_dir(phase)) / "RESULT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"phase": phase, "pass": bool(passed), "metrics": metrics, "checks": checks or [],
                               "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": git_sha(),
                               **extra}, indent=2, default=str) + "\n")
    return out


def require(phase: str) -> dict:
    """Raise unless reports/gates/<phase>/RESULT.json exists with pass: true."""
    p = ROOT / "reports" / "gates" / phase / "RESULT.json"
    if not p.exists() or not json.loads(p.read_text()).get("pass"):
        raise GateNotPassedError(f"gate {phase} has not passed (run scripts/gate_{phase.lower()}.py)")
    return json.loads(p.read_text())
