"""Atomic checkpoints with a sha256 sidecar, retention and safetensors export (CLAUDE.md §2.9, doc 16 §6).

Explain-me:
  What: save() writes `ckpt.tmp`, fsyncs, renames, then writes `<name>.sha256`; load() verifies the hash and uses
        torch.load(weights_only=True) (no pickle code execution).  Retention never deletes the last two good
        checkpoints (nor the best one).  EMA weights can be exported as .safetensors for inference/cloud.
  In/Out: dict of tensors/primitives -> file; file -> dict.   Try: corrupt a byte of a checkpoint and call load().
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


class CheckpointError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def save(state: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)                                  # atomic on the same filesystem
    path.with_suffix(path.suffix + ".sha256").write_text(_sha(path) + "\n")
    return path


def load(path: str | Path) -> dict:
    path = Path(path)
    side = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not side.exists():
        raise CheckpointError(f"missing checkpoint or sha256 sidecar: {path}")
    if _sha(path) != side.read_text().strip():
        raise CheckpointError(f"checkpoint hash mismatch (corrupt file?): {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def prune(run_dir: str | Path, keep_last: int = 2) -> list[Path]:
    """Delete old step checkpoints but always keep the newest `keep_last` (never fewer than 2), milestones and best."""
    run_dir = Path(run_dir)
    keep_last = max(2, keep_last)
    ckpts = sorted(run_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    protected = set()
    best = run_dir / "best.json"
    if best.exists():
        protected.add(run_dir / json.loads(best.read_text()).get("checkpoint", ""))
    removed = []
    for p in ckpts[:-keep_last]:
        if p in protected or p.with_suffix(".milestone").exists():
            continue
        p.unlink()
        p.with_suffix(p.suffix + ".sha256").unlink(missing_ok=True)
        removed.append(p)
    return removed


def export_safetensors(tensors: dict[str, torch.Tensor], path: str | Path, metadata: dict | None = None) -> Path:
    """EMA weights for inference (no pickle).  Metadata values must be strings."""
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path),
              metadata={k: str(v) for k, v in (metadata or {}).items()})
    return path
