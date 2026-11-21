"""Training entry point: python -m text2mario.train.train --config configs/<overlay>.yaml (doc 16 §1).

Explain-me:
  What: loads base + overlay + machine config, checks that the data gates passed, applies the VRAM cap, creates the run
        directory (config snapshot, git SHA, machine, hashes), builds the Trainer and runs it.
  Safety: refuses to start unless gates P1 and P2 passed and data/splits/LOCK verifies; machine YAMLs cannot change
          hyperparameters; every fallback is logged to events.jsonl.
  Try: --dry-run builds everything and runs 2 steps without writing checkpoints (used by tests).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from text2mario.utils.env import bootstrap_env

bootstrap_env()

from text2mario.data import provenance, splits
from text2mario.train.loop import Trainer
from text2mario.utils.config import ROOT, config_hash, load_config
from text2mario.utils.gates import git_sha, require
from text2mario.utils.hw import apply_vram_cap, detect_machine, hw_report


def make_run_dir(cfg: dict, machine: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run = ROOT / "runs" / f"{cfg['name']}_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip())
    meta = {"config_hash": config_hash(cfg), "git_sha": git_sha(), "git_dirty": dirty, "machine": machine,
            "torch": torch.__version__, "seed": cfg["seed"], "corpus_hash": provenance.corpus_content_hash(),
            "split_lock": (ROOT / "data/splits/LOCK").read_text().strip(), "created_utc": stamp}
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    machine = a.machine or detect_machine()
    cfg = load_config(a.config, machine if (ROOT / "configs/machine" / f"{machine}.yaml").exists() else None, a.overrides)
    require("P1")                                      # the next phase checks the previous gate (CLAUDE.md §3)
    require("P2")
    splits.load(verify=True)                           # LOCK must verify
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        apply_vram_cap(cfg["train"]["vram_cap_gib"])
        torch.backends.cudnn.benchmark = bool(cfg["cudnn_benchmark"])
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    run = Path(a.run_dir) if a.run_dir else make_run_dir(cfg, machine)
    tr = Trainer(cfg, run, device)
    if a.resume:
        tr.resume(a.resume)
    if a.dry_run:
        for _ in range(2):
            print(tr.train_step())
        return
    try:
        tr.run()
    finally:
        hw_report(run / "hw_report.json", {"steps": tr.step})


if __name__ == "__main__":
    main(sys.argv[1:])
