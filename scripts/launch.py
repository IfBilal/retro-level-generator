"""Launch a long training run with provenance (CLAUDE.md §3): python scripts/launch.py --config configs/cond_main.yaml --run-dir runs/NAME [--resume CKPT] [--set key=value ...]

Explain-me:
  What: creates the run directory, writes runs/NAME/launch.json (git SHA, dirty flag, command line, config snapshot, machine profile, Python/torch versions, start time),
        then starts the trainer DETACHED with its output in runs/NAME.log, so a closed terminal does not kill it.  Update HANDOFF.md yourself when you start or stop a run.
  In/Out: -> runs/NAME/launch.json, runs/NAME.log, a detached process.  Stop safely with:  touch runs/NAME/STOP
  Try: python scripts/launch.py --config configs/uncond_linear.yaml --run-dir runs/demo --set train.max_steps=50 --foreground
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--foreground", action="store_true")
    a = ap.parse_args()
    run = ROOT / a.run_dir
    run.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "text2mario.train.train", "--config", a.config, "--run-dir", a.run_dir]
    if a.resume:
        cmd += ["--resume", a.resume]
    for s in a.set:
        cmd += ["--set", s]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    meta = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "command": cmd[2:], "config_file": a.config, "config_snapshot": (ROOT / a.config).read_text(encoding="utf-8"), "python": sys.version.split()[0]}
    try:
        import torch
        meta.update({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception:
        pass
    (run / "launch.json").write_text(json.dumps(meta, indent=1))
    if a.foreground:
        sys.exit(subprocess.run(cmd, cwd=ROOT, env=env).returncode)
    log = open(ROOT / f"{a.run_dir}.log", "ab")
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=(os.name != "nt"))
    print(f"launched pid {p.pid}; log {a.run_dir}.log; stop safely with: touch {a.run_dir}/STOP")


if __name__ == "__main__":
    main()
