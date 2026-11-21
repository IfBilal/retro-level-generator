"""Freeze the model and the evaluation plan BEFORE the one-shot test (doc 22 §6.5, ledger E3/E5, CLAUDE.md hard rules 1-2).

Explain-me:
  What: records, in FROZEN.json, exactly which checkpoint (sha256), CFG scale, config hash, physics hash, split LOCK hash and gate
        results will be used; and writes reports/final/eval_plan.json (the pre-registration: prompt sets, seeds, metrics, CI method,
        thresholds) whose own hash is stored in FROZEN.json.  After this file exists nothing may be tuned (selection was on dev only).
  In/Out: --ckpt runs/C-main/step_XXXX.pt --w 5 -> FROZEN.json, reports/final/eval_plan.json.  Also exports the cloud bundle.
  Previous: gate_p4/gate_p5 results.  Next: scripts/eval_test.py (once).
  Try: python scripts/freeze.py --ckpt runs/C-main/step_0042000.pt --w 5 --reason "best dev S_noplay"
"""
import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.playability.physics import Physics
from text2mario.train import checkpoint as ckpt
from text2mario.utils.gates import git_sha


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    if (ROOT / "FROZEN.json").exists():
        sys.exit("FROZEN.json already exists: the freeze is final (delete only with the group's explicit decision, and log why).")
    gates = {}
    for g in ("P1", "P2", "P3", "P4", "P5", "P5b"):
        p = ROOT / "reports/gates" / g / "RESULT.json"
        if p.exists():
            r = json.loads(p.read_text())
            gates[g] = {"pass": r["pass"], "sha256": sha_file(p), "timestamp": r.get("timestamp")}
    s = ckpt.load(a.ckpt)
    cfg = json.loads(s["config_json"])
    plan = {
        "version": 1, "pre_registered_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {"checkpoint": str(a.ckpt), "step": s["step"], "w": a.w, "ddim_steps": 50, "eta": 0.0, "ema": True},
        "prompt_sets": {"P-test": {"file": "data/prompts/p_test.json", "seeds": 3, "seed_base": 8_000_000},
                        "P-rand": {"file": "data/prompts/p_rand.json", "seeds": 3, "seed_base": 9_000_000, "note": "fresh seed base, not used in dev"},
                        "P-casual-test": {"note": "30 held-out casual prompts through the normalizer (product path), 3 seeds per scene"}},
        "metrics": ["C-score (+ delta vs null)", "exact match", "per-concept table", "broken-structure rate (raw)", "floor validity",
                    "novelty / copy rate vs TRAIN windows", "tile-distribution JS", "checker-verified rate of framed windows (diagnostic)",
                    "per-level table (6 held-out levels)"],
        "statistics": "95% percentile bootstrap over held-out LEVELS (clusters), 2000 resamples, seed 0; only 6 test levels -> wide intervals",
        "selection_rule": "dev composite S_noplay = 0.5 C + 0.25 (1 - broken) + 0.25 min(1, novelty/ref); playable term excluded (G-P5b failed)",
        "no_tuning_after_freeze": True, "one_shot": "scripts/eval_test.py refuses a second run; a metric-code bug fix must re-run and document BOTH runs"}
    out = ROOT / "reports/final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_plan.json").write_text(json.dumps(plan, indent=1))
    frozen = {"checkpoint": str(a.ckpt), "checkpoint_sha256": sha_file(Path(a.ckpt)), "step": s["step"], "w": a.w, "reason": a.reason,
              "config_hash": s.get("config_hash") or hashlib.sha256(s["config_json"].encode()).hexdigest(), "schedule": cfg["diffusion"]["schedule"],
              "physics_sha256": Physics.load().sha256, "split_lock_sha256": (ROOT / "data/splits/LOCK").read_text().strip().split()[0],
              "eval_plan_sha256": sha_file(out / "eval_plan.json"), "gates": gates, "git_sha": git_sha(),
              "frozen_utc": plan["pre_registered_utc"]}
    (ROOT / "FROZEN.json").write_text(json.dumps(frozen, indent=1))
    print(json.dumps(frozen, indent=1))


if __name__ == "__main__":
    main()
