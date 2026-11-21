"""Collect the evidence-of-evaluation folder (doc 35 §3): python scripts/collect_evidence.py [--skip-tests]

Explain-me:
  What: runs the fast test suite (writes reports/testing/pytest_summary.json and the full log), then copies every gate RESULT.json,
        the final test results and one-shot log, the dev evaluations, the figures, the freeze record and the latency numbers into
        evidence/, with a MANIFEST.json of sha256 hashes.  Nothing is edited on the way: the folder is a snapshot of saved artifacts.
  In/Out: reports/** + FROZEN.json + report/figures -> evidence/**.  Previous: all gates.  Next: scripts/build_submission_zip.py.
  Try: python scripts/collect_evidence.py --skip-tests  (reuses the last pytest summary)
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"


def run_tests():
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, env=env)
    slow = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "slow", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, env=env)
    last = [ln for ln in r.stdout.strip().splitlines() if "passed" in ln or "failed" in ln][-1]
    m = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|deselected|error)", last)}
    last_s = [ln for ln in slow.stdout.strip().splitlines() if "passed" in ln or "failed" in ln][-1:]
    out = {"passed": m.get("passed", 0), "failed": m.get("failed", 0), "total": m.get("passed", 0) + m.get("failed", 0),
           "summary_line": last, "slow_suite": (last_s[0] if last_s else "not run")}
    (ROOT / "reports/testing").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/testing/pytest_summary.json").write_text(json.dumps(out, indent=1))
    (ROOT / "reports/testing/pytest_fast.log").write_text(r.stdout + r.stderr)
    (ROOT / "reports/testing/pytest_slow.log").write_text(slow.stdout + slow.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    summ = None if a.skip_tests else run_tests()
    if EV.exists():
        shutil.rmtree(EV)
    EV.mkdir()
    plan = {"gates": [p for p in (ROOT / "reports/gates").glob("*/RESULT.json")], "metrics": [ROOT / "reports/final/RESULTS.md", ROOT / "reports/final/metrics.json"],
            "test_run": [ROOT / "reports/final/eval_plan.json", ROOT / "reports/final/test_invocations.jsonl", ROOT / "FROZEN.json"],
            "testing": [ROOT / "reports/testing/pytest_summary.json", ROOT / "reports/testing/pytest_fast.log", ROOT / "reports/testing/pytest_slow.log"],
            "app": [ROOT / "reports/app/latency.json"], "normalizer": [ROOT / "reports/normalizer/dev.json"], "eda": [ROOT / "reports/eda/AUDIT.json"]}
    for sub, files in plan.items():
        for f in files:
            if f.exists():
                d = EV / sub / (f.parent.name if sub == "gates" else "")
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, d / f.name)
    for pat, sub in (("reports/final/*.csv", "metrics"), ("reports/final/*.png", "figures"), ("report/figures/*.png", "figures"), ("reports/dev_eval/*.json", "dev_eval"),
                     ("reports/gates/P5b/*.json", "gates/P5b_details"), ("reports/gates/P3_*/RESULT.json", "gates/P3_variants")):
        for f in ROOT.glob(pat):
            d = EV / sub
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, d / (f.parent.name + "_" + f.name if "P3_" in pat else f.name))
    man = {str(p.relative_to(EV)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(EV.rglob("*")) if p.is_file()}
    (EV / "MANIFEST.json").write_text(json.dumps({"files": man, "pytest": summ}, indent=1))
    print(f"evidence/: {len(man)} files", summ or "")


if __name__ == "__main__":
    main()
