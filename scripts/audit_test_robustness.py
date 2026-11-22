"""Are the tests deterministic and independent of each other?  python scripts/audit_test_robustness.py [--quick]

Explain-me:
  What: runs the fast suite under three different hypothesis seeds (a flaky property test would fail under one of them), and the WHOLE suite (fast + slow, real model, headless browser)
        with the test files in reverse alphabetical order (a test that only passes because an earlier test set something up, or that leaves module/CUDA state behind, would fail).
  In/Out: -> reports/audit/test_robustness.json (pytest summary line of every run).  Previous: tests/.  Next: reports/audit/AUDIT_REPORT.md.
  Try: python scripts/audit_test_robustness.py --quick   (seeds only)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pytest(args):
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT / 'tests'}"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    return {"args": " ".join(args)[:140], "returncode": p.returncode, "summary": (p.stdout.strip().splitlines() or [""])[-1], "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    runs = [pytest(["-m", "not slow", f"--hypothesis-seed={s}"]) for s in (1, 2, 3)]
    if not a.quick:
        files = sorted((p.as_posix() for p in (ROOT / "tests").glob("test_*.py")), reverse=True)
        runs.append(pytest(["--hypothesis-seed=0", *[str(Path(f).relative_to(ROOT).as_posix()) if Path(f).is_absolute() else f for f in files], "tests/e2e"]))
        runs[-1]["args"] = "ALL tests (fast + slow), test files in REVERSE order"
    out = {"runs": runs, "all_pass": all(r["returncode"] == 0 for r in runs)}
    (ROOT / "reports/audit/test_robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in runs:
        print(r["returncode"], r["summary"], "|", r["args"])
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
