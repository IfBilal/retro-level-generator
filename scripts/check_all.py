"""One command to audit the whole repository: python scripts/check_all.py [--slow]

Explain-me:
  What: runs the fast test suite (and, with --slow, the end-to-end browser tests), prints every gate's recorded status, verifies the split LOCK and the freeze
        hashes, runs the provenance check and reports whether the working tree is clean.  Known failed gates are listed, not hidden; the exit code is non-zero only
        when a test or an integrity check fails (a failed gate is a recorded result, not a broken build).
  In/Out: -> printed summary, exit code.  Previous: all scripts.  Try: python scripts/check_all.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def main() -> int:
    slow = "--slow" in sys.argv
    bad = 0
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    t = sh([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"], env=env)
    line = [x for x in t.stdout.strip().splitlines() if "passed" in x or "failed" in x][-1:]
    print("fast tests :", line[0] if line else t.stdout[-200:])
    bad += t.returncode != 0
    if slow:
        t2 = sh([sys.executable, "-m", "pytest", "-q", "-m", "slow", "-p", "no:cacheprovider"], env=env)
        print("slow tests :", [x for x in t2.stdout.strip().splitlines() if "passed" in x or "failed" in x][-1:])
        bad += t2.returncode != 0
    print("gates      :")
    for g in ("P1", "P2", "P3", "P4", "P5", "P5b", "P6", "P7", "P8"):
        p = ROOT / "reports/gates" / g / "RESULT.json"
        if p.exists():
            r = json.loads(p.read_text())
            print(f"   G-{g:4s} {'PASS' if r['pass'] else 'FAIL (recorded)'}  {r.get('timestamp', '')}")
        else:
            print(f"   G-{g:4s} not run" + (" (needs your deployed URL: scripts/gate_p8.py)" if g == "P8" else ""))
    try:
        from text2mario.data import splits
        splits.load(verify=True)
        print("split LOCK : verifies")
    except Exception as e:
        print("split LOCK : FAILED", e)
        bad += 1
    fz = ROOT / "FROZEN.json"
    if fz.exists():
        import hashlib
        j = json.loads(fz.read_text())
        ok = hashlib.sha256((ROOT / "reports/final/eval_plan.json").read_bytes()).hexdigest() == j["eval_plan_sha256"]
        print("freeze     :", "eval plan hash matches" if ok else "EVAL PLAN CHANGED AFTER FREEZE")
        bad += not ok
    pr = sh([sys.executable, "scripts/check_provenance.py"], env=env)
    print("provenance :", "all checks pass" if pr.returncode == 0 else "FAILED\n" + pr.stdout)
    bad += pr.returncode != 0
    st = sh(["git", "status", "--porcelain"]).stdout.strip()
    print("git        :", "working tree clean" if not st else f"{len(st.splitlines())} uncommitted change(s)")
    print("RESULT     :", "OK" if not bad else f"{bad} integrity problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
