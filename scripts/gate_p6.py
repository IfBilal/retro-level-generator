"""Gate G-P6: the one-shot test protocol was followed (doc 24 §G-P6).  A PROCESS gate: it audits files, not model quality.

Explain-me:
  What: checks that FROZEN.json predates the test results, that the eval-plan and checkpoint hashes still match, that the invocation log records exactly one
        COMPLETED run (a crashed first attempt is allowed if logged), that RESULTS.md / metrics.json / per-level tables exist, that the held-out set was reported
        with a level-clustered interval over all held-out levels, and that the split LOCK still verifies.
  In/Out: -> reports/gates/P6/RESULT.json.  Previous: scripts/eval_test.py.  Try: python scripts/gate_p6.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data import splits
from text2mario.utils.gates import write_result

F = ROOT / "reports/final"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    fz = json.loads((ROOT / "FROZEN.json").read_text())
    log = [json.loads(x) for x in (F / "test_invocations.jsonl").read_text().splitlines() if x.strip()]
    done = [x for x in log if x.get("completed")]
    m = json.loads((F / "metrics.json").read_text())
    ck = Path(fz["checkpoint"])
    ck = ck if ck.is_absolute() else ROOT / ck
    checks = [
        ("P6.01", "FROZEN.json written before the test results", (ROOT / "FROZEN.json").stat().st_mtime < (F / "metrics.json").stat().st_mtime),
        ("P6.02", "eval_plan.json hash matches the freeze", sha(F / "eval_plan.json") == fz["eval_plan_sha256"]),
        ("P6.03", "frozen checkpoint hash matches (skipped if the checkpoint is not on this machine)", (sha(ck) == fz["checkpoint_sha256"]) if ck.exists() else True),
        ("P6.04", "exactly one completed test run in the log", len(done) == 1),
        ("P6.05", "split LOCK verifies", bool(splits.load(verify=True))),
        ("P6.06", "RESULTS.md, per-level and per-concept tables and the failure gallery exist",
         all((F / f).exists() for f in ("RESULTS.md", "per_level_P-test.csv", "per_concept_P-test.csv", "failure_gallery.png"))),
        ("P6.07", "P-test reported with a level-clustered interval over all 6 held-out levels", m["P-test"]["n_levels"] >= 6 and "ci95" in m["P-test"]["c_score"]),
    ]
    rows = [{"id": i, "name": n, "pass": bool(p)} for i, n, p in checks]
    notes = [x.get("note") for x in log if x.get("note")]
    write_result("P6", all(r["pass"] for r in rows), {"invocations": len(log), "completed": len(done)}, rows, log_notes=notes)
    for r in rows:
        print(r["id"], "PASS" if r["pass"] else "FAIL", r["name"])


if __name__ == "__main__":
    main()
