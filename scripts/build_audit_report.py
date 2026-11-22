"""Render reports/audit/AUDIT_REPORT.md from the template and the saved audit artifacts: python scripts/build_audit_report.py

Explain-me:
  What: every number in the audit report is read from a saved file (audit runs, variety, towers, cloud run, latency, browser evidence, gate results, test
        summary) and substituted into reports/audit/AUDIT_REPORT_template.md, so the report cannot drift from the evidence (CLAUDE.md section 3).
  In/Out: reports/audit/*/results.json, rows.jsonl, variety.json, reports/app/latency.json, reports/gates/*, evidence/audit/<latest>/browser_*.json
          -> reports/audit/AUDIT_REPORT.md.  A missing input leaves "n/a (not run)" in the output and check_provenance.py flags it.
  Previous: scripts/audit_adherence.py, audit_variety.py, audit_browser*.py.  Next: PROGRESS.md / README link to the report.
  Try: python scripts/build_audit_report.py && grep -n "n/a" reports/audit/AUDIT_REPORT.md
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data import tiles as T
from text2mario.eval.count_audit import detector_objects, visible_towers

AUD = ROOT / "reports/audit"


def J(p):
    p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rows_of(run):
    p = AUD / run / "rows.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def cat_table(runs):
    """Markdown table: category x run -> pass/n = rate [CI]."""
    res = {r: J(f"reports/audit/{r}/results.json") for r in runs}
    if any(v is None for v in res.values()):
        return "n/a (not run)"
    cats = sorted(set().union(*[set(v["by_category"]) for v in res.values()]))
    head = ["category"] + list(runs)
    rows = []
    for c in cats:
        cells = []
        for r in runs:
            d = res[r]["by_category"].get(c)
            cells.append("-" if not d else f"{d['pass']}/{d['n']} = {d['rate']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]")
        rows.append([c.replace("_", " ")] + cells)
    rows.append(["**all levels**"] + [f"{res[r]['overall'][0]}/{res[r]['overall'][1]} = {res[r]['overall'][0] / res[r]['overall'][1]:.3f}" for r in runs])
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def misses(run):
    """Prompts of a run whose pass rate is below the target, as a bullet list ('none' if empty)."""
    res = J(f"reports/audit/{run}/results.json")
    if not res:
        return "n/a (not run)"
    target = {"count": 0.9, "example": 1.0, "phrasing": 0.9, "negation": 1.0, "ordering": 1.0, "two_scene_counts": 0.9, "synonym": 0.9, "fresh": 0.9, "adversarial": 1.0}
    bad = [f"- `{k}` ({v['text'][:60]}): {v['pass']}/{v['n']}" for k, v in res["per_prompt"].items() if v["rate"] < target.get(v["cat"], 1.0)]
    single = [f"- `{k}`: {v['pass']}/{v['n']}" for k, v in res["per_prompt"].items() if v["pass"] < v["n"] and v["rate"] >= target.get(v["cat"], 1.0)]
    out = "\n".join(bad) if bad else "none below target"
    if single:
        out += "\n\nMisses that stay within the per-prompt target:\n" + "\n".join(single)
    return out


def tower_stats(run):
    n = det = vis = 0
    for r in rows_of(run):
        if r["id"] in ("cnt_tower_1", "cnt_tower_2", "cnt_tower_3") and r.get("grid"):
            g = np.array([[T.CHARS.index(c) for c in row] for row in r["grid"]], np.uint8)
            want = int(r["id"][-1])
            n += 1
            det += len(detector_objects(g, visible_tower_rule=False).get("tower", [])) == want
            vis += len(visible_towers(g)) == want
    return n, det, vis


def timing(run):
    xs = [r["seconds"] for r in rows_of(run) if r.get("seconds") is not None]
    return (st.median(xs), float(np.quantile(xs, 0.95)), max(xs)) if xs else (float("nan"),) * 3


def browser_summary():
    dirs = sorted((ROOT / "evidence/audit").glob("2*"))
    out = {"states_pass": "n/a (not run)", "states_total": "n/a (not run)", "dice": "n/a (not run)", "defects": "n/a (not run)", "dir": "n/a (not run)"}
    for d in reversed(dirs):
        s, b = d / "browser_states.json", d / "browser_final_local.json"
        if s.exists() and b.exists():
            rs = json.loads(s.read_text(encoding="utf-8"))
            bj = json.loads(b.read_text(encoding="utf-8"))
            dice = next((x for x in bj["steps"] if x["step"] == "dice"), {})
            out.update(states_pass=sum(1 for r in rs if r["pass"]), states_total=len(rs), dice=f"{dice.get('ok')}/{dice.get('clicks')}", defects=len(bj["defects"]), dir=d.name)
            break
    return out


def values():
    v = {}
    base, A, B1, B2, cl = (J(f"reports/audit/{r}/results.json") for r in ("baseline_full", "roundA", "roundB_fresh_seeds", "roundB_fresh_prompts", "cloud_cpu"))
    for key, d in (("base", base), ("A", A), ("B1", B1), ("B2", B2), ("cloud", cl)):
        v[f"{key}_overall"] = f"{d['overall'][0]}/{d['overall'][1]}" if d else "n/a (not run)"
        v[f"{key}_pct"] = f"{100 * d['overall'][0] / d['overall'][1]:.1f}" if d else "n/a (not run)"
        v[f"{key}_git"] = d["meta"]["git"] if d else "n/a (not run)"
        v[f"{key}_utc"] = d["meta"]["utc"] if d else "n/a (not run)"
    v["table_base_A"] = cat_table(["baseline_full", "roundA"])
    v["table_B1"] = cat_table(["roundA", "roundB_fresh_seeds"])
    v["table_B2"] = cat_table(["roundB_fresh_prompts"])
    v["misses_A"], v["misses_B1"], v["misses_B2"] = misses("roundA"), misses("roundB_fresh_seeds"), misses("roundB_fresh_prompts")
    var = J("reports/audit/variety.json")
    if var:
        s = var["summary"]
        v.update(var_distinct_b=s["before"]["distinct"], var_distinct_a=s["after"]["distinct"], var_ham_b=s["before"]["mean_pairwise_hamming"], var_ham_a=s["after"]["mean_pairwise_hamming"],
                 var_plain_b=s["before"]["plain_floor_or_sky_share"], var_plain_a=s["after"]["plain_floor_or_sky_share"], var_min=s["min_distinct_after"])
    for tag, run in (("base", "baseline_full"), ("r5", "round5"), ("A", "roundA")):
        n, det, vis = tower_stats(run)
        v[f"tw_{tag}_n"], v[f"tw_{tag}_det"], v[f"tw_{tag}_vis"] = n, det, vis
    rows = rows_of("roundA")
    v["A_counter_levels"] = sum(1 for r in rows if r.get("independent"))
    v["A_counter_disagree"] = sum(1 for r in rows if r.get("counter_disagreement"))
    v["A_tower_disagree"] = sum(1 for r in rows if r.get("counts") and r["counts"].get("tower", 0) != r["counts"].get("tower_detector", 0))
    v["A_verified"] = sum(1 for r in rows if r.get("verification") == "verified")
    v["A_levels"] = sum(1 for r in rows if r.get("verification"))
    bro = rows_of("baseline_full")
    v["base_verified"] = sum(1 for r in bro if r.get("verification") == "verified")
    v["base_levels"] = sum(1 for r in bro if r.get("verification"))
    for tag, run in (("base", "baseline_full"), ("A", "roundA"), ("B1", "roundB_fresh_seeds")):
        p50, p95, mx = timing(run)
        v[f"{tag}_t50"], v[f"{tag}_t95"], v[f"{tag}_tmax"] = f"{p50:.1f}", f"{p95:.1f}", f"{mx:.1f}"
    crow = rows_of("cloud_cpu")
    if crow:
        v["cloud_t50"] = f"{st.median(r['seconds'] for r in crow):.1f}"
        v["cloud_tmax"] = f"{max(r['seconds'] for r in crow):.1f}"
        v["cloud_budget_hits"] = sum(1 for r in crow if r.get("reason") == "time budget")
        v["cloud_ex3"] = next((f"{d['pass']}/{d['n']}" for k, d in (cl or {}).get("per_prompt", {}).items() if k == "ex_button_3"), "n/a")
    lat = J("reports/app/latency.json") or {}
    for k in ("cuda_48_fast_p50", "cuda_48_fast_p95", "cuda_112_fast_p50", "cuda_112_fast_p95", "cpu_48_p50", "cpu_112_p50", "cold_start_s"):
        v[f"lat_{k}"] = lat.get(k, "n/a (not run)")
    we = J("reports/audit/window_exactness.json")
    if we:
        e = we["exact"]
        for key, cap in (("pipe1", "one pipe."), ("pipe2", "two pipes."), ("enemy1", "one enemy."), ("platform1", "one platform."), ("cannon1", "one cannon.")):
            v[f"we_{key}"] = f"{e[cap]['exact']}/{we['n']}" if cap in e else "n/a"
        v["we_enemy3"] = f"{e['full floor. a few enemies.']['exact']}/{we['n']}"
        kh = we["known_half"]
        v["we_known_one"], v["we_known_none"] = json.dumps(kh["one pipe."]["new_half_pipes"]), json.dumps(kh["full floor."]["new_half_pipes"])
    gates = {}
    for g in ("P1", "P2", "P3", "P4", "P5", "P5b", "P6", "P7", "P8"):
        r = J(f"reports/gates/{g}/RESULT.json")
        gates[g] = "PASS" if r and r["pass"] else ("FAIL" if r else "not run")
    v["gates"] = ", ".join(f"G-{k} {x}" for k, x in gates.items())
    b = browser_summary()
    v.update(br_states=f"{b['states_pass']}/{b['states_total']}", br_dice=b["dice"], br_defects=b["defects"], br_dir=b["dir"])
    ts = J("reports/testing/pytest_summary.json") or {}
    v["tests_total"] = ts.get("passed", "n/a (not run)")
    import subprocess
    log = subprocess.run(["git", "log", "--reverse", "--format=%h %s", "main..HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip().splitlines()
    v["commits"] = "\n".join(f"* `{x.split(' ', 1)[0]}` {x.split(' ', 1)[1][:160]}" for x in log) or "n/a (no commits ahead of main)"
    dep = J("reports/gates/P8/RESULT.json") or {}
    v["deployed_pass"] = "PASS" if dep.get("pass") else "n/a"
    return v


def main():
    vals = values()
    tpl = (AUD / "AUDIT_REPORT_template.md").read_text(encoding="utf-8")
    missing = set()

    def sub(m):
        k = m.group(1)
        if k not in vals:
            missing.add(k)
            return "n/a (not run)"
        return str(vals[k])

    out = re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", sub, tpl)
    (AUD / "AUDIT_REPORT.md").write_text(out, encoding="utf-8")
    print("wrote reports/audit/AUDIT_REPORT.md;", "placeholders without a value: " + ", ".join(sorted(missing)) if missing else "all placeholders resolved")


if __name__ == "__main__":
    main()
