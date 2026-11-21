"""Build report/report.pdf (and report.html) from report/report_template.md + saved result files.

Explain-me:
  What: every {{key}} in the template is filled from the JSON artifacts (gates, dev evals, final test metrics, checker calibration),
        so no number in the report is typed by hand.  Missing artifacts print as "n/a (not run)".  Markdown -> HTML (python-markdown)
        -> PDF through headless Chromium (Playwright).  Figures come from report/figures (scripts/make_figures.py).
  In/Out: report/report_template.md + reports/** -> report/report.md, report.html, report.pdf.
  Try: python scripts/build_report.py --no-pdf   (fast check of the placeholders)
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"


def J(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=3):
    if v is None:
        return "n/a (not run)"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def ci(d, nd=3, clusters=None):
    """mean [95% CI].  With a single cluster (P-rand has no source levels) an interval is meaningless: show the point estimate only."""
    if not d:
        return "n/a (not run)"
    if clusters is not None and clusters <= 1:
        return f"{d['mean']:.{nd}f} (no level clusters, no interval)"
    return f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"


NL = chr(10)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return NL.join(out)


def p3_table() -> str:
    rows = []
    for d, lab in (("P3_lin_dense", "linear, MSE"), ("P3_cos_dense", "cosine, MSE"), ("P3_lin_auxdense", "linear, aux CE 0.001"), ("P3_cos_aux", "cosine, aux CE 0.001")):
        r = J(f"reports/gates/{d}/RESULT.json")
        if not r:
            continue
        for x in sorted(r["diagnostics"]["all_checkpoint_rows"], key=lambda x: x["step"]):
            if x["step"] >= 8000:
                rows.append((lab, x["step"], f"{x['broken_rate']:.3f}", f"{x['copy_rate']:.3f}", f"{x['floor_valid']:.3f}", f"{x['novelty_med']:.1f}"))
    return md_table(["variant", "step", "broken rate (gate <= 0.05)", "copy rate (gate <= 0.154)", "floor validity", "median nearest-train Hamming"], rows) if rows else "n/a (not run)"


def cfg_table() -> str:
    p = ROOT / "reports/gates/P4/cfg_sweep.csv"
    if not p.exists():
        return "n/a (not run)"
    import csv
    rows = list(csv.DictReader(open(p)))
    return md_table(["w", "C-score", "exact match", "broken rate", "median novelty", "copy rate", "S_noplay"],
                    [(r["w"], r["c_score"], r["exact_match"], r["broken_rate"], r["novelty_med"], r["copy_rate"], r["S_noplay"]) for r in rows])


def x1_table() -> str:
    rows = []
    for tag, lab in (("C-main", "C-main (MSE)"), ("C-x01", "X1 (aux CE 0.001)")):
        for f in sorted((ROOT / "reports/dev_eval").glob(f"{tag}_*_p_dev_w5.json")):
            r = json.loads(f.read_text())
            rows.append((lab, r["step"], f"{r['c_score']:.3f}", f"{r['broken_rate']:.3f}", f"{r['novelty_med']:.1f}", f"{r['copy_rate']:.3f}"))
    return NL + NL + md_table(["run", "step", "C-score (w=5)", "broken rate", "median novelty", "copy rate"], sorted(rows, key=lambda r: (r[0], r[1]))) if rows else "n/a (not run)"


def per_level_table() -> str:
    p = ROOT / "reports/final/per_level_P-test.csv"
    if not p.exists():
        return ""
    import csv
    rows = list(csv.DictReader(open(p)))
    return "**Per held-out level (P-test)**" + NL + NL + md_table(["level", "samples", "C-score", "exact match", "broken rate", "floor validity"],
                                                     [(r["level"], r["n_samples"], r["c_score"], r["exact_match"], r["broken_rate"], r["floor_valid"]) for r in rows])


def concept_table() -> str:
    p = ROOT / "reports/final/per_concept_P-test.csv"
    if not p.exists():
        return ""
    import csv
    rows = list(csv.DictReader(open(p)))
    return "**Per concept (P-test)**" + NL + NL + md_table(["concept", "support", "precision", "recall", "F1", "exact match", "bucket MAE"],
                                                  [(r["concept"], r["support"], r["precision"], r["recall"], r["f1"], r["exact_match"], r["bucket_mae"]) for r in rows])


def values() -> dict:
    v = {}
    fz = J("FROZEN.json") or {}
    v["frozen_step"], v["frozen_w"], v["frozen_sha"] = fmt(fz.get("step")), fmt(fz.get("w")), (fz.get("checkpoint_sha256") or "n/a")[:16]
    v["frozen_schedule"] = fmt(fz.get("schedule"))
    m = J("reports/final/metrics.json") or {}
    for nm, key in (("P-test", "t"), ("P-rand", "r")):
        s = m.get(nm) or {}
        for k in ("c_score", "exact_match", "broken_rate", "floor_valid", "checker_verified_rate"):
            v[f"{key}_{k}"] = ci(s.get(k), 3, s.get("n_levels"))
        for k in ("novelty_med", "copy_rate", "tile_js", "rare_exact_match"):
            v[f"{key}_{k}"] = fmt(s.get(k))
        v[f"{key}_n_levels"] = fmt(s.get("n_levels"))
        v[f"{key}_n_prompts"] = fmt(s.get("n_prompts"))
        for k in ("null_other_level", "always_empty", "always_full_floor", "always_modal"):
            v[f"{key}_ref_{k}"] = ("undefined (no source levels)" if (s.get("references") or {}).get(k) != (s.get("references") or {}).get(k) else fmt((s.get("references") or {}).get(k)))
    c = m.get("P-casual-test") or {}
    v["casual_n1"], v["casual_c"] = fmt(c.get("N1_scene_count_acc")), fmt(c.get("c_score_generated_scenes_vs_expected"))
    v["casual_em"] = fmt(c.get("N4_exact_caption_match_vs_fullcorpus_table"))
    p4 = J("reports/gates/P4/RESULT.json") or {}
    d4 = p4.get("diagnostics", {})
    v["p4_pass"] = "PASS" if p4.get("pass") else ("FAIL" if p4 else "n/a (not run)")
    v["p4_w"] = fmt(d4.get("best_w"))
    v["p4_null"], v["p4_delta"] = fmt(d4.get("cscore_null_dev")), fmt(d4.get("delta_vs_null"))
    for ch in p4.get("checks", []):
        v[f"p4_{ch['id'].split('.')[1]}"] = f"{ch['value']}" + (f" (95% CI {ch['ci95'][0]:.3f} to {ch['ci95'][1]:.3f})" if "ci95" in ch else "") + (" PASS" if ch["pass"] else " FAIL")
    a = d4.get("at_w_star", {})
    for k in ("c_score", "exact_match", "broken_rate", "floor_valid", "novelty_med", "copy_rate", "tile_js", "rare_exact_match"):
        v[f"dev_{k}"] = fmt(a.get(k))
    p5 = J("reports/gates/P5/RESULT.json") or {}
    d5 = p5.get("diagnostics", {})
    v["p5_pass"] = "PASS" if p5.get("pass") else ("FAIL / partial" if p5 else "n/a (not run)")
    for ch in p5.get("checks", []):
        v[f"p5_{ch['id'].split('.')[1]}"] = f"{ch['value']}" + (f" (95% CI {ch['ci95'][0]:.3f} to {ch['ci95'][1]:.3f})" if "ci95" in ch else "") + (" PASS" if ch["pass"] else " FAIL")
    for k in ("verified_rate", "pre_repair_broken_rate", "seconds_per_level_mean", "seconds_per_level_p95", "real_dev_pseudo_seam_ratio_mean"):
        v[f"p5_{k}"] = fmt(d5.get(k))
    cal = J("reports/gates/P5b/calibration.json") or {}
    v["cal_levels"] = fmt(cal.get("full_level_solved_rate"))
    v["cal_windows"] = fmt((cal.get("windows") or {}).get("solved_rate"))
    p5b = J("reports/gates/P5b/RESULT.json") or {}
    mm = p5b.get("metrics", {})
    v["p5b_agree"], v["p5b_kappa"], v["p5b_fv"] = fmt(mm.get("agreement")), fmt(mm.get("kappa")), fmt(mm.get("false_verified_rate"))
    v["p5b_pass"] = "PASS" if p5b.get("pass") else ("FAIL" if p5b else "n/a (not run)")
    rv = J("reports/gates/P5b/replay_real_windows.json") or {}
    v["replay_rate"], v["replay_n"] = fmt(rv.get("replay_success_rate")), fmt(rv.get("n"))
    for tag in ("showcase", "p5"):
        rl = J(f"reports/gates/P5b/replay_levels_{tag}.json") or {}
        v[f"rl_{tag}_rate"], v[f"rl_{tag}_n"] = fmt(rl.get("success_rate")), fmt(rl.get("n"))
    gr = J("reports/gates/P5b/game_replay.json") or {}
    v["game_unsound"] = fmt(len(gr.get("unsound_checker_solved_game_failed", [])) if gr else None)
    v["game_cases"] = fmt(gr.get("n"))
    p3 = J("reports/gates/P3_baseline_mse/RESULT.json") or {}
    v["p3_pass"] = "PASS" if p3.get("pass") else ("FAIL" if p3 else "n/a")
    lat = J("reports/app/latency.json") or {}
    for k, val in lat.items():
        v[f"lat_{k}"] = fmt(val)
    x1 = J("reports/dev_eval/x01_summary.json") or {}
    for k, val in x1.items():
        v[f"x1_{k}"] = fmt(val)
    pytest = J("reports/testing/pytest_summary.json") or {}
    v["tests_total"], v["tests_pass"] = fmt(pytest.get("total")), fmt(pytest.get("passed"))
    v["p3_table"], v["cfg_table"], v["x1_table"] = p3_table(), cfg_table(), x1_table()
    v["per_level_table"], v["concept_table"] = per_level_table(), concept_table()
    return v


def render(md: str, vals: dict) -> str:
    missing = set()

    def sub(m):
        k = m.group(1)
        if k not in vals:
            missing.add(k)
            return "n/a (not run)"
        return str(vals[k])
    out = re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", sub, md)
    if missing:
        print("placeholders without a value:", ", ".join(sorted(missing)), file=sys.stderr)
    return out


CSS = """body{font-family:Georgia,serif;max-width:860px;margin:24px auto;padding:0 20px;line-height:1.45;font-size:11.5pt;color:#111}
h1,h2,h3{font-family:Arial,sans-serif;color:#1b2a49}h1{font-size:22pt}h2{font-size:15pt;border-bottom:1px solid #ccd;padding-bottom:3px;margin-top:26px}
table{border-collapse:collapse;margin:10px 0;font-size:9.5pt}td,th{border:1px solid #bbb;padding:3px 7px}th{background:#eef}
code,pre{font-family:Consolas,monospace;font-size:9.5pt;background:#f4f4f6}pre{padding:8px;overflow-x:auto}img{max-width:100%}
.figcap{font-size:9pt;color:#444;margin-top:-6px}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    import markdown
    md = render((REP / "report_template.md").read_text(encoding="utf-8"), values())
    (REP / "report.md").write_text(md, encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>text2mario report</title><style>{CSS}</style></head><body>{html}</body></html>"
    (REP / "report.html").write_text(page, encoding="utf-8")
    if a.no_pdf:
        print("wrote report.md / report.html")
        return
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto((REP / "report.html").as_uri())
        pg.wait_for_timeout(500)
        pg.pdf(path=str(REP / "report.pdf"), format="A4", margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
               print_background=True)
        b.close()
    print("wrote report/report.pdf", (REP / "report.pdf").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
