"""Prompt-adherence audit: python scripts/audit_adherence.py --label baseline [--seeds baseline|fresh] [--only cat,cat] [--mode fast|full] [--profile cloud]

Explain-me:
  What: for a fixed prompt list x fixed seeds, runs the SAME path the app uses (LevelService.run), counts the objects in the RAW
        generated grid (content columns only, no spawn/end template; measure-before-repair, CLAUDE.md rule 8) with the repo's own
        caption detector, cross-checks pipes/enemies/coins/cannons/question blocks with an independent counter, and compares with what
        the prompt asked for.  Adversarial prompts check rejection / warnings / injection safety.  Writes rows.jsonl (resumable),
        results.json and RESULTS.md with per-prompt pass rate + 95% Wilson CI (levels are the cluster, rule 16).
  In/Out: configs/audit/adherence.yaml -> reports/audit/<label>/{rows.jsonl, results.json, RESULTS.md}.
  Previous: src/text2mario/eval/count_audit.py (counters).  Next: reports/audit/AUDIT_REPORT.md.
  Try: --only example,ordering --seeds fresh  (fresh seeds were never used while tuning).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.app.service import LevelService
from text2mario.captions.detector import caption as detect_caption
from text2mario.eval import count_audit as CA
from text2mario.generate.full_level import STRIDE
from text2mario.text.normalizer import normalize

CFG = yaml.safe_load((ROOT / "configs/audit/adherence.yaml").read_text(encoding="utf-8"))
TOPICS = CFG["topics"]
LEAD = 3                                                   # spawn columns before the generated content (generate/framing.py)
TARGET = {"fresh": 0.9, "count": 0.9, "example": 1.0, "phrasing": 0.9, "negation": 1.0, "ordering": 1.0, "two_scene_counts": 0.9, "synonym": 0.9}


def build_prompts():
    """Expand the count grid (topic x 0..3) and append the explicit prompts."""
    ps = []
    for t in CFG["count_grid"]["topics"]:
        sp = TOPICS[t]
        for n in CFG["count_grid"]["counts"]:
            text = {0: f"full floor with no {sp['plur']}", 1: f"one {sp['sing']}", 2: f"two {sp['plur']}", 3: f"three {sp['plur']}"}[n]
            ps.append({"id": f"cnt_{t.replace(' ', '_')}_{n}", "cat": "count", "text": text, "expect": [{t: [n, n]}]})
    ps += CFG["prompts"]
    return ps


def count_level(lvl: dict, expect, order, width):
    raw = np.array(lvl["raw_grid"], dtype=np.uint8)
    content = raw[:, LEAD:LEAD + width]
    anchors = CA.objects_by_topic(content, TOPICS)
    starts = CA.scene_starts(lvl["prompt"]["scenes"], STRIDE)
    ok, details = CA.check_expectation(anchors, expect, starts, width, order)
    ind = CA.independent_counts(content)
    det = {k: len(v) for k, v in anchors.items()}
    disagree = {k: [det.get(k, 0), v] for k, v in ind.items() if det.get(k, 0) != v}
    return ok, details, det, ind, disagree, detect_caption(content)


def run_one(svc, p, seed, a, width):
    t0 = time.time()
    row = {"id": p["id"], "cat": p["cat"], "text": p["text"], "seed": seed, "device": svc.device, "profile": svc.profile, "mode": a.mode}
    try:
        res = svc.run(p["text"], seed=seed, w=CFG["defaults"]["w"], width=width, fast=(a.mode == "fast"))
    except ValueError as e:
        row.update(status="rejected", message=str(e), seconds=round(time.time() - t0, 2))
        return row
    except Exception as e:
        row.update(status="error", message=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 2))
        return row
    lvl = res.level
    ok, details, det, ind, disagree, cap = count_level(lvl, p["expect"], p.get("order"), width)
    from text2mario.data import tiles as T
    content = np.array(lvl["raw_grid"], dtype=np.uint8)[:, LEAD:LEAD + width]
    row.update(status="ok", ok=ok, details=details, grid=["".join(T.CHARS[int(v)] for v in r) for r in content], counts=det, independent=ind, counter_disagreement=disagree, raw_caption=cap,
               scenes=sorted({s["caption"] for s in lvl["prompt"]["scenes"]}, key=lambda c: [s["caption"] for s in lvl["prompt"]["scenes"]].index(c)),
               warnings=lvl["prompt"]["warnings"], forbid=lvl["prompt"]["forbid"], verification=lvl["verification"]["status"],
               reason=lvl["verification"]["reason"], attempts=lvl["verification"]["attempts_total"], broken=lvl["sanitizer"],
               seconds=res.seconds["generate_and_verify"], total_seconds=round(time.time() - t0, 2))
    return row


def run_adversarial(svc, p, seeds, width, clean_html_closers):
    rows = []
    text = p["text"] * p.get("repeat", 1)
    must = p["must"]
    if must == "warn":
        # pass = the service refuses with a clear message, OR a level comes back AND the UI-visible signals say the sentence was not (fully)
        # understood (top-level warnings, ignored words, or no recognised scene).  A silent reinterpretation ("-3 pipes" -> "two pipes") fails.
        n = normalize(text)
        flagged = bool(n.warnings) or bool(n.ignored_tokens) or not n.scenes
        row = {"id": p["id"], "cat": "adversarial", "text": text[:60], "seed": seeds[0], "status": "ok",
               "details": {"must": "warn", "scenes": [s.caption for s in n.scenes], "warnings": n.warnings, "ignored": n.ignored_tokens[:5]}}
        try:
            svc.run(text, seed=seeds[0], w=CFG["defaults"]["w"], width=width, fast=True)
            row["ok"] = flagged
        except ValueError as e:
            row.update(status="rejected", ok=True, message=str(e)[:160])
        except Exception as e:
            row.update(status="error", ok=False, message=f"{type(e).__name__}: {e}")
        rows.append(row)
        return rows
    for seed in (seeds[:1] if must in ("reject", "inject") else seeds[:2]):
        row = {"id": p["id"], "cat": "adversarial", "text": text[:60], "seed": seed, "details": {"must": must}}
        try:
            res = svc.run(text, seed=seed, w=CFG["defaults"]["w"], width=width, fast=True)
            if must == "reject":
                row.update(status="ok", ok=False, details={"must": must, "note": "a level was returned for input that should be rejected"})
            elif must == "inject":
                html = res.files["html"].decode("utf-8", "replace")
                closers = html.count("</script>")
                # a real breakout = an extra </script>, or a raw '</' / '<!--' left in the inlined level JSON (the payload text itself may stay:
                # inside a JSON string in a script element it is inert)
                part = html.split("window.__LEVEL__=")[1].split(";window.__PHYSICS__")[0] if "window.__LEVEL__=" in html else ""
                leak = closers != clean_html_closers or "</" in part or "<!--" in part
                row.update(status="ok", ok=not leak, details={"must": must, "script_closers": closers, "clean_closers": clean_html_closers,
                                                              "raw_close_or_comment_in_json": ("</" in part) or ("<!--" in part),
                                                              "payload_text_present_but_inert": p["payload"] in html})
            else:
                row.update(status="ok", ok=True)
        except ValueError as e:
            row.update(status="rejected", ok=(must == "reject"), message=str(e)[:120])
        except Exception as e:
            row.update(status="error", ok=False, message=f"{type(e).__name__}: {e}")
        rows.append(row)
    return rows


def summarise(rows):
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    out = {}
    for pid, rs in by_id.items():
        k = sum(1 for r in rs if r.get("ok"))
        lo, hi = CA.wilson(k, len(rs))
        out[pid] = {"cat": rs[0]["cat"], "text": rs[0]["text"], "n": len(rs), "pass": k, "rate": k / len(rs), "ci95": [round(lo, 3), round(hi, 3)],
                    "errors": sum(r["status"] == "error" for r in rs), "rejected": sum(r["status"] == "rejected" for r in rs),
                    "median_s": float(np.median([r["seconds"] for r in rs if "seconds" in r])) if any("seconds" in r for r in rs) else None}
    cats = defaultdict(lambda: [0, 0])
    for r in rows:
        cats[r["cat"]][0] += bool(r.get("ok"))
        cats[r["cat"]][1] += 1
    cat_out = {c: {"pass": k, "n": n, "rate": round(k / n, 3), "ci95": [round(x, 3) for x in CA.wilson(k, n)]} for c, (k, n) in cats.items()}
    tot = [sum(v[0] for v in cats.values()), sum(v[1] for v in cats.values())]
    return out, cat_out, tot


def write_md(path, label, a, per, cat, tot, rows, meta):
    L = [f"# Adherence audit `{label}`", "",
         (f"device **{meta['device']}**, mode **{a.mode}**, profile **{meta['profile']}**, width {meta['width']}, "
          f"seeds {meta['seeds']}, git {meta['git']}, UTC {meta['utc']}"), "",
         f"**Overall: {tot[0]}/{tot[1]} levels pass ({tot[0]/max(1,tot[1]):.1%})** (adversarial rows included)", "", "## By category", "",
         "| category | pass / n | rate | 95% CI | target |", "|---|---|---|---|---|"]
    for c, v in sorted(cat.items()):
        L.append(f"| {c} | {v['pass']}/{v['n']} | {v['rate']:.2f} | {v['ci95'][0]:.2f}-{v['ci95'][1]:.2f} | {TARGET.get(c, 1.0):.2f} |")
    L += ["", "## Per prompt", "", "| id | prompt | pass/n | CI | median s | meets target |", "|---|---|---|---|---|---|"]
    for pid, v in per.items():
        tgt = TARGET.get(v["cat"], 1.0)
        L.append(f"| {pid} | `{v['text'][:55]}` | {v['pass']}/{v['n']} | {v['ci95'][0]:.2f}-{v['ci95'][1]:.2f} | {v['median_s']} | {'yes' if v['rate'] >= tgt else 'NO'} |")
    dis = [r for r in rows if r.get("counter_disagreement")]
    tw = [r for r in rows if r.get("counts") and r["counts"].get("tower", 0) != r["counts"].get("tower_detector", 0)]
    L += ["", f"## Counter cross-check: {len(dis)} of {sum(1 for r in rows if r['status'] == 'ok' and 'counter_disagreement' in r)} levels where detector and independent counter disagree", ""]
    for r in dis[:15]:
        L.append(f"- {r['id']} seed {r['seed']}: {r['counter_disagreement']} (detector, independent)")
    L += ["", (f"Tower cross-check (ledger AD9): the visible-tower count differs from the caption detector's tower count in {len(tw)} of "
               f"{sum(1 for r in rows if r.get('counts'))} levels (the detector also calls a staircase's narrow top a tower and merges neighbouring pillars).")]
    ver = defaultdict(int)
    for r in rows:
        if r.get("verification"):
            ver[r["verification"]] += 1
    L += ["", f"verification status counts: {dict(ver)}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", default="baseline")
    ap.add_argument("--only", default=None, help="comma list of categories")
    ap.add_argument("--ids", default=None, help="comma list of prompt ids")
    ap.add_argument("--mode", default="fast", choices=["fast", "full"])
    ap.add_argument("--width", type=int, default=CFG["defaults"]["width"])
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument("--device", default=None, help="cpu to measure the Streamlit-Cloud situation (use with T2M_PROFILE=cloud); default: cuda if present")
    ap.add_argument("--adversarial-only", action="store_true", help="skip the prompt set, run only the input-safety prompts")
    ap.add_argument("--fresh-prompts", action="store_true", help="run the held-out prompt set (configs/audit/adherence.yaml prompts_fresh) instead")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    seeds = {"baseline": CFG["seeds_baseline"], "fresh": CFG["seeds_fresh"], "fresh2": CFG["seeds_fresh2"]}.get(a.seeds) or [int(x) for x in a.seeds.split(",")]
    seeds = seeds[:a.n_seeds] if a.n_seeds else seeds
    out = ROOT / "reports/audit" / a.label
    out.mkdir(parents=True, exist_ok=True)
    svc = LevelService(device=a.device)
    prompts = CFG["prompts_fresh"] if a.fresh_prompts else build_prompts()
    if a.adversarial_only:
        prompts = []
    if a.fresh_prompts:
        a.no_adversarial = True
    if a.only:
        prompts = [p for p in prompts if p["cat"] in a.only.split(",")]
    if a.ids:
        prompts = [p for p in prompts if p["id"] in a.ids.split(",")]
    rows_path = out / "rows.jsonl"
    done = set()
    rows = []
    if a.resume and rows_path.exists():
        rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        done = {(r["id"], r["seed"]) for r in rows}
    elif rows_path.exists():
        rows_path.unlink()
    f = open(rows_path, "a", encoding="utf-8")
    t_start = time.time()
    for i, p in enumerate(prompts):
        for s in seeds:
            if (p["id"], s) in done:
                continue
            r = run_one(svc, p, s, a, a.width)
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
        print(f"[{i + 1}/{len(prompts)}] {p['id']}: {sum(1 for r in rows if r['id'] == p['id'] and r.get('ok'))}/{len(seeds)}  ({time.time() - t_start:.0f}s)", flush=True)
    if not a.no_adversarial and not a.only and not a.ids:   # (also runs for --adversarial-only: prompts is empty)
        clean = svc.run("one pipe", seed=seeds[0], w=CFG["defaults"]["w"], width=a.width, fast=True).files["html"].decode().count("</script>")
        for p in CFG["adversarial"]:
            for r in run_adversarial(svc, p, seeds, a.width, clean):
                rows.append(r)
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[adv] {p['id']}", flush=True)
    f.close()
    import subprocess
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    per, cat, tot = summarise(rows)
    meta = {"device": svc.device, "profile": svc.profile, "width": a.width, "seeds": seeds, "git": git, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "results.json").write_text(json.dumps({"meta": meta, "mode": a.mode, "overall": tot, "by_category": cat, "per_prompt": per}, indent=1), encoding="utf-8")
    write_md(out / "RESULTS.md", a.label, a, per, cat, tot, rows, meta)
    print(f"overall {tot[0]}/{tot[1]}  -> {out}")


if __name__ == "__main__":
    main()
