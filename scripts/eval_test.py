"""THE one-shot test evaluation (doc 22 §6.5): P-test, P-rand (fresh seeds), P-casual-test, on the FROZEN model.  Run exactly once.

Explain-me:
  What: refuses to start unless data/splits/LOCK verifies, FROZEN.json exists and its checkpoint / eval-plan hashes match, and no
        earlier test run is logged.  Then samples the held-out prompts at the frozen CFG scale and computes every pre-registered
        metric with level-clustered bootstrap intervals, a per-level table, a per-concept table, the null / constant-generator
        references and a failure gallery.  The invocation is appended to reports/final/test_invocations.jsonl.
  In/Out: FROZEN.json + p_test.json + p_rand.json + doc 12 §8.2 -> reports/final/{metrics.json, RESULTS.md, per_level.csv, per_concept.csv,
          failure_gallery.png}.
  Previous: scripts/freeze.py.  Next: the report quotes RESULTS.md verbatim.
  Try: there is nothing to tune here.  --dry-run checks every guard and exits without touching a test prompt.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from text2mario.utils.env import bootstrap_env  # noqa: E402

bootstrap_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from text2mario.data import splits  # noqa: E402
from text2mario.utils.gates import git_sha  # noqa: E402

FINAL = ROOT / "reports/final"
LOG = FINAL / "test_invocations.jsonl"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def guards(allow_rerun: str | None) -> dict:
    splits.load(verify=True)                                              # LOCK must verify
    fz_path = ROOT / "FROZEN.json"
    if not fz_path.exists():
        sys.exit("FROZEN.json does not exist: freeze first (scripts/freeze.py).")
    fz = json.loads(fz_path.read_text())
    if sha_file(FINAL / "eval_plan.json") != fz["eval_plan_sha256"]:
        sys.exit("eval_plan.json changed after the freeze.")
    ck = ROOT / fz["checkpoint"] if not Path(fz["checkpoint"]).is_absolute() else Path(fz["checkpoint"])
    if sha_file(ck) != fz["checkpoint_sha256"]:
        sys.exit("the frozen checkpoint's hash does not match FROZEN.json.")
    if (ROOT / "data/splits/LOCK").read_text().strip().split()[0] != fz["split_lock_sha256"]:
        sys.exit("split LOCK differs from the frozen one.")
    if LOG.exists() and any(json.loads(line).get("completed") for line in LOG.read_text().splitlines() if line.strip()):
        if not allow_rerun:
            sys.exit("a test run is already logged: the test set is one-shot. (A metric-code bug is the only reason to re-run; "
                     "pass --allow-rerun 'reason' and document BOTH runs.)")
    fz["_ckpt_path"] = str(ck)
    return fz


def boot(vals, clusters):
    from text2mario.eval.stats import cluster_bootstrap
    m, lo, hi = cluster_bootstrap(vals, clusters)
    return {"mean": m, "ci95": [lo, hi]}


def summarize(res: dict) -> dict:
    ps = res["per_sample"]
    cl = [p["level"] for p in ps]
    out = {"n_prompts": res["n_prompts"], "n_samples": res["n_samples"], "n_levels": len({c for c in cl}),
           "c_score": boot([p["c_score"] for p in ps], cl), "exact_match": boot([float(p["exact_match"]) for p in ps], cl),
           "broken_rate": boot([float(p["broken"]) for p in ps], cl), "floor_valid": boot([float(p["floor_valid"]) for p in ps], cl),
           "novelty_med": res.get("novelty_med"), "copy_rate": res.get("copy_rate"), "tile_js": res.get("tile_js"),
           "rare_exact_match": res.get("rare_exact_match")}
    if all(p["nn_hamming"] is not None for p in ps):
        out["copy_rate_ci"] = boot([float(p["nn_hamming"] <= 4) for p in ps], cl)
    return out


def per_level(res: dict) -> list[dict]:
    rows = {}
    for p in res["per_sample"]:
        r = rows.setdefault(p["level"], {"level": p["level"], "n": 0, "c": [], "em": [], "broken": [], "floor": []})
        r["n"] += 1; r["c"].append(p["c_score"]); r["em"].append(p["exact_match"]); r["broken"].append(p["broken"]); r["floor"].append(p["floor_valid"])
    return [{"level": k, "n_samples": v["n"], "c_score": float(np.mean(v["c"])), "exact_match": float(np.mean(v["em"])),
             "broken_rate": float(np.mean(v["broken"])), "floor_valid": float(np.mean(v["floor"]))} for k, v in sorted(rows.items(), key=lambda kv: str(kv[0]))]


def references(rows) -> dict:
    """What a generator that ignores the prompt would score (so a C-score is never read in isolation, doc 22 §3.1)."""
    from text2mario.eval import adherence as A
    prompts = [r["prompt"] for r in rows]
    lv = [(r.get("source") or {}).get("level") for r in rows]
    exp = [r["expected_caption"] for r in rows]
    null = [A.c_score(prompts[i], exp[j]) for i in range(len(rows)) for j in range(len(rows)) if lv[i] != lv[j]]
    const = {name: float(np.mean([A.c_score(p, c) for p in prompts])) for name, c in
             (("always_empty", ""), ("always_full_floor", "full floor."), ("always_modal", "full floor. a few enemies."))}
    return {"null_other_level": float(np.mean(null)), **const}


def casual_test(bundle, w, device) -> dict:
    import eval_normalizer as EN

    from text2mario.eval import adherence as A
    from text2mario.eval.generate import sample_conditional
    from text2mario.captions.detector import caption
    from text2mario.text.normalizer import normalize
    rows = EN.parse_test()
    n_scene = em = 0
    cs = []
    for r in rows:
        out = normalize(r["prompt"])
        exp_scenes = [c.strip() for c in r["expected_caption_fullcorpus"].replace("(CLAMPED)", "").split(" / ") if c.strip()]
        n_scene += len(out.scenes) == len(exp_scenes)
        em += [s.caption for s in out.scenes] == exp_scenes
        for k, s in enumerate(out.scenes[:len(exp_scenes)]):
            seeds = [10_000_000 + 1000 * len(cs) + j for j in range(3)]
            g = sample_conditional(bundle[0], bundle[1], bundle[2], [s.caption] * 3, w, seeds, 50, device=device,
                                   autocast_dtype=torch.bfloat16 if device == "cuda" else None)
            cs.append(float(np.mean([A.c_score(exp_scenes[k], caption(x)) for x in g])))
    return {"n_prompts": len(rows), "N1_scene_count_acc": n_scene / len(rows), "N4_exact_caption_match_vs_fullcorpus_table": em / len(rows),
            "c_score_generated_scenes_vs_expected": float(np.mean(cs)) if cs else None, "n_scenes_generated": len(cs),
            "note": "expected captions come from the full-corpus clamp table; the normalizer clamps on the train split, so small differences are expected"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-rerun", default=None, help="reason (metric-code bug only)")
    a = ap.parse_args()
    fz = guards(a.allow_rerun)
    print("guards OK; frozen step", fz["step"], "w", fz["w"])
    if a.dry_run:
        return
    from text2mario.eval.devloop import evaluate_rows
    from text2mario.eval.generate import load_ema_conditional
    from text2mario.playability.checker import check_level, frame_window
    FINAL.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOG, "a") as f:
        f.write(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_sha": git_sha(), "checkpoint_sha256": fz["checkpoint_sha256"],
                            "completed": False, "rerun_reason": a.allow_rerun}) + "\n")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_ema_conditional(fz["_ckpt_path"], dev, max_len=64)   # P-rand has a prompt of 49 tokens (> 48 seen in training)
    w = fz["w"]
    p_test = json.loads((ROOT / "data/prompts/p_test.json").read_text())
    p_rand = json.loads((ROOT / "data/prompts/p_rand.json").read_text())
    results, tables = {}, {}
    for name, rows, base in (("P-test", p_test, 8_000_000), ("P-rand", p_rand, 9_000_000)):
        res = evaluate_rows(bundle, rows, w, base, name, 3, 50, dev)
        grids = res.pop("_grids"); caps = res.pop("_captions")
        ver = []
        for g in grids:                                        # structure-only verification of each framed 16-wide sample
            fg, fr = frame_window(g)
            ver.append(check_level(fg, mode="level", framing=fr, enemies="ignore").status == "solved")
        s = summarize(res)
        s["checker_verified_rate"] = boot([float(v) for v in ver], [p["level"] for p in res["per_sample"]])
        s["references"] = references(rows)
        s["concepts"] = res["concepts"]
        results[name] = s
        tables[name] = {"per_level": per_level(res), "worst": sorted(range(len(grids)), key=lambda i: res["per_sample"][i]["c_score"])[:20],
                        "grids": grids, "caps": caps, "prompts": [r["prompt"] for r in rows for _ in range(3)], "per_sample": res["per_sample"]}
        print(name, json.dumps({k: (v["mean"] if isinstance(v, dict) and "mean" in v else None) for k, v in s.items() if isinstance(v, dict) and "mean" in v}))
    results["P-casual-test"] = casual_test(bundle, w, dev)
    tok = bundle[2].tok
    results["_token_lengths"] = {nm: {"prompts_over_48_tokens": int(sum(len(tok(r["prompt"])["input_ids"]) > 48 for r in rws)), "max": int(max(len(tok(r["prompt"])["input_ids"]) for r in rws))}
                                 for nm, rws in (("P-test", p_test), ("P-rand", p_rand))}
    results["_meta"] = {"frozen": {k: fz[k] for k in ("checkpoint", "checkpoint_sha256", "step", "w", "config_hash")}, "git_sha": git_sha(),
                        "minutes": round((time.time() - t0) / 60, 1), "device": dev}
    (FINAL / "metrics.json").write_text(json.dumps(results, indent=1, default=float))
    for name, t in tables.items():
        (FINAL / f"per_level_{name}.csv").write_text("level,n_samples,c_score,exact_match,broken_rate,floor_valid\n" + "\n".join(
            f"{r['level']},{r['n_samples']},{r['c_score']:.4f},{r['exact_match']:.4f},{r['broken_rate']:.4f},{r['floor_valid']:.4f}" for r in t["per_level"]))
    rep = results["P-test"]["concepts"]
    (FINAL / "per_concept_P-test.csv").write_text("concept,support,precision,recall,f1,exact_match,absence_respect,bucket_mae,bucket_bias\n" + "\n".join(
        f"{k},{v['support']},{v['precision']:.3f},{v['recall']:.3f},{v['f1']:.3f},{v['exact_match']:.3f},{v['absence_respect']:.3f},{v['bucket_mae']:.3f},{v['bucket_bias']:.3f}"
        for k, v in rep.items()))
    from PIL import Image, ImageDraw
    from text2mario.render.renderer import grid_to_sprite_image
    t = tables["P-test"]
    ims = [(grid_to_sprite_image(t["grids"][i], 8), t["prompts"][i][:60], t["per_sample"][i]["c_score"]) for i in t["worst"][:20]]
    W, H = ims[0][0].size
    sheet = Image.new("RGB", (4 * (W + 6), 5 * (H + 22)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (im, pr, c) in enumerate(ims):
        x, y = (k % 4) * (W + 6), (k // 4) * (H + 22)
        sheet.paste(im, (x, y + 18)); d.text((x + 1, y + 2), f"C={c:.2f} {pr}", fill=(0, 0, 0))
    sheet.save(FINAL / "failure_gallery.png")
    r = results["P-test"]
    md = [f"# One-shot test results (frozen step {fz['step']}, CFG w = {w})", "",
          f"Checkpoint sha256 `{fz['checkpoint_sha256'][:16]}...`, git `{git_sha()[:10]}`. 95 % intervals resample the **{r['n_levels']} held-out levels** "
          "(clusters); with so few levels they are wide.", "",
          "| Metric | P-test | P-rand |", "|---|---|---|"]
    for k, label in (("c_score", "C-score"), ("exact_match", "exact match"), ("broken_rate", "broken-structure rate (raw)"), ("floor_valid", "floor validity"),
                     ("checker_verified_rate", "checker-verified rate (framed windows, diagnostic)")):
        cells = []
        for nm in ("P-test", "P-rand"):
            v = results[nm][k]
            cells.append(f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        md.append(f"| {label} | {cells[0]} | {cells[1]} |")
    for k, label in (("novelty_med", "median nearest-train Hamming"), ("copy_rate", "copy rate"), ("tile_js", "tile JS divergence"), ("rare_exact_match", "rare-concept exact match")):
        md.append(f"| {label} | {results['P-test'][k]:.3f} | {results['P-rand'][k]:.3f} |")
    md += ["", "References on the same prompts (a generator that ignores the prompt):", "",
           "| Reference | P-test C-score | P-rand C-score |", "|---|---|---|"]
    for k in ("null_other_level", "always_empty", "always_full_floor", "always_modal"):
        md.append(f"| {k} | {results['P-test']['references'][k]:.3f} | {results['P-rand']['references'][k]:.3f} |")
    c = results["P-casual-test"]
    md += ["", f"P-casual-test (normalizer on, {c['n_prompts']} prompts): scene-count accuracy {c['N1_scene_count_acc']:.2f}, caption match vs the expected table "
           f"{c['N4_exact_caption_match_vs_fullcorpus_table']:.2f}, C-score of generated scenes vs expected {c['c_score_generated_scenes_vs_expected']:.3f}.",
           "", "Per-level, per-concept tables and the failure gallery are alongside this file."]
    (FINAL / "RESULTS.md").write_text("\n".join(md) + "\n")
    lines = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    lines[-1].update({"completed": True, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    LOG.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    print((FINAL / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
