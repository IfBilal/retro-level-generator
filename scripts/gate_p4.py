"""Gate G-P4: text conditioning + classifier-free guidance (doc 24 §G-P4, doc 15, doc 22 §6.2).

Explain-me:
  What: sweeps the CFG scale w on P-dev (3 seeds, common random numbers), picks w* by the dev composite S_noplay
        (0.5 C-score + 0.25 (1 - broken) + 0.25 min(1, novelty/ref); the playable term stays out until G-P5b passes),
        then checks P4.01 C-score(w*) >= 0.75, P4.02 CFG helps (paired, level-clustered bootstrap CI above 0),
        P4.03 optimiser membership, P4.04 the PDF contrast test (8 seeds) and P4.05 diagnostics (vs the null, per concept).
  In/Out: a conditional checkpoint -> reports/gates/P4/RESULT.json, cfg_sweep.csv/png, contrast.png.
  Previous: scripts/eval_dev.py, eval/devloop.py.  Next: the config-v1 decision and Phase 5.
  Try: --w 1 5 for a fast run; --n-prompts 30 --seeds 1 for a smoke check (not a result).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.utils.env import bootstrap_env  # noqa: E402

bootstrap_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from text2mario.captions.detector import detect  # noqa: E402
from text2mario.eval import adherence as A  # noqa: E402
from text2mario.eval.devloop import evaluate, load_prompts  # noqa: E402
from text2mario.eval.generate import load_ema_conditional, sample_conditional  # noqa: E402
from text2mario.eval.stats import cluster_bootstrap, n_clusters, paired_cluster_bootstrap  # noqa: E402
from text2mario.render.renderer import sheet  # noqa: E402
from text2mario.text.normalizer import normalize  # noqa: E402
from text2mario.utils.gates import write_result  # noqa: E402

REFS = json.loads((ROOT / "reports/references/REFERENCES.json").read_text())


def s_noplay(c, broken, nov):
    return 0.5 * c + 0.25 * (1 - broken) + 0.25 * min(1.0, nov / REFS["novelty_ref_median_stride8"])


def null_cscore(rows) -> float:
    """C-score of a realistic scene that ignores the prompt: every dev prompt against the captions of other levels."""
    pr = load_prompts("p_dev")
    vals = []
    for i, a in enumerate(pr):
        for j, b in enumerate(pr):
            if (a.get("source") or {}).get("level") != (b.get("source") or {}).get("level"):
                vals.append(A.c_score(a["prompt"], b["expected_caption"]))
    return float(np.mean(vals))


def contrast(bundle, w, dev, seeds=8):
    """The PDF contrast test (P4.04): two normalised prompts, 8 seeds each, three paired comparisons."""
    pa = normalize("full floor with three pipes and no enemies").scenes[0].caption
    pb = normalize("staircase with gap and several enemies").scenes[0].caption
    seed_list = [9_000_000 + k for k in range(seeds)]
    ga = sample_conditional(bundle[0], bundle[1], bundle[2], [pa] * seeds, w, seed_list, 50, device=dev, autocast_dtype=torch.bfloat16 if dev == "cuda" else None)
    gb = sample_conditional(bundle[0], bundle[1], bundle[2], [pb] * seeds, w, seed_list, 50, device=dev, autocast_dtype=torch.bfloat16 if dev == "cuda" else None)

    def feats(g):
        d = {x.topic: x for x in detect(g)}
        pipes = sum(d[t].count or 0 for t in ("pipe", "upside down pipe", "broken pipe") if t in d)
        enemies = int((g == 5).sum())
        gaps = (d["floor"].count or 0) if "floor" in d and d["floor"].count else 0
        stair = int("ascending staircase" in d or "descending staircase" in d)
        return pipes, enemies, gaps + stair
    fa, fb = [feats(g) for g in ga], [feats(g) for g in gb]
    wins = [sum(a[i] > b[i] for a, b in zip(fa, fb)) if i != 1 else sum(a[1] < b[1] for a, b in zip(fa, fb)) for i in (0, 1)]
    wins.append(sum(b[2] > a[2] for a, b in zip(fa, fb)))
    means = {"pipes_A": float(np.mean([f[0] for f in fa])), "pipes_B": float(np.mean([f[0] for f in fb])),
             "enemy_tiles_A": float(np.mean([f[1] for f in fa])), "enemy_tiles_B": float(np.mean([f[1] for f in fb])),
             "gaps_plus_stairs_A": float(np.mean([f[2] for f in fa])), "gaps_plus_stairs_B": float(np.mean([f[2] for f in fb]))}
    ok = (means["pipes_A"] > means["pipes_B"] and means["enemy_tiles_A"] < means["enemy_tiles_B"]
          and means["gaps_plus_stairs_B"] > means["gaps_plus_stairs_A"] and all(x >= 6 for x in wins))
    return {"prompt_A": pa, "prompt_B": pb, "paired_wins_of_8": {"pipes": wins[0], "enemy_presence": wins[1], "gaps_stairs": wins[2]},
            "means": means, "pass": bool(ok)}, np.concatenate([ga, gb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--w", type=float, nargs="+", default=[1, 2, 3, 5, 7.5, 10])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else ROOT / "reports/gates/P4"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_ema_conditional(a.ckpt, dev)
    sweep = {}
    for w in a.w:
        r = evaluate(bundle, "p_dev", w, a.seeds, 50, a.n_prompts, dev)
        grids = r.pop("_grids")
        r.pop("_captions")
        r["S_noplay"] = s_noplay(r["c_score"], r["broken_rate"], r["novelty_med"])
        sweep[w] = (r, grids)
        print(f"w={w:<4g} C={r['c_score']:.3f} EM={r['exact_match']:.3f} broken={r['broken_rate']:.3f} nov={r['novelty_med']:.1f} "
              f"copy={r['copy_rate']:.3f} S={r['S_noplay']:.3f}", flush=True)
    best_w = max(a.w, key=lambda w: (round(sweep[w][0]["S_noplay"], 2), -w))      # |dS| < 0.01 ties -> the smaller w
    r_star, r_one = sweep[best_w][0], sweep.get(1.0, sweep[a.w[0]])[0]
    clusters = [p["level"] for p in r_star["per_prompt"]]
    dC = paired_cluster_bootstrap([p["c_score"] for p in r_star["per_prompt"]], [p["c_score"] for p in r_one["per_prompt"]], clusters)
    cs = cluster_bootstrap([p["c_score"] for p in r_star["per_prompt"]], clusters)
    null = null_cscore(None)
    opt = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    n_opt = sum(v["exp_avg"].numel() for v in opt["opt"]["state"].values()) if opt["opt"]["state"] else 0
    n_model = sum(p.numel() for p in bundle[0].parameters()) + sum(p.numel() for p in bundle[2].adapter.parameters())
    contrast_res, cg = contrast(bundle, best_w, dev)
    sheet(list(cg), cols=8, tile_px=8).save(out / "contrast.png")
    sheet(list(sweep[best_w][1][:64]), cols=8, tile_px=8).save(out / "sample_sheet_best_w.png")
    rows = ["w,c_score,exact_match,broken_rate,floor_valid,novelty_med,copy_rate,rare_exact_match,S_noplay"]
    for w in a.w:
        r = sweep[w][0]
        rows.append(",".join(str(round(r[k], 4)) for k in ("c_score", "exact_match", "broken_rate", "floor_valid", "novelty_med", "copy_rate", "rare_exact_match", "S_noplay")).join([f"{w},", ""]))
    (out / "cfg_sweep.csv").write_text("\n".join(rows) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3))
        ws = list(a.w)
        ax[0].plot(ws, [sweep[w][0]["c_score"] for w in ws], "o-"); ax[0].axhline(0.75, ls="--"); ax[0].set_title("C-score vs w")
        ax[1].plot(ws, [sweep[w][0]["broken_rate"] for w in ws], "o-"); ax[1].set_title("broken rate vs w")
        ax[2].plot(ws, [sweep[w][0]["copy_rate"] for w in ws], "o-"); ax[2].set_title("copy rate vs w")
        for x in ax:
            x.set_xlabel("CFG w")
        fig.tight_layout(); fig.savefig(out / "cfg_sweep.png", dpi=120)
    except Exception as e:  # plotting is a convenience
        print("plot skipped:", e)
    checks = [{"id": "P4.01", "name": "dev C-score at w*", "value": round(cs[0], 4), "ci95": [round(cs[1], 4), round(cs[2], 4)], "op": ">=", "threshold": 0.75,
               "pass": cs[0] >= 0.75, "n_prompts": r_star["n_prompts"], "n_clusters": n_clusters(clusters)},
              {"id": "P4.02", "name": "CFG helps (paired dC, w* - w=1)", "value": round(dC[0], 4), "ci95": [round(dC[1], 4), round(dC[2], 4)],
               "op": "ci_lo >", "threshold": 0, "pass": dC[1] > 0},
              {"id": "P4.03", "name": "optimiser membership (U-Net + adapter, no MiniLM)", "value": int(n_opt), "expected": int(n_model),
               "op": "==", "pass": int(n_opt) == int(n_model)},
              {"id": "P4.04", "name": "PDF contrast test", "value": contrast_res["paired_wins_of_8"], "op": "all >= 6/8", "pass": contrast_res["pass"]}]
    passed = all(c["pass"] for c in checks)
    diag = {"best_w": best_w, "cscore_null_dev": null, "delta_vs_null": cs[0] - null,
            "warning": ("conditioning adds little beyond realism" if cs[0] - null < 0.15 else None),
            "per_concept_at_w_star": r_star["concepts"], "S_vs_w": {str(w): sweep[w][0]["S_noplay"] for w in a.w},
            "at_w_star": {k: r_star[k] for k in ("c_score", "exact_match", "broken_rate", "floor_valid", "novelty_med", "copy_rate",
                                                  "tile_js", "rare_exact_match")}, "contrast": contrast_res,
            "note": "5 dev levels -> wide intervals (doc 22 §7.3); selection uses S_noplay because G-P5b has not passed"}
    write_result("P4", passed, {"w_star": best_w, "checkpoint": str(a.ckpt), "step": bundle[4]["step"]}, checks, out_dir=out, diagnostics=diag)
    print(json.dumps({c["id"]: (c["value"], c["pass"]) for c in checks}, default=str))
    print("G-P4", "PASS" if passed else "FAIL", "| w* =", best_w)


if __name__ == "__main__":
    main()
