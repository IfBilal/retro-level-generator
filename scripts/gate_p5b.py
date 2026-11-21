"""Gate G-P5b: is the fast playability checker valid enough to be used in selection? (doc 24 §G-P5b, doc 21 §7-§9)

Explain-me:
  What: builds the fixed 300-item agreement set (100 real train/dev windows, 100 Phase-3 unconditional samples, 100 framed
        48-wide real segments; test levels never), asks MarioEval.jar and the fast checker, and reports agreement, Cohen's kappa,
        the false-"verified" rate and the jar timeouts.  Threshold: agreement >= 0.85 (P5b.01).  It also folds in the
        game-engine contract check (reports/gates/P5b/game_replay.json) as supplementary evidence.
        A FAIL keeps S_noplay in every selection (CLAUDE.md rule 15); nothing is waived.
  In/Out: reports/gates/P5b/RESULT.json, agreement_set.json, disagreements.csv.  Previous: scripts/jar_agreement.py (cache).
  Try: python scripts/gate_p5b.py --samples runs/U-lin-d/step_0012000.pt
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from text2mario.utils.env import bootstrap_env

bootstrap_env()
import numpy as np
from calibrate_checker import real_levels
from jar_agreement import CACHE, kappa, key

from text2mario.playability.checker import check_level, frame_window
from text2mario.playability.jar_bridge import jar_solve_many, java_available
from text2mario.playability.physics import Physics
from text2mario.utils.gates import write_result

OUT = ROOT / "reports/gates/P5b"


def build_set(samples_ckpt: str | None, seed: int = 20261117):
    rng = np.random.default_rng(seed)
    ph = Physics.load()
    lv = real_levels()
    real_w, real_seg = [], []
    for lid, (name, g, sp) in lv.items():
        for c0 in range(0, g.shape[1] - 15, 16):
            real_w.append((f"real_window:L{lid}:{c0}", *frame_window(g[:, c0:c0 + 16])))
        for c0 in range(0, g.shape[1] - 47, 48):
            real_seg.append((f"real_seg48:L{lid}:{c0}", *frame_window(g[:, c0:c0 + 48])))
    def pick(items, n):
        idx = rng.permutation(len(items))[:n]
        return [items[i] for i in sorted(idx)]
    items = pick(real_w, 100) + pick(real_seg, 100)
    if samples_ckpt:
        import torch

        from text2mario.eval.generate import load_ema_model, sample_unconditional
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        unet, sched, cfg, st = load_ema_model(samples_ckpt, dev)
        grids = sample_unconditional(unet, sched, 100, 5_000_000, 50, device=dev, autocast_dtype=torch.bfloat16 if dev == "cuda" else None)
        items += [(f"p3_sample:{i}", *frame_window(g)) for i, g in enumerate(grids)]
    return items, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None, help="a Phase-3 unconditional checkpoint to draw the 100 sample items from")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--enemies", default="ignore", choices=["ignore", "hazard"])
    ap.add_argument("--strip-enemies", action="store_true", help="remove enemies for the jar (structure-only comparison)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not java_available():
        sys.exit("Java is unavailable: use the human-labelled fallback (P5b.01-alt) - not implemented here (needs two human labellers).")
    items, ph = build_set(a.samples)
    prep = (lambda g: np.where(g == 5, 0, g).astype(g.dtype)) if a.strip_enemies else (lambda g: g)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [(lab, prep(g)) for lab, g, fr in items if key(prep(g)) not in cache]
    print(f"{len(items)} items; {len(todo)} need the jar")
    for i in range(0, len(todo), 48):
        chunk = todo[i:i + 48]
        for (lab, g), r in zip(chunk, jar_solve_many([g for _, g in chunk], workers=a.workers, timeout=120)):
            cache[key(g)] = r
        CACHE.write_text(json.dumps(cache))
        print(f"  jar {min(i + 48, len(todo))}/{len(todo)}", flush=True)
    rows = []
    for lab, g, fr in items:
        fast = check_level(g, mode="level", framing=fr, physics=ph, enemies=a.enemies).status == "solved"
        rows.append((lab, fast, cache[key(prep(g))]))
    ok = [(l, f, j) for l, f, j in rows if j is not None]
    fa, ja = [f for _, f, _ in ok], [bool(j) for _, _, j in ok]
    agree = float(np.mean([x == y for x, y in zip(fa, ja)]))
    fv = int(sum(f and not j for _, f, j in ok))
    per_src = {s: float(np.mean([f == bool(j) for l, f, j in ok if l.startswith(s)])) for s in ("real_window", "real_seg48", "p3_sample")
               if any(l.startswith(s) for l, _, _ in ok)}
    tn = int(sum((not f) and (not j) for _, f, j in ok)); tp = int(sum(f and j for _, f, j in ok))
    fn = int(sum((not f) and j for _, f, j in ok))
    metrics = {"n_items": len(rows), "jar_timeouts": len(rows) - len(ok), "agreement": agree, "kappa": kappa(fa, ja),
               "confusion": {"both_solved": tp, "both_failed": tn, "fast_solved_jar_failed": fv, "fast_failed_jar_solved": fn},
               "false_verified_rate": fv / max(1, sum(fa)), "agreement_by_source": per_src, "jar_solved_rate": float(np.mean(ja)),
               "fast_solved_rate": float(np.mean(fa)), "enemy_mode_fast": a.enemies, "jar_enemies_stripped": a.strip_enemies,
               "physics_sha256": ph.sha256}
    game = ROOT / "reports/gates/P5b/game_replay.json"
    supp = {}
    if game.exists():
        g = json.loads(game.read_text())
        supp = {"game_engine_contract": {"n_cases": g["n"], "unsound": len(g["unsound_checker_solved_game_failed"]),
                                          "conservative": len(g["conservative_checker_unsolved_game_won"]), "soundness_pass": g["soundness_pass"]}}
    checks = [{"id": "P5b.01", "name": "agreement (fast vs jar)", "value": round(agree, 4), "op": ">=", "threshold": 0.85, "pass": agree >= 0.85},
              {"id": "P5b.02a", "name": "false-verified rate (warning > 0.10)", "value": round(metrics["false_verified_rate"], 4), "op": "<=",
               "threshold": 0.10, "pass": metrics["false_verified_rate"] <= 0.10, "waivable": True}]
    passed = checks[0]["pass"]
    dis = [(l, f, j) for l, f, j in ok if f != bool(j)]
    (OUT / "disagreements.csv").write_text("item,fast_solved,jar_beaten\n" + "\n".join(f"{l},{f},{j}" for l, f, j in dis[:200]))
    (OUT / "agreement_set.json").write_text(json.dumps([l for l, _, _ in rows]))
    write_result("P5b", passed, metrics, checks, supplementary=supp, out_dir=OUT,
                 decision=("playable term MAY enter S" if passed else "KEEP S_noplay in every selection (CLAUDE.md rule 15, ledger E4); "
                           "the checker's verified badge is backed by the game-engine contract check, not by jar agreement"))
    print(json.dumps(metrics, indent=1))
    print("G-P5b", "PASS" if passed else "FAIL (documented fallback: S_noplay)")


if __name__ == "__main__":
    main()
