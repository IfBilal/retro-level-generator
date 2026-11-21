"""Dev evaluation of conditional checkpoints (and the CFG sweep): python scripts/eval_dev.py CKPT [CKPT...] --w 1 5 ...

Explain-me:
  What: for each checkpoint and CFG scale, samples the dev prompts (3 seeds each), prints C-score / exact match /
        broken rate / floor validity / novelty / copy rate and saves one JSON per (checkpoint, set, w) under
        reports/dev_eval/.  Dev sets only (p_dev, p_rand); selection and the CFG choice are made HERE, never on test numbers.
  In/Out: checkpoints -> reports/dev_eval/<run>_<step>_<set>_w<w>.json (+ a sample sheet PNG for the first scale).
  Previous: eval/devloop.py.  Next: choosing the checkpoint (best dev composite) and the default w (G-P4).
  Try: --w 1 2 3 5 7.5 10 for the Day-9 sweep; --n-prompts 20 for a quick look.
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

from text2mario.eval.devloop import evaluate  # noqa: E402
from text2mario.eval.generate import load_ema_conditional  # noqa: E402
from text2mario.render.renderer import sheet  # noqa: E402

KEYS = ("c_score", "exact_match", "broken_rate", "floor_valid", "novelty_med", "copy_rate", "tile_js", "rare_exact_match")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--w", type=float, nargs="+", default=[5.0])
    ap.add_argument("--set", default="p_dev", choices=["p_dev", "p_rand"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "reports" / "dev_eval"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for c in a.ckpts:
        c = Path(c)
        bundle = load_ema_conditional(c, dev)
        run = c.parent.name
        for w in a.w:
            r = evaluate(bundle, a.set, w, a.seeds, a.steps, a.n_prompts, dev)
            grids, caps = r.pop("_grids"), r.pop("_captions")
            tag = f"{run}_{r['step']:07d}_{a.set}_w{w:g}"
            (out / f"{tag}.json").write_text(json.dumps(r, indent=1, default=float))
            sheet(list(grids[:64]), cols=8, tile_px=8).save(out / f"{tag}.png")
            print(f"{run} step {r['step']:6d} {a.set} w={w:<4g} " + "  ".join(f"{k}={r[k]:.3f}" for k in KEYS if k in r and r[k] == r[k]), flush=True)


if __name__ == "__main__":
    main()
