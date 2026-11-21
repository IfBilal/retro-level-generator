"""Trace ONE sentence through the whole pipeline, printing every intermediate form (viva: "trace input through preprocessing and prediction").

Explain-me:
  What: sentence -> normalizer scenes -> canonical caption -> MiniLM tokens -> adapter context -> noisy start tensor -> DDIM steps (x0 clip rate)
        -> tile grid -> detector caption read back -> C-score -> framing -> checker verdict -> PNG.  Uses the exported bundle in app/model.
  In/Out: python scripts/trace_example.py "full floor with two pipes and a few enemies" [--seed 3] -> prints and writes reports/app/trace.png
  Previous: everything.  Next: nothing; it is a teaching tool.
  Try: change the sentence ("no gaps and zero cannons, two towers") or --w 1 vs 10 and compare the read-back caption and C-score.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch

from text2mario.app.service import default_ckpt
from text2mario.captions.detector import caption as read_caption
from text2mario.data.tiles import CHARS
from text2mario.diffusion.sampler import ddim_sample
from text2mario.eval import adherence as A
from text2mario.eval.generate import load_ema_conditional, load_exported
from text2mario.generate.framing import frame
from text2mario.playability.checker import Framing, check_level
from text2mario.render.renderer import grid_to_sprite_image, tensor_to_grid
from text2mario.text.normalizer import normalize
from text2mario.utils import appcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence", nargs="?", default="full floor with two pipes and a few enemies")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--w", type=float, default=None)
    a = ap.parse_args()
    cfg = appcfg.load()
    w = a.w if a.w is not None else cfg["cfg"]["default_w"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = default_ckpt()
    unet, sched, cond, _mcfg, _ = (load_exported(ck, dev) if ck.is_dir() else load_ema_conditional(ck, dev))
    print(f"1. INPUT      : {a.sentence!r}   (seed {a.seed}, guidance w = {w}, device {dev})")
    n = normalize(a.sentence)
    for i, s in enumerate(n.scenes, 1):
        print(f"2. NORMALIZER : scene {i}: raw topics {s.topics}  forbid {s.forbid}  notes {sorted(set(s.warnings))}")
    cap = n.scenes[0].caption if n.scenes else ""
    print(f"3. CAPTION    : {cap!r}  (the training grammar; this exact string is what the model was conditioned on)")
    toks = cond.tok(cap)["input_ids"]
    print(f"4. TOKENS     : {len(toks)} MiniLM word-pieces (limit 48): {cond.tok.convert_ids_to_tokens(toks)[:14]}...")
    ctx, mask = cond([cap])
    print(f"5. CONTEXT    : frozen MiniLM -> adapter -> tensor {tuple(ctx.shape)} (batch, tokens, 128-d), mask {tuple(mask.shape)}; null context for guidance: {tuple(cond.null(1)[0].shape)}")
    g = torch.Generator().manual_seed(a.seed)
    x_init = torch.randn((1, 13, 16, 16), generator=g)
    print(f"6. START NOISE: x_T ~ N(0,I), shape {tuple(x_init.shape)} = 13 tile-class channels x 16 rows x 16 columns, mean {x_init.mean():.3f}, std {x_init.std():.3f}")
    ctx_u, mask_u = cond.null(1)
    x, info = ddim_sample(unet, sched, (1, 13, 16, 16), ctx_c=ctx, mask_c=mask, ctx_u=ctx_u, mask_u=mask_u, w=w, steps=cfg["modes"]["full"]["steps"], x_init=x_init,
                          device=dev, autocast_dtype=torch.bfloat16 if dev == "cuda" else None)
    print(f"7. DENOISING  : {info['steps']} DDIM steps, eps = eps_u + w (eps_c - eps_u), x0 clipped to [-1,1] (clip rate {info['clip_rate']:.3f}); output {tuple(x.shape)} in [-1,1]")
    grid = tensor_to_grid(x)[0]
    print("8. ARGMAX     : tile grid (rows 0-15, characters per ledger F6):")
    for r in range(16):
        print("      " + "".join(CHARS[v] for v in grid[r]))
    back = read_caption(grid)
    print(f"9. READ BACK  : the detector describes the sample as {back!r}")
    print(f"10. C-SCORE   : {A.c_score(cap, back):.3f}   exact match: {A.exact_match(cap, back)}   (null level scores ~0.46, an always-empty level ~0.50)")
    framed, fr = frame(grid)
    res = check_level(framed, mode="level", framing=Framing((fr['spawn']['col'], fr['spawn']['row']), fr['flag']['col']))
    print(f"11. FRAMING   : +3 spawn columns, +6 end-template columns -> {framed.shape[1]} columns; spawn {fr['spawn']}, flag column {fr['flag']['col']}")
    print(f"12. CHECKER   : {res.status}  ({len(res.path)} moves, {res.nodes} search nodes, {res.seconds*1000:.0f} ms): {[m[2] for m in res.path][:8]}")
    out = ROOT / "reports/app"
    out.mkdir(parents=True, exist_ok=True)
    grid_to_sprite_image(framed).resize((framed.shape[1] * 36, 16 * 36)).save(out / "trace.png")
    print("13. OUTPUT    : reports/app/trace.png (sprite render); the app additionally builds the offline HTML game from the same grid")


if __name__ == "__main__":
    main()
