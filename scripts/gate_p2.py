"""Gate G-P2 (scheduler): oracle DDIM exact, alpha-bar bounds, x_999 moments, corruption strips (doc 13, ledger §9).

Explain-me:
  What: numerically checks both noise schedules on REAL training windows and draws what noising does to a level
        at t = 0, 250, 500, 750, 999 (argmax of x_t).  The linear schedule is mostly noise by t=500; cosine is not.
  In/Out: needs gate P1 passed -> reports/gates/P2/{RESULT.json, strips_<schedule>.png}.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data.dataset import MarioWindows  # noqa: E402
from text2mario.diffusion.scheduler import DiffusionSchedule  # noqa: E402
from text2mario.render.renderer import grid_to_image, tensor_to_grid  # noqa: E402
from text2mario.utils.gates import gate_dir, require, write_result  # noqa: E402

checks, metrics = [], {}


def check(name, ok, detail=""):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    require("P1")
    d = MarioWindows("train", cfg=__import__("text2mario.data.dataset", fromlist=["DataConfig"]).DataConfig(dtype="float64"))
    g = torch.Generator().manual_seed(0)
    rows = torch.randperm(len(d), generator=g)
    x0 = d.onehot(rows[:512])
    out = gate_dir("P2")
    for kind, ab_bound in (("linear", 1e-4), ("cosine", 1e-8)):
        s = DiffusionSchedule(kind)
        ab_T = s.buf["alphas_cumprod"][-1].item()
        check(f"{kind}: alpha_bar_T < {ab_bound}", ab_T < ab_bound, f"= {ab_T:.3e}")
        xt = s.q_sample(x0, torch.full((512,), 999), torch.randn(x0.shape, generator=g, dtype=x0.dtype))
        check(f"{kind}: x_999 mean within 0.02 and std within 1 +- 0.02", abs(xt.mean()) < 0.02 and abs(xt.std() - 1) < 0.02,
              f"(mean {xt.mean():+.4f}, std {xt.std():.4f})")
        ts = s.ddim_timesteps(50)
        x = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
        for i, t in enumerate(ts):
            tt = t.repeat(len(x))
            nxt = (ts[i + 1] if i + 1 < len(ts) else torch.tensor(-1)).repeat(len(x))
            x = s.ddim_step(s.predict_eps_from_x0(x, tt, x0), tt, nxt, x, eta=0.0)   # oracle epsilon
        mse = torch.nn.functional.mse_loss(x, x0).item()
        agree = (x.argmax(1) == x0.argmax(1)).float().mean().item()
        check(f"{kind}: oracle-eps DDIM reconstructs x0 (fp64 MSE < 1e-6, argmax 100 %)", mse < 1e-6 and agree == 1.0,
              f"(mse {mse:.2e}, argmax agreement {agree:.0%})")
        metrics[kind] = {"alpha_bar_T": ab_T, "oracle_mse": mse, "oracle_argmax": agree}
        # corruption strips for 4 windows x 5 timesteps
        tl, tile = [0, 250, 500, 750, 999], 6
        wins = x0[:4]
        sheet = Image.new("RGB", (len(tl) * (16 * tile + 4) + 4, 4 * (16 * tile + 4) + 18), "white")
        dr = ImageDraw.Draw(sheet)
        for j, t in enumerate(tl):
            dr.text((8 + j * (16 * tile + 4), 3), f"t={t}", fill=(0, 0, 0))
        for i in range(4):
            for j, t in enumerate(tl):
                xt = s.q_sample(wins[i:i + 1], torch.tensor([t]), torch.randn(wins[i:i + 1].shape, generator=g, dtype=x0.dtype))
                im = grid_to_image(tensor_to_grid(xt)[0], tile, glyphs=False)
                sheet.paste(im, (4 + j * (16 * tile + 4), 18 + i * (16 * tile + 4)))
        sheet.save(out / f"strips_{kind}.png")
    passed = all(c["pass"] for c in checks)
    write_result("P2", passed, metrics, checks)
    print("G-P2", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
