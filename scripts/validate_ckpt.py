"""Validate a training checkpoint on the dev windows with EMA weights (doc 16 §10; the monitoring tool for Phase 3).

Explain-me:
  What: loads a step_*.pt checkpoint, swaps in the EMA weights, and computes the deterministic dev loss per t-bucket
        (same fixed noise every time, so numbers are comparable across checkpoints).  Prints it next to the
        "context-free floor" of the schedule (doc 13 §5): a model that really uses spatial context ends BELOW the floor.
  In/Out: python scripts/validate_ckpt.py runs/<run>/step_0005000.pt  -> table + <ckpt>.val.json
  Rule (doc 13 §5): after 5k steps, if mean dev loss is not below the floor mean, stop and debug; do not train longer.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.utils.env import bootstrap_env  # noqa: E402

bootstrap_env()
from text2mario.data.dataset import DataConfig, MarioWindows  # noqa: E402
from text2mario.diffusion.scheduler import DiffusionSchedule  # noqa: E402
from text2mario.models.factory import build_unet  # noqa: E402
from text2mario.train import checkpoint as ckpt  # noqa: E402
from text2mario.train.validate import Validator  # noqa: E402

# Context-free floors per 100-step bucket (doc 13 §5): best eps-MSE of a denoiser that sees each pixel alone.
FLOOR = {"linear": [0.0000, 0.0091, 0.0299, 0.0218, 0.0101, 0.0037, 0.0011, 0.0003, 0.0001, 0.0000],
         "cosine": [0.0000, 0.0000, 0.0026, 0.0176, 0.0303, 0.0290, 0.0205, 0.0112, 0.0043, 0.0007]}
FLOOR_MEAN = {"linear": 0.0076, "cosine": 0.0116}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    s = ckpt.load(path)
    cfg = json.loads(s["config_json"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conditional = bool(cfg["text"]["enabled"])
    unet = build_unet(cfg["model"], conditional=conditional).to(dev)
    unet.load_state_dict(s["unet"])
    cond = None
    if conditional:
        from text2mario.text.conditioner import TextConditioner
        cond = TextConditioner(cfg["text"]["encoder_dir"], cfg["data"]["max_len"], cfg["text"]["adapter_hidden"],
                               cfg["model"]["context_dim"], cfg["text"]["cache_max_mb"], dev)
        cond.check_probe(s["text"]["probe"])
        cond.adapter.load_state_dict(s["text"]["adapter"])
    modules = {"unet": unet, **({"adapter": cond.adapter} if cond else {})}
    with torch.no_grad():                                       # swap in the EMA weights (evaluation always uses EMA)
        for k, m in modules.items():
            for n, p in m.named_parameters():
                p.copy_(s["ema"]["shadow"][f"{k}.{n}"].to(p.dtype))
    sched = DiffusionSchedule(cfg["diffusion"]["schedule"], cfg["diffusion"]["T"], dev)
    data = MarioWindows("dev", device=dev, cfg=DataConfig(seed=cfg["seed"]))
    res = Validator(data, sched, device=dev).run(unet, cond)
    kind = cfg["diffusion"]["schedule"]
    print(f"{path.name}  step {s['step']}  schedule {kind}  ({'conditional' if conditional else 'unconditional'})")
    print("bucket   t-range     val_loss   floor    below floor?")
    for k in range(10):
        v, f = res[f"val/loss_t{k}"], FLOOR[kind][k]
        print(f"  {k}     {k * 100:3d}-{k * 100 + 99:3d}    {v:.5f}   {f:.4f}   {'yes' if v < f else 'no'}")
    print(f"mean dev loss {res['val/loss']:.5f}   (context-free floor mean {FLOOR_MEAN[kind]})   "
          f"{'BELOW floor: learning spatial structure' if res['val/loss'] < FLOOR_MEAN[kind] else 'NOT below the floor yet'}")
    if conditional:
        print("conditioning gain (null loss - cond loss) per bucket:", [round(res[f'val/gain_t{k}'], 5) for k in range(10)])
    path.with_suffix(".val.json").write_text(json.dumps({"step": s["step"], **res}, indent=2) + "\n")


if __name__ == "__main__":
    main()
