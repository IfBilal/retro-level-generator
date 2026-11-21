"""Full-level generation: scenes -> windows -> outpainting -> framing -> sanitizer -> checker -> resample (doc 20, G1-G5).

Explain-me:
  What: turns a typed sentence into a complete, framed, checked level.  The prompt is normalised into scenes; the content is
        generated as 16x16 windows with stride 8, each window keeping the right 8 columns of the previous one (RePaint-style
        known-region replacement); the result is framed, scanned for broken pipes/cannons, and judged by the playability
        checker.  If the checker fails, the window where the search got stuck is re-sampled with a new derived seed
        (max 5 attempts per window, 20 per level).  Same prompt + seed + settings -> same level on the same machine.
  In/Out: generate(bundle, text, seed=..., width=48, ...) -> level dict (doc 29) with raw_grid, display_grid, verification.
  Previous: text/normalizer.py, eval/generate.load_ema_conditional.  Next: render/schema.py (exports), app/streamlit_app.py.
  Try: width=16 (one window) vs 112 (13 windows), or resample_r=0 to see how visible the seams become.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from text2mario.captions.detector import caption as detect_caption
from text2mario.data import tiles as T
from text2mario.diffusion.sampler import ddim_sample
from text2mario.generate.framing import frame
from text2mario.generate.sanitizer import detect as detect_broken
from text2mario.generate.sanitizer import repair
from text2mario.playability.checker import Framing, check_level
from text2mario.playability.physics import Physics
from text2mario.render.renderer import grid_to_tensor, tensor_to_grid
from text2mario.render.schema import make_level, validate_level
from text2mario.text.normalizer import normalize
from text2mario.utils import appcfg

_C = appcfg.load()
WINDOW, STRIDE = int(_C["window"]), int(_C["stride"])     # configs/generate/app.yaml (drill 5)


def overlap() -> int:
    """Columns a window shares with the previous one (the known region)."""
    return WINDOW - STRIDE
FORBID_WORDS = {"enemy": "enem", "pipe": "pipe", "cannon": "cannon", "coin": "coin", "platform": "platform",
                "question block": "question block", "loose block": "loose block", "tower": "tower"}


def n_windows(width: int) -> int:
    return (width - WINDOW) // STRIDE + 1


def window_seed(seed: int, k: int, attempt: int) -> int:
    """Deterministic CPU seed for (level seed, window, attempt): independent of the GPU RNG (NFR-REP-04)."""
    return int(hashlib.sha256(f"{seed}:{k}:{attempt}".encode()).hexdigest()[:12], 16)


def scene_for_windows(captions: list[str], n: int) -> list[int]:
    """Spread the scenes over n windows in order, each scene covering a contiguous run (doc 20 §3.1)."""
    s = len(captions)
    return [min(s - 1, k * s // n) for k in range(n)]


@torch.no_grad()
def sample_window(bundle, caption: str, w: float, steps: int, r: int, gen_seed: int, known_cols: np.ndarray | None,
                  device: str) -> np.ndarray:
    """One 16x16 window.  known_cols: uint8 (16, 8) = the previous window's right half (fixed left half), or None."""
    unet, sched, cond = bundle[0], bundle[1], bundle[2]
    g = torch.Generator().manual_seed(gen_seed)
    ctx_c, mask_c = cond([caption])
    ctx_u, mask_u = cond.null(1)
    known = None
    if known_cols is not None:
        full = np.zeros((WINDOW, WINDOW), np.uint8)
        full[:, :overlap()] = known_cols
        x0 = grid_to_tensor(full)[None]                           # (1, 13, 16, 16) in {-1, +1}
        m = torch.zeros(1, 1, WINDOW, WINDOW)
        m[..., :overlap()] = 1.0
        known = (x0, m)
    x, _ = ddim_sample(unet, sched, (1, 13, WINDOW, WINDOW), ctx_c=ctx_c, mask_c=mask_c, ctx_u=ctx_u, mask_u=mask_u, w=w,
                       steps=steps, generator=g, known=known, resample_r=r if known is not None else 0, device=device,
                       autocast_dtype=torch.bfloat16 if device == "cuda" else None)
    return tensor_to_grid(x)[0]


def _assemble(windows: list[np.ndarray], width: int) -> np.ndarray:
    out = np.zeros((16, width), np.uint8)
    for k, wg in enumerate(windows):
        out[:, k * STRIDE:k * STRIDE + WINDOW] = wg                # later windows overwrite the (identical) known half
    return out


def _forbidden_hit(grid: np.ndarray, forbid: list[str]) -> bool:
    cap = detect_caption(grid)
    return any(FORBID_WORDS.get(f, f) in cap for f in forbid)


def generate(bundle, text: str, *, seed: int, w: float = 5.0, width: int = 48, steps: int = 50, resample_r: int = 2,
             device: str = "cuda", verify: bool = True, physics: Physics | None = None, max_window_attempts: int = int(_C["verify"]["max_window_attempts"]),
             max_total_attempts: int = int(_C["verify"]["max_total_attempts"]), time_budget_s: float | None = None, profile: str = "local",
             meta: dict | None = None, enemies: str = "ignore", checker_budget: int = 20000) -> dict:
    """Prompt -> level dict.  `bundle` = eval.generate.load_ema_conditional(...) result (unet, schedule, cond, cfg, state)."""
    t0 = time.time()
    assert width in _C["widths"], f"widths offered: {_C['widths']} (G2, configs/generate/app.yaml)"
    norm = normalize(text)
    scenes = [s.caption for s in norm.scenes] or [""]            # an empty prompt generates an unconditioned level
    forbid = sorted({f for s in norm.scenes for f in s.forbid})
    n = n_windows(width)
    scene_of = scene_for_windows(scenes, n)
    ph = physics or Physics.load()
    attempts = [0] * n
    windows: list[np.ndarray] = [None] * n
    attempt_log, total = [], 0

    def regen_from(k0: int):
        for k in range(k0, n):
            known = windows[k - 1][:, STRIDE:] if k > 0 else None      # the previous window's last (window - stride) columns
            windows[k] = sample_window(bundle, scenes[scene_of[k]], w, steps, resample_r, window_seed(seed, k, attempts[k]), known, device)

    regen_from(0)
    status, reason, checker_info = "not_verified", None, None
    framed, framing = None, None
    while True:
        content = _assemble(windows, width)
        framed, framing = frame(content)
        fr = Framing((framing["spawn"]["col"], framing["spawn"]["row"]), framing["flag"]["col"])
        res = check_level(framed, mode="level", framing=fr, physics=ph, enemies=enemies, budget_nodes=checker_budget) if verify else None
        forb_k = next((k for k in range(n) if forbid and _forbidden_hit(windows[k], forbid)), None)
        if not verify:
            reason = "verification skipped"
            break
        checker_info = {"tier": 1, "physics_sha256": ph.sha256, "seconds": round(res.seconds, 3), "path_len": len(res.path),
                        "enemy_mode": enemies}
        if res.status == "solved" and forb_k is None:
            status = "verified"
            break
        # which window to re-sample: the one containing the obstacle after the furthest column reached (or a forbidden hit)
        if res.status != "solved":
            cc = max(0, min(width - 1, (res.fail_col or 0) - 3 + 2))
            k = min(n - 1, cc // STRIDE)
            kind = res.status
        else:
            k, kind = forb_k, "forbidden"
        attempts[k] += 1
        total += 1
        attempt_log.append({"window": int(k), "attempt": int(attempts[k]), "status": kind, "fail_col": int(res.fail_col or 0)})
        if attempts[k] > max_window_attempts or total > max_total_attempts or (time_budget_s and time.time() - t0 > time_budget_s):
            reason = "attempt budget exhausted" if not (time_budget_s and time.time() - t0 > time_budget_s) else "time budget"
            attempt_log[-1]["attempt"] = int(min(attempts[k], max_window_attempts))
            break
        regen_from(k)
    content = _assemble(windows, width)
    framed, framing = frame(content)
    san = detect_broken(content)
    display, removed = repair(framed)
    san.update({"repaired": bool(removed), "removed_cells": int(removed)})
    prompt = {"text": text, "canonical": [[p.strip() for p in s.split(".") if p.strip()] for s in scenes],
              "ignored_tokens": norm.ignored_tokens, "warnings": norm.warnings, "forbid": forbid,
              "scenes": [{"caption": scenes[scene_of[k]], "col_from": 3 + k * STRIDE, "col_to": 3 + k * STRIDE + WINDOW - 1}
                         for k in range(n)]}
    cfg, state = (bundle[3], bundle[4]) if bundle is not None and len(bundle) >= 5 else ({}, {})
    generation = {"seed": int(seed), "w": float(w), "width": int(width), "ddim_steps": int(steps), "resample_r": int(resample_r),
                  "fast_mode": bool(steps < 50), "profile": profile, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "checkpoint_step": int(state.get("step", -1)), "seconds": round(time.time() - t0, 2), **(meta or {})}
    verification = {"status": status, "reason": reason, "attempts_total": int(total), "attempt_log": attempt_log, "checker": checker_info}
    lvl = make_level(framed, display_grid=display, prompt=prompt, generation=generation, framing=framing, sanitizer=san,
                     verification=verification, provenance={"source": "app", "dataset_level_id": None, "dataset_level_name": None})
    return validate_level(lvl)
