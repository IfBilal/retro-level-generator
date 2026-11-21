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
from text2mario.diffusion.sampler import ddim_sample
from text2mario.generate import adherence
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


@torch.no_grad()
def sample_window_batch(bundle, caption: str, w: float, steps: int, r: int, gen_seed: int, known_cols: np.ndarray | None,
                        device: str, n: int) -> np.ndarray:
    """n candidate windows for the SAME caption and known columns in ONE forward pass -> uint8 (n, 16, 16).  On a GPU this costs little more than one
    window, so the adherence loop can pick an exact one from many instead of redrawing one at a time.  Deterministic for (seed, n)."""
    unet, sched, cond = bundle[0], bundle[1], bundle[2]
    g = torch.Generator().manual_seed(gen_seed)
    ctx_c, mask_c = cond([caption] * n)
    ctx_u, mask_u = cond.null(n)
    known = None
    if known_cols is not None:
        full = np.zeros((WINDOW, WINDOW), np.uint8)
        full[:, :overlap()] = known_cols
        x0 = grid_to_tensor(full)[None]
        m = torch.zeros(1, 1, WINDOW, WINDOW)
        m[..., :overlap()] = 1.0
        known = (x0, m)
    x, _ = ddim_sample(unet, sched, (n, 13, WINDOW, WINDOW), ctx_c=ctx_c, mask_c=mask_c, ctx_u=ctx_u, mask_u=mask_u, w=w, steps=steps,
                       generator=g, known=known, resample_r=r if known is not None else 0, device=device,
                       autocast_dtype=torch.bfloat16 if device == "cuda" else None)
    return tensor_to_grid(x)


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
    if adherence.enabled():
        adherence.check_feasible(norm, width)                       # contradiction / impossible count -> ValueError with a clear message
    use_adh = adherence.enabled() and bool(norm.scenes)
    adh_verify = use_adh and verify and bool(adherence._A["verify"])      # verify=False skips every check, including this one
    adh_max = int(adherence._A["max_attempts"])
    adh_tries = int(adherence._A["max_window_tries"])
    adh_batch = int(adherence._A["batch_tries"])                    # candidates per window in one GPU pass (1 = off)
    adh_batch_cpu = int(adherence._A["batch_tries_cpu"])            # extra candidates a CPU draws when its first draw missed (1 = off -> sequential redraws)
    adh_draw_cap = int(adherence._A["max_extra_draws"])             # hard, deterministic cap on window draws spent on counts (bounds latency)
    local_resamples = 0
    adh_draws = 0                                                   # extra window draws caused by adherence (local redraws + whole-level resamples)
    bounds = adherence.region_bounds(scene_of, width) if use_adh else []
    ph = physics or Physics.load()
    attempts = [0] * n                                             # seed index per window (any reason to resample bumps it)
    play_att = [0] * n                                             # playability / forbidden resamples per window (ledger G5 limit)
    windows: list[np.ndarray] = [None] * n
    win_caps: list[str] = [None] * n
    win_plans: list[dict] = [None] * n
    forb_sets = adherence.scene_forbids(norm) if use_adh else []
    attempt_log, total, adh_total = [], 0, 0

    def caption_for(k: int):
        """(caption, plan) of window k: the scene caption, or (adherence on) only what the scene still needs this window to add."""
        si = scene_of[k]
        if not use_adh:
            return scenes[si], {}
        ks = [j for j, sj in enumerate(scene_of) if sj == si]
        before = _assemble(windows[:k], STRIDE * (k - 1) + WINDOW) if k > 0 else None
        return adherence.plan_caption(norm.scenes[si], scenes[si], k, ks[0], ks[-1], before, bounds[si][0])

    def regen_from(k0: int):
        nonlocal local_resamples, adh_draws
        for k in range(k0, n):
            known = windows[k - 1][:, STRIDE:] if k > 0 else None      # the previous window's last (window - stride) columns
            win_caps[k], plan = caption_for(k)
            win_plans[k] = plan
            scene_k = norm.scenes[scene_of[k]] if use_adh else None
            forb_k = forb_sets[scene_of[k]] if use_adh else None
            checked = bool(use_adh and adh_verify and adherence.has_constraints(scene_k, forb_k))
            tries = adh_tries if checked else 1
            best, best_miss = None, None
            n_batch = adh_batch if device == "cuda" else adh_batch_cpu
            if checked and n_batch > 1 and not (time_budget_s and time.time() - t0 > time_budget_s):
                if device == "cuda":                                   # a GPU draws the whole batch for the price of one window
                    cands = list(sample_window_batch(bundle, win_caps[k], w, steps, resample_r, window_seed(seed, k, attempts[k]), known, device, n_batch))
                else:                                                  # a CPU pays per candidate: one draw first, a small batch only if it missed
                    first = sample_window(bundle, win_caps[k], w, steps, resample_r, window_seed(seed, k, attempts[k]), known, device)
                    cands = [first]
                    if adherence.window_miss(first, k, scene_k, plan, forb_k) > 0:
                        cands += list(sample_window_batch(bundle, win_caps[k], w, steps, resample_r, window_seed(seed, k, attempts[k]) + 1, known,
                                                          device, n_batch))
                misses = [adherence.window_miss(c, k, scene_k, plan, forb_k) for c in cands]
                j = int(np.argmin(misses))                             # the first candidate with the fewest misses (0 = exact)
                best, best_miss = cands[j], misses[j]
                if len(cands) > 1 or best_miss > 0:
                    local_resamples += 1
                    adh_draws += 1                                     # one batch = one unit of the extra-draw budget
                tries = 0
            for a in range(tries):                                     # local rejection sampling: only THIS window is redrawn
                seed_idx = attempts[k] * tries + a if use_adh else attempts[k]
                win = sample_window(bundle, win_caps[k], w, steps, resample_r, window_seed(seed, k, seed_idx), known, device)
                miss = adherence.window_miss(win, k, scene_k, plan, forb_k) if tries > 1 else 0
                if best is None or miss < best_miss:
                    best, best_miss = win, miss                        # keep the closest draw if none is exact
                if miss == 0 or adh_draws >= adh_draw_cap or (time_budget_s and time.time() - t0 > time_budget_s):
                    break
                local_resamples += 1
                adh_draws += 1
            windows[k] = best

    regen_from(0)
    status, reason, checker_info = "not_verified", None, None
    play_done = False                                              # route budget (ledger G5) spent: only counts are still being fixed
    framed, framing = None, None
    while True:
        content = _assemble(windows, width)
        framed, framing = frame(content)
        fr = Framing((framing["spawn"]["col"], framing["spawn"]["row"]), framing["flag"]["col"])
        res = check_level(framed, mode="level", framing=fr, physics=ph, enemies=enemies, budget_nodes=checker_budget) if verify else None
        forb_k = next((k for k in range(n) if forbid and not use_adh and _forbidden_hit(windows[k], forbid)), None)
        if not verify:
            reason = "verification skipped"
            break
        checker_info = {"tier": 1, "physics_sha256": ph.sha256, "seconds": round(res.seconds, 3), "path_len": len(res.path),
                        "enemy_mode": enemies}
        out_of_time = bool(time_budget_s and time.time() - t0 > time_budget_s)
        adh_now = adherence.evaluate(content, norm, scene_of, width) if use_adh else None
        adh_pending = bool(adh_verify and adh_now["unmet"] and adh_total < adh_max and adh_draws < adh_draw_cap and not out_of_time)
        if res.status == "solved" and forb_k is None and not adh_pending:
            status = "verified"
            reason = None
            if adh_now and adh_now["unmet"]:                       # budget ended with a count still unmet: say so (verification.adherence)
                reason = "adherence budget exhausted" if not out_of_time else "time budget"
            break
        # what to re-sample next.  Route problems first (the window holding the obstacle after the furthest column reached), then forbidden words
        # (legacy path), then unmet counts.  Once the ROUTE budget is spent (play_done) the loop keeps fixing counts: the level cannot get a
        # route badge any more, but the sentence can still be honoured (audit: every failing 'underground' level had stopped here).
        if res.status != "solved" and not play_done:
            cc = max(0, min(width - 1, (res.fail_col or 0) - 3 + 2))
            k = min(n - 1, cc // STRIDE)
            kind = res.status
        elif forb_k is not None and not play_done:
            k, kind = forb_k, "forbidden"
        elif adh_pending:
            k, kind = adherence.window_to_resample(adh_now["unmet"][0], scene_of, n, win_plans), "adherence"
        else:                                                      # route budget spent and nothing left to fix for the counts
            reason = reason or ("time budget" if out_of_time else "attempt budget exhausted")
            break
        attempts[k] += 1
        if kind == "adherence":
            adh_total += 1
            adh_draws += n - k                                      # every window from k on is drawn again
            attempt_log.append({"window": int(k), "attempt": int(attempts[k]), "status": kind, "fail_col": 0,
                                "unmet": [f"{r['topic']} want {r['lo']}-{r['hi']} got {r['got']}" for r in adh_now["unmet"]]})
        else:
            total += 1
            play_att[k] += 1
            attempt_log.append({"window": int(k), "attempt": int(attempts[k]), "status": kind, "fail_col": int(res.fail_col or 0)})
        if out_of_time:
            reason = "time budget"
            attempt_log[-1]["attempt"] = int(min(attempts[k], max_window_attempts))
            break
        if kind != "adherence" and (play_att[k] > max_window_attempts or total > max_total_attempts):
            reason = "attempt budget exhausted"
            attempt_log[-1]["attempt"] = int(min(attempts[k], max_window_attempts))
            if not adh_pending:
                break
            play_done = True                                       # keep the level as it is for the route, go on fixing counts
            continue
        regen_from(k)
    content = _assemble(windows, width)
    framed, framing = frame(content)
    san = detect_broken(content)
    display, removed = repair(framed)
    san.update({"repaired": bool(removed), "removed_cells": int(removed)})
    prompt = {"text": text, "canonical": [[p.strip() for p in s.split(".") if p.strip()] for s in scenes],
              "ignored_tokens": norm.ignored_tokens, "warnings": norm.warnings, "forbid": forbid,
              "scenes": [{"caption": win_caps[k], "scene": int(scene_of[k]), "plan": win_plans[k], "col_from": 3 + k * STRIDE, "col_to": 3 + k * STRIDE + WINDOW - 1}
                         for k in range(n)]}
    _cfg, state = (bundle[3], bundle[4]) if bundle is not None and len(bundle) >= 5 else ({}, {})
    generation = {"seed": int(seed), "w": float(w), "width": int(width), "ddim_steps": int(steps), "resample_r": int(resample_r),
                  "fast_mode": bool(steps < 50), "profile": profile, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "checkpoint_step": int(state.get("step", -1)), "seconds": round(time.time() - t0, 2), **(meta or {})}
    adh_final = adherence.evaluate(content, norm, scene_of, width) if use_adh else {"status": "not_applicable", "constraints": [], "unmet": []}
    # the counts are judged on the RAW content (measure-before-repair); the display repair below never feeds this.
    # attempts_total is capped at the ledger limit: the attempt that hits the cap is logged but never executed.
    verification = {"status": status, "reason": reason, "attempts_total": int(min(total, max_total_attempts)), "attempt_log": attempt_log, "checker": checker_info,
                    "adherence": {"status": adh_final["status"], "attempts": int(adh_total), "window_redraws": int(local_resamples), "extra_draws": int(adh_draws), "draw_cap": adh_draw_cap, "planned": bool(use_adh and adherence._A["plan"]),
                                  "verified_by_resampling": bool(adh_verify), "judged_on": "raw", "constraints": adh_final["constraints"]}}
    lvl = make_level(framed, display_grid=display, prompt=prompt, generation=generation, framing=framing, sanitizer=san,
                     verification=verification, provenance={"source": "app", "dataset_level_id": None, "dataset_level_name": None})
    return validate_level(lvl)
