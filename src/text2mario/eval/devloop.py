"""Dev-set evaluation of a conditional checkpoint: C-score, exact match, structure metrics, per-concept table (doc 22 §6.1-6.3).

Explain-me:
  What: samples every prompt of P-dev (or P-rand) with several seeds at a CFG scale w, reads captions back with the
        detector, and scores prompt adherence (C-score), broken structures, floor validity, novelty and copy rate.
        Only dev prompt sets are accepted here; the test prompts are touched by scripts/eval_test.py alone (CLAUDE.md rule 1).
  In/Out: evaluate(ckpt, prompt_set, w, seeds) -> dict of metrics (+ per-prompt rows for bootstrap / error analysis).
  Previous: eval/generate.py (sampling), eval/adherence.py (C-score), eval/structure.py.  Next: scripts/eval_dev.py, CFG sweep.
  Try: w=1 vs w=5 on the same checkpoint and watch c_score and broken_rate move in opposite directions (doc 15).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from text2mario.captions.detector import caption
from text2mario.data.dataset import MarioWindows
from text2mario.eval import adherence as A
from text2mario.eval import structure as S
from text2mario.eval.generate import load_ema_conditional, sample_conditional

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = {"p_dev": "data/prompts/p_dev.json", "p_rand": "data/prompts/p_rand.json"}
SEED_BASE = {"p_dev": 6_000_000, "p_rand": 7_000_000}   # SET_BASE (doc 22 §2.3); seeds = base + 1000 * prompt + seed index


def load_prompts(name: str) -> list[dict]:
    if name not in ALLOWED:
        raise PermissionError(f"{name!r} is not a dev prompt set; test prompts are only used by scripts/eval_test.py")
    return json.loads((ROOT / ALLOWED[name]).read_text())


_TRAIN = None


def train_grids() -> np.ndarray:
    global _TRAIN
    if _TRAIN is None:
        _TRAIN = MarioWindows("train").grids.numpy()
    return _TRAIN


def evaluate(model_bundle, prompt_set: str, w: float, n_seeds: int = 3, steps: int = 50, n_prompts: int | None = None,
             device: str = "cuda", with_novelty: bool = True) -> dict:
    """Dev prompt sets only (p_dev / p_rand).  model_bundle = load_ema_conditional(ckpt, device) result."""
    rows = load_prompts(prompt_set)[:n_prompts]
    return evaluate_rows(model_bundle, rows, w, SEED_BASE[prompt_set], prompt_set, n_seeds, steps, device, with_novelty)


def evaluate_rows(model_bundle, rows: list[dict], w: float, seed_base: int, prompt_set: str, n_seeds: int = 3, steps: int = 50,
                  device: str = "cuda", with_novelty: bool = True) -> dict:
    """The shared metric code.  scripts/eval_test.py calls this directly (the ONLY place test prompts are used)."""
    unet, sched, cond, cfg, state = model_bundle
    prompts = [r["prompt"] for r in rows for _ in range(n_seeds)]
    seeds = [seed_base + 1000 * i + k for i in range(len(rows)) for k in range(n_seeds)]
    grids = sample_conditional(unet, sched, cond, prompts, w, seeds, steps, device=device,
                               autocast_dtype=torch.bfloat16 if device == "cuda" else None)
    caps = [caption(g) for g in grids]
    cs = np.array([A.c_score(p, c) for p, c in zip(prompts, caps)])
    em = np.array([A.exact_match(p, c) for p, c in zip(prompts, caps)])
    out = {"step": state["step"], "prompt_set": prompt_set, "w": w, "n_prompts": len(rows), "n_seeds": n_seeds,
           "n_samples": len(grids), "c_score": float(cs.mean()), "exact_match": float(em.mean()),
           "broken_rate": S.broken_rate(grids), "floor_valid": S.floor_valid_rate(grids),
           "floorless_share": S.floorless_share(grids)}
    rep = A.concept_report(list(zip(prompts, caps)))
    out["rare_exact_match"] = A.rare_exact_match(rep)
    out["concepts"] = rep
    nn = None
    if with_novelty:
        nov = S.novelty(grids, train_grids())
        nn = nov["nn"]
        out.update({"novelty_med": nov["novelty_med"], **S.copy_rates(nov["nn"]), "tile_js": S.tile_js(grids, train_grids())})
    # per-sample facts (for level-clustered confidence intervals on EVERY metric, CLAUDE.md rule 16)
    out["per_sample"] = [{"pid": rows[i // n_seeds]["pid"], "level": (rows[i // n_seeds].get("source") or {}).get("level"),
                          "c_score": float(cs[i]), "exact_match": bool(em[i]),
                          "broken": bool(any(S.broken_counts(grids[i])[:2])), "floor_valid": bool(S.valid_floor(grids[i])),
                          "nn_hamming": (float(nn[i]) if nn is not None else None)} for i in range(len(grids))]
    out["per_prompt"] = [{"pid": r["pid"], "level": (r.get("source") or {}).get("level"),
                          "c_score": float(cs[i * n_seeds:(i + 1) * n_seeds].mean()),
                          "exact_match": float(em[i * n_seeds:(i + 1) * n_seeds].mean())} for i, r in enumerate(rows)]
    out["_grids"] = grids
    out["_captions"] = caps
    return out
