"""LevelService: typed sentence -> checked level -> exports (doc 28 §7).

Explain-me:
  What: loads the EMA model once (cached by the caller), turns a prompt into a level with `generate_level`, and builds the
        downloadable artifacts (PNG, JSON, CSV, VGLC text, playable HTML).  Picks "fast mode" (25 DDIM steps, no resampling)
        automatically without CUDA, and a time budget on the cloud profile.
  In/Out: LevelService(ckpt).run(text, seed, w, width, fast) -> RunResult(level, timings, files);  preview(text) -> Normalized.
  Previous: generate/full_level.py.  Next: app/streamlit_app.py (UI) and scripts/make_showcase.py / bench_app.py.
  Try: T2M_PROFILE=cloud makes the service stop after the time budget and report 'time budget' in verification.reason.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from text2mario.eval.generate import load_ema_conditional, load_exported
from text2mario.generate.full_level import generate
from text2mario.render.game_html import build_game_html
from text2mario.render.renderer import grid_to_csv, grid_to_vglc_text
from text2mario.render.schema import level_grid, to_png_bytes
from text2mario.text.normalizer import Normalized, normalize
from text2mario.utils import appcfg

ROOT = Path(__file__).resolve().parents[3]
_C = appcfg.load()                                              # configs/generate/app.yaml
FAST = dict(_C["modes"]["fast"])        # app default: seconds, not minutes (measured, reports/app/latency.json)
FULL = dict(_C["modes"]["full"])


@dataclass
class RunResult:
    level: dict
    seconds: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)


def default_ckpt() -> Path | None:
    """T2M_CKPT, else the exported bundle in app/model/, else app/model/ckpt.json -> {"path": ...}, else FROZEN.json's checkpoint, else None."""
    if os.environ.get("T2M_CKPT"):
        return Path(os.environ["T2M_CKPT"])
    if (ROOT / "app/model/t2m_ema.safetensors").exists():          # exported bundle (cloud / zip)
        return ROOT / "app/model"
    j = ROOT / "app/model/ckpt.json"
    if j.exists():
        return ROOT / json.loads(j.read_text())["path"]
    fz = ROOT / "FROZEN.json"
    if fz.exists():
        return ROOT / json.loads(fz.read_text())["checkpoint"]
    return None


MAX_PROMPT_CHARS = int(_C["validation"]["max_prompt_chars"])
VALID_WIDTHS = tuple(_C["widths"])


def validate_prompt(text) -> tuple[bool, str]:
    """Input validation for the UI and the service: (ok, message).  Never raises; the message is user-facing."""
    if not isinstance(text, str):
        return False, "The prompt must be text."
    t = text.strip()
    if not t:
        return False, "Please type a sentence describing the level (for example: 'full floor with two pipes')."
    if len(t) > MAX_PROMPT_CHARS:
        return False, f"That is {len(t)} characters; please keep the prompt under {MAX_PROMPT_CHARS}."
    if any(ord(c) < 32 and c not in "\n\t" for c in t):
        return False, "The prompt contains control characters; please type plain text."
    if re.search(r"(?<![\w])-\s?\d", t):
        return False, "Negative numbers do not make sense for a level (for example '-3 pipes'). Use 'no pipes' to rule something out."
    return True, ""


WARNING_TEXT = {      # normalizer warning code -> plain sentence for the person typing (the UI never shows raw codes)
    "BARE_PLURAL_DEFAULT": "You wrote a plural without a number, so a default amount was used.",
    "DUPLICATE_TOPIC": "The same thing was mentioned twice with different amounts; the last amount was used.",
    "EMPTY_SCENE": "Part of the sentence had no level words and was ignored.",
    "IMPLIED_COINS": "A coin line implies coins, so some coins were added to the request.",
    "NEGATION_FORBID": "Things you ruled out (no ...) are checked on the generated level.",
    "NEGATION_ONLY": "Only prohibitions were given, so a plain flat level is used as the starting point.",
    "POSITIONAL_IGNORED": "Words like left/right/top are not supported; the order of 'then' scenes is used instead.",
    "STYLE_HINT": "Difficulty words are noted but do not change the level yet.",
    "THEME_UNSUPPORTED": "Themes (castle, lava, snow...) are not supported and were ignored.",
    "TOO_MANY_SCENES": "Too many scenes; the last ones were merged.",
    "CANONICAL_PASSTHROUGH": "",
}


def explain_warning(code: str) -> str:
    """Plain-English version of a normalizer warning code ('' = nothing worth telling the user)."""
    if code.startswith("CLAMPED:"):
        _, topic, change = code.split(":", 2)
        return f"The model only knows some amounts of {topic}: {change.replace('->', ' became ')}."
    if code.startswith("CONTRADICTION:"):
        return f"The sentence asks for and rules out the same thing ({code.split(':', 1)[1]})."
    return WARNING_TEXT.get(code, code.replace("_", " ").lower())


def validate_settings(seed, w, width) -> tuple[bool, str]:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**53 - 1:
        return False, "The seed must be an integer between 0 and 2^53-1."
    if not isinstance(w, (int, float)) or isinstance(w, bool) or not float(_C["cfg"]["min_w"]) <= float(w) <= float(_C["cfg"]["max_w"]):
        return False, "The prompt strength (CFG w) must be between {} and {}.".format(_C["cfg"]["min_w"], _C["cfg"]["max_w"]) + ""
    if width not in VALID_WIDTHS:
        return False, f"The width must be one of {VALID_WIDTHS}."
    return True, ""


class LevelService:
    def __init__(self, ckpt: str | Path | None = None, device: str | None = None):
        self.ckpt = Path(ckpt) if ckpt else default_ckpt()
        if self.ckpt is None or not self.ckpt.exists():
            raise FileNotFoundError("no model checkpoint: set T2M_CKPT or write app/model/ckpt.json")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.profile = os.environ.get("T2M_PROFILE", "local")
        t0 = time.time()
        # a directory is an exported bundle (scripts/export_cloud_ckpt.py); a file is a training checkpoint
        self.bundle = load_exported(self.ckpt, self.device) if self.ckpt.is_dir() else load_ema_conditional(self.ckpt, self.device)
        self.load_seconds = time.time() - t0

    @property
    def fast_forced(self) -> bool:
        return self.device != "cuda"

    @staticmethod
    def preview(text: str) -> Normalized:
        return normalize(text)

    def run(self, text: str, *, seed: int, w: float = 5.0, width: int = 48, fast: bool = False, verify: bool = True,
            time_budget_s: float | None = None) -> RunResult:
        for ok, msg in (validate_prompt(text), validate_settings(seed, w, width)):
            if not ok:
                raise ValueError(msg)
        s = FAST if (fast or self.fast_forced) else FULL
        budget = time_budget_s if time_budget_s is not None else (float(_C["cloud"]["time_budget_s"]) if self.profile == "cloud" else None)
        t0 = time.time()
        lvl = generate(self.bundle, text, seed=seed, w=w, width=width, steps=s["steps"], resample_r=s["resample_r"],
                       device=self.device, verify=verify, profile=self.profile, time_budget_s=budget, checker_budget=s["checker_budget"])
        t_gen = time.time() - t0
        t1 = time.time()
        files = {"png": to_png_bytes(lvl), "json": json.dumps(lvl, indent=1).encode(),
                 "csv": grid_to_csv(level_grid(lvl)).encode(), "vglc": (grid_to_vglc_text(level_grid(lvl)) + "\n").encode(),
                 "html": build_game_html(lvl).encode()}
        return RunResult(lvl, {"generate_and_verify": round(t_gen, 2), "render": round(time.time() - t1, 2),
                               "total": round(time.time() - t0, 2)}, files)
