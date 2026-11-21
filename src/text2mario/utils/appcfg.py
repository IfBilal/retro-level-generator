"""Reads configs/generate/app.yaml: window/stride, widths, fast/full modes, verify budgets, validation, cloud budget, examples.

Explain-me:
  What: one place for every number the generator, the service and the app use (CLAUDE.md rule 4).  Cached; call load(refresh=True) after editing.
  In/Out: load() -> dict.  Used by generate/full_level.py, app/service.py, app/streamlit_app.py, eval/seams.py.
  Try: set stride: 4 and generate a level: the number of windows and the seam positions change (drill 5).
"""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[3] / "configs" / "generate" / "app.yaml"
_CACHE: dict = {}


def load(refresh: bool = False) -> dict:
    if refresh or not _CACHE:
        _CACHE.clear()
        _CACHE.update(yaml.safe_load(PATH.read_text(encoding="utf-8")))
    return _CACHE
