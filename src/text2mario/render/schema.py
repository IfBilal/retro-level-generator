"""Level JSON schema v1 (doc 29): build, validate and round-trip levels; export PNG / CSV / VGLC / JSON.

Explain-me:
  What: one dictionary format for generated, real and showcase levels, used by the game, the app, the evaluation and
        the exports.  `make_level` fills the required fields, `validate_level` enforces doc 29 §3, `export_all` writes
        the downloadable files.
  In/Out: grids (uint8 16 x cols) + metadata -> dict (JSON-safe, ints only);  validate_level(dict) raises ValueError.
  Previous: generate/full_level.py (framed grid, verification).  Next: render/game_html.py, app/streamlit_app.py.
  Try: change a tile id to 99 in raw_grid and watch validate_level reject it.
"""
from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from text2mario.data import tiles as T
from text2mario.utils import appcfg
from text2mario.render.renderer import grid_to_csv, grid_to_sprite_image, grid_to_vglc_text

SCHEMA_VERSION = 1
MAX_SEED = 2**53 - 1


def new_level_id() -> str:
    return time.strftime("t2m_%Y%m%d-%H%M%S_", time.gmtime()) + uuid.uuid4().hex[:6]


ADHERENCE_MAX_ATTEMPTS = int(appcfg.load()["adherence"]["max_attempts"])      # configs/generate/app.yaml


def make_level(raw_grid, *, kind: str = "generated", display_grid=None, prompt: dict | None = None,
               generation: dict | None = None, framing: dict | None = None, sanitizer: dict | None = None,
               verification: dict | None = None, provenance: dict | None = None, level_id: str | None = None,
               metrics: dict | None = None) -> dict:
    raw = np.asarray(raw_grid)
    disp = raw if display_grid is None else np.asarray(display_grid)
    rows, cols = raw.shape
    lvl = {"schema_version": SCHEMA_VERSION, "level_id": level_id or new_level_id(), "kind": kind,
           "size": {"rows": int(rows), "cols": int(cols)}, "raw_grid": raw.astype(int).tolist(),
           "display_grid": disp.astype(int).tolist(),
           "framing": framing or {"templated_cols": [], "spawn": {"col": 0, "row": rows - 2},
                                  "flag": {"col": cols - 1, "row_top": 5, "row_bottom": rows - 3}, "checkpoints": []},
           "verification": verification or {"status": "not_verified", "reason": None, "attempts_total": 0,
                                            "attempt_log": [], "checker": None},
           "metrics": metrics or {"difficulty": None, "cscore": None, "notes": None},
           "provenance": provenance or {"source": "app", "dataset_level_id": None, "dataset_level_name": None},
           "ext": {}}
    if prompt is not None:
        lvl["prompt"] = prompt
    if generation is not None:
        lvl["generation"] = generation
    if sanitizer is not None:
        lvl["sanitizer"] = sanitizer
    elif kind == "generated":
        lvl["sanitizer"] = {"broken_pipes": 0, "broken_cannons": 0, "repaired": False}
    return lvl


def validate_level(lvl: dict) -> dict:
    """Raise ValueError on any violation of doc 29 §3; return the level for chaining."""
    def bad(msg):
        raise ValueError(f"invalid level: {msg}")
    for k in ("schema_version", "level_id", "kind", "size", "raw_grid", "display_grid", "framing", "verification"):
        if k not in lvl:
            bad(f"missing field {k}")
    if lvl["schema_version"] != SCHEMA_VERSION:
        bad("schema_version")
    if lvl["kind"] == "generated":
        for k in ("prompt", "generation", "sanitizer"):
            if k not in lvl:
                bad(f"generated level needs {k}")
    rows, cols = lvl["size"]["rows"], lvl["size"]["cols"]
    for name in ("raw_grid", "display_grid"):
        g = np.asarray(lvl[name])
        if g.shape != (rows, cols):
            bad(f"{name} shape {g.shape} != size {(rows, cols)}")
        if g.min() < 0 or g.max() >= T.N_TILES or not np.issubdtype(g.dtype, np.integer):
            bad(f"{name} has tile ids outside 0..{T.N_TILES - 1}")
    fr = lvl["framing"]
    sc, sr = fr["spawn"]["col"], fr["spawn"]["row"]
    disp = np.asarray(lvl["display_grid"])
    if not (0 <= sc < cols and 0 <= sr < rows - 1):
        bad("spawn outside the grid")
    if int(disp[sr, sc]) in T.SOLID or int(disp[sr + 1, sc]) not in T.SOLID:
        bad("spawn is not a standable cell")
    if not 0 <= fr["flag"]["col"] < cols:
        bad("flag.col outside the grid")
    for a, b in fr.get("templated_cols", []):
        if not (0 <= a <= b < cols):
            bad("templated_cols outside the grid")
    v = lvl["verification"]
    if v["status"] == "verified" and not (v.get("checker") or {}).get("physics_sha256"):
        bad("verified without physics_sha256")
    if v.get("attempts_total", 0) > 20:                     # playability + forbidden resamples (ledger G5); adherence has its own budget below
        bad("attempts_total > 20")
    for a in v.get("attempt_log", []):
        if a.get("status") != "adherence" and a.get("attempt", 1) > 5:
            bad("per-window attempts > 5")
    adh = v.get("adherence") or {}
    if adh.get("attempts", 0) > ADHERENCE_MAX_ATTEMPTS:
        bad("adherence attempts over its budget")
    if adh and adh.get("judged_on") != "raw":
        bad("adherence must be judged on the raw grid (CLAUDE.md rule 8)")
    g = lvl.get("generation")
    if g and not (0 <= g["seed"] <= MAX_SEED and isinstance(g["seed"], int)):
        bad("seed is not JS-safe")
    json.dumps(lvl, allow_nan=False)   # NaN / inf are not allowed
    return lvl


def level_grid(lvl: dict, which: str = "display_grid") -> np.ndarray:
    return np.asarray(lvl[which], dtype=np.uint8)


def to_png_bytes(lvl: dict, scale: int = 2) -> bytes:
    im = grid_to_sprite_image(level_grid(lvl), 18)
    im = im.resize((im.width * scale // 1, im.height * scale // 1), Image.NEAREST) if scale != 1 else im
    meta = PngInfo()
    meta.add_text("level_id", lvl["level_id"])
    if "generation" in lvl:
        meta.add_text("seed", str(lvl["generation"]["seed"]))
    if "prompt" in lvl:
        meta.add_text("prompt", lvl["prompt"]["text"])
    buf = io.BytesIO()
    im.save(buf, "PNG", pnginfo=meta)
    return buf.getvalue()


def export_all(lvl: dict, out_dir: str | Path) -> dict[str, Path]:
    """Write the five downloadable files of doc 29 §4 (the playable HTML is built by render/game_html.py)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lid = lvl["level_id"]
    files = {"json": out / f"t2m_{lid}.level.json", "png": out / f"t2m_{lid}.png",
             "csv": out / f"t2m_{lid}.ids.csv", "vglc": out / f"t2m_{lid}.vglc16.txt"}
    files["json"].write_text(json.dumps(lvl, indent=1))
    files["png"].write_bytes(to_png_bytes(lvl))
    files["csv"].write_text(grid_to_csv(level_grid(lvl)))
    files["vglc"].write_text(grid_to_vglc_text(level_grid(lvl)) + "\n")
    return files
