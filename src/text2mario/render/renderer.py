"""Grid <-> tensor <-> image <-> text conversions (doc 02 FR-A-08/11, doc 29, doc 11 §2).

Explain-me:
  What: everything that turns a tile grid into something to look at or export: argmax decoding of model output,
        a PNG render with the placeholder skin (colour + glyph per tile), VGLC text, CSV and a minimal level JSON.
  Axis convention (fixed everywhere): row 0 = TOP, row 15 = bottom (floor), column 0 = left.
  In/Out: grids are uint8 (16, W) arrays, batches (B, 16, W); tensors are (B, 13, 16, W).
  Used by: gate G-P1/G-P2 figures, the EDA, the evaluation figures and (later) the Streamlit app and game export.
  Try: change tile_px in grid_to_image, or recolour a tile in data/tiles.py COLOURS.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from text2mario.data import tiles as T


def tensor_to_grid(x) -> np.ndarray:
    """(B,13,16,W) or (13,16,W) model-space tensor -> uint8 grid(s) by argmax over channels (ties -> lowest id)."""
    x = torch.as_tensor(x)
    return x.argmax(dim=-3).to(torch.uint8).cpu().numpy()


def grid_to_tensor(grid, dtype=torch.float32) -> torch.Tensor:
    """uint8 grid(s) -> {-1, +1} one-hot tensor (D7): x = 2*onehot - 1."""
    g = torch.as_tensor(np.asarray(grid)).long()
    oh = torch.nn.functional.one_hot(g, T.N_TILES)
    oh = oh.permute(0, 3, 1, 2) if g.dim() == 3 else oh.permute(2, 0, 1)
    return oh.to(dtype) * 2 - 1


def _rgb(hexcol: str) -> tuple[int, int, int]:
    return tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))


def grid_to_image(grid: np.ndarray, tile_px: int = 16, glyphs: bool = True) -> Image.Image:
    """Placeholder-skin render (doc 26 A2): one coloured square per tile, with its character on top."""
    grid = np.asarray(grid)
    h, w = grid.shape
    img = Image.new("RGB", (w * tile_px, h * tile_px))
    d = ImageDraw.Draw(img)
    for r in range(h):
        for c in range(w):
            t = int(grid[r, c])
            x0, y0 = c * tile_px, r * tile_px
            d.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1], fill=_rgb(T.COLOURS[t]))
            if glyphs and t != T.SKY and tile_px >= 12:
                d.text((x0 + tile_px // 3, y0 + tile_px // 4), T.CHARS[t], fill=(0, 0, 0))
    return img


_ATLAS = {}


def _atlas():
    """Load assets/atlas.png once -> {tile id or frame name: RGBA image} (None if the atlas has not been built)."""
    if not _ATLAS:
        from pathlib import Path
        root = Path(__file__).resolve().parents[3] / "assets"
        if not (root / "atlas.png").exists():
            _ATLAS["missing"] = True
        else:
            sheet_, lay = Image.open(root / "atlas.png").convert("RGBA"), json.loads((root / "atlas_layout.json").read_text())
            for v in lay["tiles"].values():
                _ATLAS[v["id"]] = sheet_.crop((v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]))
            for k, v in lay["frames"].items():
                _ATLAS[k] = sheet_.crop((v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]))
    return None if "missing" in _ATLAS else _ATLAS


def grid_to_sprite_image(grid: np.ndarray, tile_px: int = 18, sky=(156, 207, 255)) -> Image.Image:
    """Sprite-skin render (Kenney CC0 + drawn pipes): grass on exposed ground, enemy and coin sprites over sky.
    Falls back to the placeholder skin if the atlas is missing (that is logged by the caller, CLAUDE.md rule 12)."""
    at = _atlas()
    if at is None:
        return grid_to_image(grid, tile_px)
    grid = np.asarray(grid)
    h, w = grid.shape
    img = Image.new("RGBA", (w * 18, h * 18), sky + (255,))
    for r in range(h):
        for c in range(w):
            t = int(grid[r, c])
            if t == T.ENEMY_ID:
                img.alpha_composite(at["enemy_a"].crop((3, 3, 21, 21)), (c * 18, r * 18))
            elif t == T.COIN:
                img.alpha_composite(at["coin"], (c * 18, r * 18))
            elif t == T.GROUND and (r == 0 or int(grid[r - 1, c]) not in T.SOLID):
                img.alpha_composite(at["ground_top"], (c * 18, r * 18))
            elif t != T.SKY:
                img.alpha_composite(at[t], (c * 18, r * 18))
    img = img.convert("RGB")
    return img if tile_px == 18 else img.resize((w * tile_px, h * tile_px), Image.NEAREST)


def sheet(grids, cols: int = 8, tile_px: int = 6, pad: int = 4) -> Image.Image:
    """Contact sheet of several grids (used for sample sheets and gate figures)."""
    ims = [grid_to_image(g, tile_px, glyphs=False) for g in grids]
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), (255, 255, 255))
    for i, im in enumerate(ims):
        out.paste(im, (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    return out


def grid_to_vglc_text(grid: np.ndarray) -> str:
    """16 lines of characters per ledger F6."""
    return "\n".join("".join(T.CHARS[int(v)] for v in row) for row in np.asarray(grid)) + "\n"


def parse_vglc(text: str) -> np.ndarray:
    """Inverse of grid_to_vglc_text."""
    return np.array([[T.CHAR_TO_ID[c] for c in line] for line in text.splitlines() if line], dtype=np.uint8)


def grid_to_csv(grid: np.ndarray) -> str:
    """16 rows of comma-separated tile ids."""
    return "\n".join(",".join(str(int(v)) for v in row) for row in np.asarray(grid)) + "\n"


def grid_to_json(grid: np.ndarray, kind: str = "real", level_id: str = "", **extra) -> str:
    """Minimal level JSON (doc 29 schema v1 fields that exist for non-generated levels)."""
    g = np.asarray(grid)
    obj = {"schema_version": 1, "level_id": level_id, "kind": kind, "size": {"rows": int(g.shape[0]), "cols": int(g.shape[1])},
           "raw_grid": g.astype(int).tolist(), "display_grid": g.astype(int).tolist(), **extra}
    return json.dumps(obj)
