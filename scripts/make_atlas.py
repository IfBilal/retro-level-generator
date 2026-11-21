"""Build the game atlas from the Kenney CC0 pack plus a few drawn tiles (doc 26 §5).

Explain-me:
  What: assembles assets/atlas.png = 13 terrain tiles in the ORDER OF THE TILE IDS (ledger F6; so a Phaser tileset index
        equals the tile id), followed by sprite frames (player x4, enemy x2, flag, pole, coin spin).  Pipes and the cannon
        are drawn here with Pillow (original art); everything else is cropped from the Kenney sheets.
  In/Out: assets/kenney/*.png -> assets/atlas.png, assets/atlas_layout.json, assets/tile_map.json.
  Previous: scripts/fetch_assets.py.  Next: app/game builder (render/game_html.py) and the PNG renderer skin.
  Try: change KENNEY_FRAME below (e.g. ground 22 -> 62) and re-run; the ground sprite changes everywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data import tiles as T

TP = 18
SHEET = ROOT / "assets/kenney/tilemap_packed.png"       # 20 x 9 frames of 18 px
CHARS = ROOT / "assets/kenney/tilemap-characters_packed.png"  # 9 x 3 frames of 24 px
# frame index in tilemap_packed.png (row * 20 + col), chosen on the contact sheet
KENNEY_FRAME = {"ground_top": 22, "ground_inner": 24, "breakable": 6, "question_full": 9, "question_empty": 29,
                "coin": 151, "flag": 111, "pole": 131}
CHAR_FRAME = {"player_idle": 0, "player_walk": 1, "player_jump": 1, "enemy_a": 18, "enemy_b": 19}


def crop(sheet: Image.Image, idx: int, cols: int, size: int) -> Image.Image:
    r, c = divmod(idx, cols)
    return sheet.crop((c * size, r * size, (c + 1) * size, (r + 1) * size))


def pipe_tile(top: bool, left: bool) -> Image.Image:
    """Drawn green pipe piece: cap (top) or body, left or right half."""
    im = Image.new("RGBA", (TP, TP), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    dark, mid, light, out = (16, 96, 40, 255), (32, 160, 64, 255), (120, 220, 130, 255), (10, 50, 24, 255)
    x0, x1 = (0, TP - 1) if left else (0, TP - 1)
    if top:
        d.rectangle([0, 0, TP - 1, TP - 1], fill=mid, outline=out)
        d.rectangle([1, 1, 3, TP - 2], fill=light) if left else d.rectangle([TP - 4, 1, TP - 2, TP - 2], fill=dark)
    else:
        d.rectangle([2 if left else 0, 0, TP - 1 if left else TP - 3, TP - 1], fill=mid)
        d.line([2 if left else TP - 3, 0, 2 if left else TP - 3, TP], fill=out)
        d.rectangle([4, 0, 6, TP - 1], fill=light) if left else d.rectangle([TP - 7, 0, TP - 5, TP - 1], fill=dark)
    return im


def cannon_tile(top: bool) -> Image.Image:
    im = Image.new("RGBA", (TP, TP), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, TP - 1, TP - 1], fill=(60, 60, 70, 255), outline=(20, 20, 28, 255))
    if top:
        d.ellipse([4, 4, TP - 5, TP - 5], fill=(15, 15, 20, 255))
        d.rectangle([TP // 2, 6, TP - 2, TP - 7], fill=(15, 15, 20, 255))
    else:
        d.rectangle([3, 3, TP - 4, TP - 4], outline=(90, 90, 105, 255))
    return im


def main() -> int:
    sheet = Image.open(SHEET).convert("RGBA")
    chars = Image.open(CHARS).convert("RGBA")
    k = lambda name: crop(sheet, KENNEY_FRAME[name], 20, TP)
    tiles: dict[int, Image.Image] = {
        T.SKY: Image.new("RGBA", (TP, TP), (0, 0, 0, 0)),
        T.PIPE_TL: pipe_tile(True, True), T.PIPE_TR: pipe_tile(True, False),
        T.PIPE_BL: pipe_tile(False, True), T.PIPE_BR: pipe_tile(False, False),
        T.QBLOCK: k("question_full"), T.QEMPTY: k("question_empty"), T.BREAKABLE: k("breakable"),
        T.GROUND: k("ground_inner"), T.CANNON_TOP: cannon_tile(True), T.CANNON_SUP: cannon_tile(False),
        T.ENEMY_ID: Image.new("RGBA", (TP, TP), (0, 0, 0, 0)),     # enemies are sprites, not tiles
        T.COIN: Image.new("RGBA", (TP, TP), (0, 0, 0, 0)),         # coins are sprites, not tiles
    }
    ground_top = k("ground_top")
    # sprite frames after the 13 tiles (each 24 px wide cell for characters; stored at native size)
    names = [f"tile_{n}" for n in T.NAMES]
    W = TP * 13 + 24 * 6 + TP * 4
    atlas = Image.new("RGBA", (W + (-W) % TP, 2 * TP), (0, 0, 0, 0))  # whole tiles (Phaser warns otherwise)
    layout = {"tile_px": TP, "tiles": {}, "frames": {}}
    for i in range(13):
        atlas.alpha_composite(tiles[i], (i * TP, 0))
        layout["tiles"][T.NAMES[i]] = {"id": i, "x": i * TP, "y": 0, "w": TP, "h": TP}
    x = TP * 13
    for nm in ("player_idle", "player_walk", "player_jump", "enemy_a", "enemy_b"):
        atlas.alpha_composite(crop(chars, CHAR_FRAME[nm], 9, 24), (x, 0))
        layout["frames"][nm] = {"x": x, "y": 0, "w": 24, "h": 24}
        x += 24
    atlas.alpha_composite(Image.new("RGBA", (24, 24), (0, 0, 0, 0)), (x, 0))
    x += 24
    for nm in ("coin", "flag", "pole", "ground_top"):
        atlas.alpha_composite(k(nm), (x, 0))
        layout["frames"][nm] = {"x": x, "y": 0, "w": TP, "h": TP}
        x += TP
    out = ROOT / "assets"
    atlas.save(out / "atlas.png")
    (out / "atlas_layout.json").write_text(json.dumps(layout, indent=1) + "\n")
    (out / "tile_map.json").write_text(json.dumps({
        "source": "Kenney Pixel Platformer (CC0) + Pillow-drawn pipes/cannon", "kenney_frames": KENNEY_FRAME,
        "character_frames": CHAR_FRAME, "notes": "ground uses the dirt frame; the PNG renderer draws grass on top rows"},
        indent=1) + "\n")
    # contact sheet: the 13 tiles in id order, for the visual check (doc 26 §8)
    cs = Image.new("RGBA", (13 * TP * 4, TP * 4), (156, 207, 255, 255))
    for i in range(13):
        cs.alpha_composite(tiles[i].resize((TP * 4, TP * 4), Image.NEAREST), (i * TP * 4, 0))
    (ROOT / "reports/assets").mkdir(parents=True, exist_ok=True)
    cs.convert("RGB").save(ROOT / "reports/assets/tile_contact_sheet.png")
    ground_top.save(out / "ground_top.png")
    print("wrote assets/atlas.png", atlas.size, "and reports/assets/tile_contact_sheet.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
