"""Build ONE self-contained, offline HTML page that plays a level (doc 27 §10, doc 29 playable HTML).

Explain-me:
  What: inlines Phaser 3.90.0, the level JSON, physics.json, the sprite atlas (base64) and game.js into a single string.
        No external URLs: the page works with Wi-Fi off and as a downloaded file.
  In/Out: build_game_html(level dict) -> str.  The physics hash stored in the level is compared with the current
        physics.json; on a mismatch the badge reads "Verified with different physics" instead of "Verified beatable".
  Previous: render/schema.py (level dict).  Next: app/streamlit_app.py (embeds the string) or a file download.
  Try: change window zoom ZOOM in game.js or edit the CSS below, then rebuild.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

from text2mario.playability.physics import PHYSICS_PATH, Physics

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "app" / "static" / "vendor" / "phaser.min.js"
GAME_JS = ROOT / "app" / "static" / "game" / "game.js"
ATLAS = ROOT / "assets" / "atlas.png"


def _js_json(obj) -> str:
    """JSON that is safe inside a <script> tag ('</', '<!--' and U+2028/9 escaped).  '<!--' matters: '<!--' followed by '<script' makes the HTML
    parser treat the next '</script>' as data, which would swallow the game's own scripts (a prompt could break the page)."""
    out = json.dumps(obj, separators=(",", ":"))
    return out.replace("</", "<\\/").replace("<!--", "<\\!--").replace(chr(0x2028), "\\u2028").replace(chr(0x2029), "\\u2029")


def build_game_html(level: dict) -> str:
    lvl = copy.deepcopy(level)
    ver = lvl.get("verification", {})
    cur = Physics.load().sha256
    if ver.get("status") == "verified" and (ver.get("checker") or {}).get("physics_sha256") != cur:
        ver["status"] = "different_physics"
    phys = json.loads(PHYSICS_PATH.read_text(encoding="utf-8"))
    atlas = base64.b64encode(ATLAS.read_bytes()).decode()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>text2mario - {lvl['level_id']}</title>
<style>html,body{{margin:0;background:#1b2330;color:#fff;font-family:system-ui,sans-serif}}
#wrap{{position:relative;width:800px;max-width:100%;margin:0 auto}}canvas{{display:block;width:100%!important;height:auto!important;image-rendering:pixelated;outline:none}}
#focus{{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#000a;cursor:pointer;font-size:20px;text-align:center}}
#help{{font-size:13px;text-align:center;padding:6px;opacity:.85}}</style></head>
<body><div id="wrap"><div id="game"></div><div id="focus">Click to play<br><small>arrows / WASD move, Space jump, Shift run, R restart</small></div></div>
<div id="help">Arrows or A/D to move, Space or Up to jump, Shift to run. Stomp enemies from above; reach the flag.</div>
<script>{VENDOR.read_text(encoding='utf-8')}</script>
<script>window.__LEVEL__={_js_json(lvl)};window.__PHYSICS__={_js_json(phys)};window.__ATLAS__="{atlas}";</script>
<script>{GAME_JS.read_text(encoding='utf-8')}</script>
</body></html>"""
