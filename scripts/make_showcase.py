"""Generate the showcase gallery (12 levels) with the FROZEN model: python scripts/make_showcase.py

Explain-me:
  What: for 12 fixed prompts and seeds, generates a level at width 48, keeps it, replays its checker path with the ghost bot in
        the real game, and records whether the bot reached the flag.  Output goes to app/showcase/ (levels + PNG + manifest.json).
        'human_played' is false until a person plays it: this script never claims a human played anything.
  In/Out: FROZEN.json / app/model bundle -> app/showcase/*.level.json, *.png, manifest.json.  Used by the Gallery page (fallback demo).
  Try: python scripts/make_showcase.py --device cpu
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROMPTS = ["full floor with two pipes and a few enemies", "flat ground, one pipe, then several coins and a staircase",
           "an underground level: three platforms, then a cannon and two enemies", "mostly pits with two islands of ground, then a coin line",
           "two pipes followed by a coin line", "a few question blocks and several enemies", "full floor. one ascending staircase. two towers.",
           "lots of coins and a couple of goombas", "begin with a platform and end with stairs going down", "three pipes and no enemies",
           "full floor with a few gaps and one cannon", "several platforms, then two pipes and a tower"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()
    from text2mario.app.service import LevelService
    from text2mario.render.game_html import build_game_html
    fz = json.loads((ROOT / "FROZEN.json").read_text()) if (ROOT / "FROZEN.json").exists() else {}
    w = a.w if a.w is not None else fz.get("w", 5.0)
    svc = LevelService()
    out = ROOT / "app/showcase"
    out.mkdir(parents=True, exist_ok=True)
    pw = None
    if not a.no_replay:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 700})
    items = []
    for i, p in enumerate(PROMPTS):
        res = svc.run(p, seed=100 + i, w=w, width=a.width)
        lvl = res.level
        name = f"showcase_{i:02d}.level.json"
        (out / name).write_text(json.dumps(lvl))
        (out / f"showcase_{i:02d}.png").write_bytes(res.files["png"])
        bot = None
        if pw and lvl["verification"]["status"] == "verified":
            import numpy as np

            from text2mario.playability.checker import Framing, check_level
            fr = lvl["framing"]
            r = check_level(np.asarray(lvl["raw_grid"], np.uint8), mode="level", framing=Framing((fr["spawn"]["col"], fr["spawn"]["row"]), fr["flag"]["col"]), enemies="ignore")
            f = Path(tempfile.gettempdir()) / "t2m_show.html"
            f.write_text(build_game_html(lvl), encoding="utf-8")
            page.goto(f.as_uri()); page.wait_for_selector("canvas"); page.click("#focus")
            page.evaluate("p => window.__t2m.runPath(p)", [list(s) for s in r.path])
            t0 = time.time(); st = {}
            while time.time() - t0 < 60:
                st = page.evaluate("window.__t2m.state()")
                if st["won"] or st["deaths"] > 2:
                    break
                page.wait_for_timeout(200)
            bot = bool(st.get("won"))
        items.append({"prompt": p, "seed": 100 + i, "w": w, "file": name, "verified": lvl["verification"]["status"] == "verified",
                      "bot_reached_flag": bot, "human_played": False, "seconds": res.seconds["total"]})
        print(i, lvl["verification"]["status"], "bot", bot, flush=True)
    if pw:
        browser.close(); pw.stop()
    (out / "manifest.json").write_text(json.dumps({"checkpoint_step": fz.get("step"), "checkpoint_sha256": fz.get("checkpoint_sha256"), "levels": items}, indent=1))


if __name__ == "__main__":
    main()
