"""End-to-end check of 'verified beatable': can a re-planning bot actually finish the level in the real game?  (doc 21 §9 supplement)

Explain-me:
  What: for each level JSON, loads it into the real Phaser game (headless Chromium), asks the checker for a path from the spawn and lets
        the closed-loop ghost bot follow it.  If the bot falls off the path or finishes the path without winning, it re-plans from the
        cell it stands in (a player retrying after a slip), up to 8 times.  Enemies can be removed (the bot does not fight, while the
        checker assumes stomps), so the number measures STRUCTURE only.  Reports success, re-plans needed, and level ids that fail.
  In/Out: levels dir or files -> reports/gates/P5b/replay_levels_<tag>.json.  Previous: scripts/replay_validation.py (16-wide windows).
  Try: python scripts/replay_levels.py --dir reports/gates/P5/levels --n 20 --only-verified --tag p5
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from playwright.sync_api import sync_playwright

from text2mario.playability.checker import Framing, check_level
from text2mario.render.game_html import build_game_html


def play(page, lvl: dict, strip: bool, max_replans: int = 8, timeout: float = 150.0) -> dict:
    lvl = json.loads(json.dumps(lvl))
    if strip:
        for k in ("raw_grid", "display_grid"):
            g = np.asarray(lvl[k])
            g[g == 5] = 0
            lvl[k] = g.tolist()
    grid = np.asarray(lvl["raw_grid"], np.uint8)
    fr = lvl["framing"]
    goal = fr["flag"]["col"]
    f = Path(tempfile.gettempdir()) / "t2m_replay_levels.html"
    f.write_text(build_game_html(lvl), encoding="utf-8")
    page.goto(f.as_uri())
    page.wait_for_selector("canvas")
    page.click("#focus")
    spawn = (fr["spawn"]["col"], fr["spawn"]["row"])
    replans, t0, last_x, last_move, seen_deaths = 0, time.time(), -1.0, time.time(), 0
    path = check_level(grid, mode="level", framing=Framing(spawn, goal), enemies="ignore").path
    if not path:
        return {"won": False, "replans": 0, "why": "checker found no path"}
    page.evaluate("p => window.__t2m.runPath(p)", [list(s) for s in path])
    while time.time() - t0 < timeout:
        st = page.evaluate("window.__t2m.state()")
        if st["won"]:
            return {"won": True, "replans": replans, "seconds": round(time.time() - t0, 1)}
        if st["deaths"] > 8:
            return {"won": False, "replans": replans, "why": "died too often"}
        died_again = st["deaths"] > seen_deaths and st["grounded"] and not st.get("dead")
        if abs(st["x"] - last_x) > 0.05:
            last_x, last_move = st["x"], time.time()
        finished = st.get("botIdx", 0) >= len(path)
        stalled = time.time() - last_move > 4.0
        if st["grounded"] and (finished or stalled or died_again):
            seen_deaths = st["deaths"]
            if replans >= max_replans:
                return {"won": False, "replans": replans, "why": "replan budget"}
            cur = (int(np.floor(st["x"])), round(st["y"]) - 1)
            r = check_level(grid, mode="level", framing=Framing(cur, goal), enemies="ignore")
            if not r.path:
                return {"won": False, "replans": replans, "why": f"no path from {cur}"}
            replans += 1
            path = r.path
            page.evaluate("p => window.__t2m.runPath(p)", [list(s) for s in path])
            last_move = time.time()
        page.wait_for_timeout(200)
    return {"won": False, "replans": replans, "why": "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only-verified", action="store_true")
    ap.add_argument("--keep-enemies", action="store_true")
    ap.add_argument("--tag", default="levels")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted(Path(a.dir).glob("*.level.json") if a.dir and list(Path(a.dir).glob("*.level.json")) else Path(a.dir).glob("*.json"))
    levels = []
    for f in files:
        d = json.loads(f.read_text())
        if "raw_grid" in d and (not a.only_verified or d["verification"]["status"] == "verified"):
            levels.append((f.name, d))
    rng = np.random.default_rng(20261117)
    if len(levels) > a.n:
        levels = [levels[i] for i in sorted(rng.permutation(len(levels))[:a.n])]
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 900, "height": 700})
        for name, lvl in levels:
            r = play(pg, lvl, strip=not a.keep_enemies)
            rows.append({"level": name, **r})
            print(rows[-1], flush=True)
        b.close()
    rate = float(np.mean([r["won"] for r in rows])) if rows else float("nan")
    rep = {"n": len(rows), "success_rate": rate, "mean_replans_when_won": float(np.mean([r["replans"] for r in rows if r["won"]])) if any(r["won"] for r in rows) else None,
           "enemies_removed": not a.keep_enemies, "failures": [r for r in rows if not r["won"]], "rows": rows}
    (ROOT / f"reports/gates/P5b/replay_levels_{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print(f"replay success {rate:.3f} of {len(rows)}")


if __name__ == "__main__":
    main()
