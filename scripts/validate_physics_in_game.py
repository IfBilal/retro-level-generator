"""Validate the checker's jump envelopes against the REAL game engine (headless Chromium + the ghost bot), doc 21 §9, doc 27 §4.

Explain-me:
  What: builds micro-levels with a long run-up (gaps of width 1..9, walls and ledges of height 1..6), asks the checker whether
        each is solvable, then plays it in the real Phaser game with a scripted bot and records whether the flag is reached.
        A checker verdict 'solved' must imply the game run succeeds (soundness: "verified" is a promise to the player).
  In/Out: -> reports/gates/P5b/game_replay.json and a table on stdout.  Needs `playwright` + `playwright install chromium`.
  Previous: playability/checker.py, render/game_html.py, app/static/game/game.js.  Next: gate_p5b.py reads the JSON.
  Try: edit jump_velocity in physics.json, run derive_physics.py and re-run: the verdicts move together (that is the point).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright

from text2mario.playability.checker import check_level, frame_window
from text2mario.playability.physics import Physics
from text2mario.render.game_html import build_game_html
from text2mario.render.schema import make_level, validate_level

RUN, TAIL = 14, 10   # cells of floor before and after the obstacle


def base() -> np.ndarray:
    g = np.zeros((16, RUN + 1 + TAIL), np.uint8)
    g[15, :] = 8
    return g


def cases():
    out = []
    for k in range(1, 10):                                   # gap of width k
        g = base(); g[15, RUN:RUN + k] = 0; g = np.concatenate([g, np.full((16, k), 0, np.uint8)], 1) if False else g
        g2 = np.zeros((16, RUN + k + TAIL), np.uint8); g2[15, :RUN] = 8; g2[15, RUN + k:] = 8
        out.append((f"gap{k}", g2))
    for h in range(1, 7):                                    # single-column wall of height h
        g = base()
        for r in range(15 - h, 15):
            g[r, RUN] = 8
        out.append((f"wall{h}", g))
    for h in range(1, 7):                                    # step up to a plateau of height h
        g = base()
        for r in range(15 - h, 15):
            g[r, RUN:] = 8
        out.append((f"step_up{h}", g))
    return out


def frame_level(g: np.ndarray):
    fg, fr = frame_window(g)
    cols = fg.shape[1]
    lvl = make_level(fg, kind="real", framing={"templated_cols": [[0, 2], [cols - 6, cols - 1]],
                                                "spawn": {"col": fr.spawn[0], "row": fr.spawn[1]},
                                                "flag": {"col": fr.flag_col, "row_top": 5, "row_bottom": 13}, "checkpoints": []},
                     provenance={"source": "real", "dataset_level_id": None, "dataset_level_name": "micro"})
    return fg, fr, validate_level(lvl)


def naive_path(fg: np.ndarray, fr):
    """Bot path when the checker found none: run along the floor and jump at every obstacle edge toward the flag."""
    path, c = [(fr.spawn[0], fr.spawn[1], "start")], fr.spawn[0]
    return path, c


def first_obstacle_edge(fg: np.ndarray) -> int:
    """Last standable floor cell before the first obstacle (a gap, a wall or a step) in a framed micro-level."""
    for c in range(1, fg.shape[1] - 1):
        if fg[15, c + 1] == 0 or fg[14, c + 1] != 0:
            return c
    return 1


def far_row(fg: np.ndarray, col: int) -> int:
    return 14


def play(page, path, timeout=30.0) -> dict:
    page.reload()
    page.wait_for_selector("canvas")
    page.click("#focus")
    page.evaluate("p => window.__t2m.runPath(p)", [list(x) for x in path])
    t0 = time.time()
    st = {}
    while time.time() - t0 < timeout:
        st = page.evaluate("window.__t2m.state()")
        if st["won"] or st["deaths"] > 0:
            break
        page.wait_for_timeout(100)
    return st


def main() -> int:
    ph = Physics.load()
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 700})
        only = set(sys.argv[1:])
        for name, g in cases():
            if only and name not in only:
                continue
            fg, fr, lvl = frame_level(g)
            html = Path(tempfile.gettempdir()) / f"t2m_{name}.html"
            html.write_text(build_game_html(lvl), encoding="utf-8")
            page.goto(html.as_uri())
            res = check_level(fg, mode="level", framing=fr, physics=ph)
            # (a) replay the checker's own path (only if it found one)
            replay = None
            if res.status == "solved":
                replay = bool(play(page, res.path).get("won"))
            # (b) single clean jump: stand on the last floor cell before the obstacle, run up, jump to the far side
            start = first_obstacle_edge(fg)
            far = fg.shape[1] - 7
            single = bool(play(page, [(start, fr.spawn[1], "start"), (far, far_row(fg, far), "jump_run")]).get("won"))
            rows.append({"case": name, "checker": res.status, "replay_of_checker_path": replay, "single_clean_jump": single})
            print(f"{name:10s} checker={res.status:9s} replay={replay}  single-jump={single}", flush=True)
        browser.close()
    # soundness: if the checker says solvable, a clean single jump (or the replay) must succeed in the real engine
    unsound = [r for r in rows if r["checker"] == "solved" and not (r["single_clean_jump"] or r["replay_of_checker_path"])]
    conservative = [r for r in rows if r["checker"] != "solved" and r["single_clean_jump"]]
    rep = {"physics_sha256": ph.sha256, "n": len(rows), "unsound_checker_solved_game_failed": unsound,
           "conservative_checker_unsolved_game_won": conservative, "rows": rows,
           "soundness_pass": not unsound}
    out = ROOT / "reports/gates/P5b/game_replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"\nunsound (checker says beatable, game fails): {len(unsound)}   conservative (checker says no, game wins): {len(conservative)}")
    return 0 if not unsound else 1


if __name__ == "__main__":
    sys.exit(main())
