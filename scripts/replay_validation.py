"""Replay the checker's paths in the real game engine on REAL train/dev windows (doc 21 §9 supplement to G-P5b).

Explain-me:
  What: picks real framed 16-wide windows (enemies removed: the bot does not fight), keeps those the checker calls solvable,
        replays each checker path with the closed-loop ghost bot in headless Chromium and records whether the flag is reached.
        The success rate is a LOWER bound on the checker's soundness (a failing replay can be the bot's fault, not the checker's).
  In/Out: -> reports/gates/P5b/replay_real_windows.json.  Previous: scripts/validate_physics_in_game.py (micro-levels).
  Try: --n 20 for a quick look; failures list the window so you can open it in the game and play it yourself.
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
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_checker import real_levels  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from text2mario.playability.checker import Framing, check_level, frame_window  # noqa: E402
from text2mario.render.game_html import build_game_html  # noqa: E402
from text2mario.render.schema import make_level, validate_level  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=40.0)
    a = ap.parse_args()
    rng = np.random.default_rng(20261117)
    cands = []
    for lid, (name, g, sp) in real_levels().items():
        g = np.where(g == 5, 0, g).astype(g.dtype)               # enemy-free: structure only
        for c0 in range(0, g.shape[1] - 15, 16):
            fg, fr = frame_window(g[:, c0:c0 + 16])
            if check_level(fg, mode="level", framing=fr).status == "solved":
                cands.append((f"L{lid}:{name}:w{c0}", fg, fr))
    idx = rng.permutation(len(cands))[:a.n]
    picks = [cands[i] for i in sorted(idx)]
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 900, "height": 700})
        for lab, fg, fr in picks:
            cols = fg.shape[1]
            fd = {"col": fr.flag_col, "row_top": 5, "row_bottom": 13}
            lvl = validate_level(make_level(fg, kind="real", framing={"templated_cols": [[0, 2], [cols - 6, cols - 1]],
                                  "spawn": {"col": fr.spawn[0], "row": fr.spawn[1]}, "flag": fd, "checkpoints": []},
                                  provenance={"source": "real", "dataset_level_id": None, "dataset_level_name": lab}))
            f = Path(tempfile.gettempdir()) / "t2m_replay.html"
            f.write_text(build_game_html(lvl), encoding="utf-8")
            res = check_level(fg, mode="level", framing=fr)
            pg.goto(f.as_uri()); pg.wait_for_selector("canvas"); pg.click("#focus")
            pg.evaluate("p => window.__t2m.runPath(p)", [list(s) for s in res.path])
            t0, st = time.time(), {}
            while time.time() - t0 < a.timeout:
                st = pg.evaluate("window.__t2m.state()")
                if st["won"] or st["deaths"] > 2:
                    break
                pg.wait_for_timeout(150)
            rows.append({"window": lab, "won": bool(st.get("won")), "deaths": st.get("deaths"), "game_seconds": round(st.get("time", 0), 1),
                         "path_len": len(res.path)})
            print(f"{lab:28s} won={rows[-1]['won']} deaths={rows[-1]['deaths']} t={rows[-1]['game_seconds']}", flush=True)
        b.close()
    rate = float(np.mean([r["won"] for r in rows]))
    rep = {"n": len(rows), "replay_success_rate": rate, "note": "lower bound on soundness: failures may be the closed-loop bot",
           "failures": [r["window"] for r in rows if not r["won"]], "rows": rows}
    (ROOT / "reports/gates/P5b/replay_real_windows.json").write_text(json.dumps(rep, indent=1))
    print(f"replay success {rate:.3f} of {len(rows)}")


if __name__ == "__main__":
    main()
