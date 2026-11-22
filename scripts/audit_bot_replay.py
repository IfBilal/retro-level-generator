"""Does the fix hurt playability END TO END?  python scripts/audit_bot_replay.py --runs baseline_full roundC_baseline_seeds [--ids ...] [--seeds 4]

Explain-me:
  What: takes the SAME (prompt, seed) jobs from two audit runs, rebuilds each framed level from the stored raw grid, asks the playability checker for a route, and for every level the
        checker calls solvable replays that route with the ghost bot in the real Phaser game (headless Chromium): did the bot reach the flag?  The share "bot wins among
        checker-verified levels" is a lower bound on beatability (the bot is not a human), and comparing before / after on identical jobs shows whether the fix changed it.
  In/Out: reports/audit/<run>/rows.jsonl -> reports/audit/bot_replay.json (+ printed table).
  Previous: scripts/audit_adherence.py, scripts/make_showcase.py (same replay code).  Next: reports/audit/AUDIT_REPORT.md.
  Try: python scripts/audit_bot_replay.py --runs baseline_full roundC_baseline_seeds --seeds 3
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

from text2mario.data import tiles as T
from text2mario.generate.framing import frame
from text2mario.playability.checker import Framing, check_level
from text2mario.playability.physics import Physics
from text2mario.render.game_html import build_game_html
from text2mario.render.schema import make_level

DEFAULT_IDS = ["cnt_pipe_1", "cnt_pipe_2", "cnt_enemy_2", "cnt_coin_3", "cnt_platform_2", "cnt_cannon_2", "cnt_question_block_2", "cnt_staircase_1",
               "ex_button_1", "ex_button_2", "ord_three", "neg_two_pipes_no_enemies", "two_scene_same"]


def load(run):
    out = {}
    for line in (ROOT / "reports/audit" / run / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("grid"):
            out[(r["id"], r["seed"])] = r
    return out


def build(row, ph):
    content = np.array([[T.CHARS.index(c) for c in line] for line in row["grid"]], np.uint8)
    grid, fr = frame(content)
    lvl = make_level(grid, framing=fr, prompt={"text": row["text"], "canonical": [], "scenes": []},
                     generation={"seed": int(row["seed"]), "w": 7.5, "width": content.shape[1], "ddim_steps": 25, "resample_r": 0},
                     sanitizer={"broken_pipes": 0, "broken_cannons": 0, "repaired": False},
                     verification={"status": "verified", "checker": {"physics_sha256": ph.sha256}})
    return grid, fr, lvl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=45.0)
    a = ap.parse_args()
    ph = Physics.load()
    data = {r: load(r) for r in a.runs}
    jobs = [(i, s) for i in a.ids.split(",") for s in (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009)[:a.seeds]]
    res = {r: {"jobs": 0, "checker_solved": 0, "bot_won": 0, "rows": []} for r in a.runs}
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 900, "height": 700})
        for run in a.runs:
            for key in jobs:
                row = data[run].get(key)
                if row is None:
                    continue
                res[run]["jobs"] += 1
                grid, fr, lvl = build(row, ph)
                r = check_level(grid, mode="level", framing=Framing((fr["spawn"]["col"], fr["spawn"]["row"]), fr["flag"]["col"]), physics=ph, enemies="ignore", budget_nodes=1000)
                won = None
                if r.status == "solved":
                    res[run]["checker_solved"] += 1
                    f = Path(tempfile.gettempdir()) / "t2m_botreplay.html"
                    f.write_text(build_game_html(lvl), encoding="utf-8")
                    page.goto(f.as_uri())
                    page.wait_for_selector("canvas")
                    page.click("#focus")
                    page.evaluate("p => window.__t2m.runPath(p)", [list(s) for s in r.path])
                    t0, st = time.time(), {}
                    while time.time() - t0 < a.timeout:
                        st = page.evaluate("window.__t2m.state()")
                        if st["won"] or st["deaths"] > 2:
                            break
                        page.wait_for_timeout(200)
                    won = bool(st.get("won"))
                    res[run]["bot_won"] += won
                res[run]["rows"].append({"id": key[0], "seed": key[1], "checker": r.status, "bot_won": won})
            print(run, {k: v for k, v in res[run].items() if k != "rows"}, flush=True)
        b.close()
    (ROOT / "reports/audit/bot_replay.json").write_text(json.dumps({"runs": a.runs, "ids": a.ids.split(","), "seeds_per_prompt": a.seeds, "results": res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
