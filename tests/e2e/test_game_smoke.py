"""Headless-Chromium smoke test of the real game (doc 27 §12.2): the canvas exists, the player moves on key presses, a flat level
can be won by the ghost bot, and there are no page errors.  Needs `playwright install chromium` (marked slow)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright")
from conftest import ROOT

from text2mario.data import tiles as T
from text2mario.generate.framing import frame
from text2mario.playability.checker import Framing, check_level
from text2mario.render.game_html import build_game_html
from text2mario.render.schema import make_level, validate_level

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (ROOT / "app/static/vendor/phaser.min.js").exists(), reason="fetch assets first")]


def flat_level():
    content = np.zeros((16, 32), np.uint8)
    content[15, :] = T.GROUND
    grid, fr = frame(content)
    return grid, fr, validate_level(make_level(grid, kind="real", framing=fr))


def test_canvas_player_moves_and_bot_wins_a_flat_level():
    from playwright.sync_api import sync_playwright
    grid, fr, lvl = flat_level()
    page_file = Path(tempfile.gettempdir()) / "t2m_smoke.html"
    page_file.write_text(build_game_html(lvl), encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 900, "height": 700})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(page_file.as_uri())
        pg.wait_for_selector("canvas", timeout=20000)
        pg.click("#focus")
        pg.wait_for_timeout(400)
        x0 = pg.evaluate("window.__t2m.state().x")
        pg.keyboard.down("ArrowRight")
        pg.wait_for_timeout(1000)
        pg.keyboard.up("ArrowRight")
        assert pg.evaluate("window.__t2m.state().x") > x0 + 2        # the player moved right on a key press
        res = check_level(grid, mode="level", framing=Framing((fr["spawn"]["col"], fr["spawn"]["row"]), fr["flag"]["col"]))
        assert res.status == "solved"
        pg.reload(); pg.wait_for_selector("canvas"); pg.click("#focus")
        pg.evaluate("p => window.__t2m.runPath(p)", [list(s) for s in res.path])
        for _ in range(200):
            if pg.evaluate("window.__t2m.state().won"):
                break
            pg.wait_for_timeout(100)
        assert pg.evaluate("window.__t2m.state().won"), "ghost bot should reach the flag of a flat level"
        b.close()
    assert not errors, errors
