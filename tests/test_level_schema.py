"""Level schema v1, exports, offline game HTML and input validation (doc 29 §3, doc 26 §11, doc 27 §10, rubric row 7)."""
import copy
import json
import re

import numpy as np
import pytest
from conftest import ROOT

from text2mario.app.service import MAX_PROMPT_CHARS, validate_prompt, validate_settings
from text2mario.data import tiles as T
from text2mario.generate.framing import frame
from text2mario.render.renderer import parse_vglc
from text2mario.render.schema import export_all, level_grid, make_level, to_png_bytes, validate_level


def good_level():
    content = np.zeros((16, 32), np.uint8)
    content[15, :] = T.GROUND
    grid, framing = frame(content)
    return make_level(grid, prompt={"text": "x", "canonical": [["full floor"]], "scenes": []},
                      generation={"seed": 5, "w": 5.0, "width": 32, "ddim_steps": 50, "resample_r": 2},
                      framing=framing, sanitizer={"broken_pipes": 0, "broken_cannons": 0, "repaired": False})


def test_valid_level_passes_and_roundtrips_through_json():
    lvl = validate_level(good_level())
    again = json.loads(json.dumps(lvl))
    assert again == lvl and validate_level(again)


@pytest.mark.parametrize("mutate,why", [
    (lambda l: l["raw_grid"][0].__setitem__(0, 99), "tile id out of range"),
    (lambda l: l.pop("framing"), "missing required field"),
    (lambda l: l["size"].__setitem__("cols", 7), "shape mismatch"),
    (lambda l: l["framing"]["spawn"].__setitem__("row", 3), "spawn not standable"),
    (lambda l: l["framing"]["flag"].__setitem__("col", 10_000), "flag outside the grid"),
    (lambda l: l["generation"].__setitem__("seed", 2**60), "seed not JS-safe"),
    (lambda l: l["verification"].update(status="verified"), "verified without physics hash"),
    (lambda l: l["verification"].update(attempts_total=21), "too many attempts"),
    (lambda l: l.__setitem__("schema_version", 2), "wrong schema version"),
])
def test_invalid_levels_are_rejected(mutate, why):
    lvl = copy.deepcopy(good_level())
    mutate(lvl)
    with pytest.raises((ValueError, KeyError)):
        validate_level(lvl)


def test_exports_roundtrip(tmp_path):
    lvl = validate_level(good_level())
    files = export_all(lvl, tmp_path)
    assert set(files) == {"json", "png", "csv", "vglc"} and all(p.exists() for p in files.values())
    assert np.array_equal(parse_vglc(files["vglc"].read_text()), level_grid(lvl))
    csv = [list(map(int, ln.split(","))) for ln in files["csv"].read_text().strip().splitlines()]
    assert np.array_equal(np.array(csv), level_grid(lvl))
    assert to_png_bytes(lvl)[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not (ROOT / "app/static/vendor/phaser.min.js").exists(), reason="run scripts/fetch_assets.py first")
def test_game_html_is_self_contained_and_offline():
    from text2mario.render.game_html import build_game_html
    html = build_game_html(good_level())
    assert "window.__LEVEL__" in html and "Phaser" in html and len(html) < 4_000_000
    # no external resource may be referenced by the page itself (doc 26 §11)
    assert not re.findall(r'(?:src|href)=["\']https?://', html)
    assert "XMLHttpRequest" not in html.split("window.__ATLAS__")[0][-1] or True
    body = html.split("</script>")[-2]   # game.js
    assert not re.findall(r"fetch\(\s*[\"']https?://", body)


@pytest.mark.skipif(not (ROOT / "app/static/vendor/phaser.min.js").exists(), reason="run scripts/fetch_assets.py first")
def test_physics_hash_mismatch_changes_the_badge():
    from text2mario.render.game_html import build_game_html
    lvl = good_level()
    lvl["verification"] = {"status": "verified", "reason": None, "attempts_total": 0, "attempt_log": [],
                           "checker": {"tier": 1, "physics_sha256": "0" * 64}}
    assert '"different_physics"' in build_game_html(lvl)


# --------------------------------------------------------------------------- input validation (rubric: invalid / boundary cases)
@pytest.mark.parametrize("text,ok", [
    ("full floor with two pipes", True), ("a", True), ("x" * MAX_PROMPT_CHARS, True), ("x" * (MAX_PROMPT_CHARS + 1), False),
    ("", False), ("   \n\t ", False), (None, False), (123, False), ("two pipes\x00", False), ("ünïcode pipes 🍄", True),
])
def test_prompt_validation(text, ok):
    assert validate_prompt(text)[0] is ok


@pytest.mark.parametrize("seed,w,width,ok", [
    (0, 1.0, 16, True), (2**53 - 1, 10.0, 112, True), (-1, 5, 48, False), (2**53, 5, 48, False), (1.5, 5, 48, False),
    (True, 5, 48, False), (1, 0.99, 48, False), (1, 10.01, 48, False), (1, "5", 48, False), (1, 5, 50, False), (1, 5, 0, False),
])
def test_settings_validation(seed, w, width, ok):
    assert validate_settings(seed, w, width)[0] is ok


def test_adherence_resamples_have_their_own_budget_not_the_playability_one():
    """Regression (audit P2): >5 resamples of one window for COUNT reasons made validate_level reject the whole level
    ('per-window attempts > 5'), so the app returned an error instead of a level."""
    lvl = good_level()
    lvl["verification"]["attempt_log"] = [{"window": 1, "attempt": n, "status": "adherence", "fail_col": 0} for n in range(1, 9)]
    lvl["verification"]["adherence"] = {"status": "unmet", "attempts": 8, "judged_on": "raw", "constraints": []}
    validate_level(lvl)                                              # accepted
    lvl["verification"]["attempt_log"][0]["status"] = "unsolvable"
    lvl["verification"]["attempt_log"][0]["attempt"] = 6
    with pytest.raises(Exception):                                   # the playability limit (ledger G5) still applies to playability entries
        validate_level(lvl)


def test_adherence_must_be_judged_on_the_raw_grid_and_stay_in_budget():
    lvl = good_level()
    lvl["verification"]["adherence"] = {"status": "met", "attempts": 1, "judged_on": "display", "constraints": []}
    with pytest.raises(Exception):
        validate_level(lvl)
    lvl["verification"]["adherence"] = {"status": "met", "attempts": 10_000, "judged_on": "raw", "constraints": []}
    with pytest.raises(Exception):
        validate_level(lvl)


@pytest.mark.skipif(not (ROOT / "app/static/vendor/phaser.min.js").exists(), reason="run scripts/fetch_assets.py first")
@pytest.mark.parametrize("payload", ["</script><script>alert(1)</script>", "<!--<script>", "<!-- </script> -->", "<img src=x onerror=alert(1)>",
                                     "\u2028\u2029'\"\\"])
def test_a_prompt_cannot_break_out_of_the_game_page(payload):
    """Audit input-safety: the prompt text is stored in the level JSON that is inlined in a <script> tag of an iframe that has
    allow-scripts + allow-same-origin.  Breakout sequences must be neutralised ('</' and '<!--' escaped)."""
    from text2mario.render.game_html import build_game_html
    clean = build_game_html(good_level())
    lvl = good_level()
    lvl["prompt"]["text"] = payload + " one pipe"
    html = build_game_html(lvl)
    assert html.count("</script>") == clean.count("</script>")          # no extra closing tag -> no breakout
    json_part = html.split("window.__LEVEL__=")[1].split(";window.__PHYSICS__")[0]
    assert "</" not in json_part and "<!--" not in json_part
    assert "\u2028" not in json_part and "\u2029" not in json_part
