"""Drive the real Streamlit app with Streamlit's AppTest: valid, empty, over-long and example prompts (rubric rows 6 and 7).

Uses the exported model bundle in app/model (real inference, not a stub).  Marked slow: each generation takes a few seconds."""
import pytest
from conftest import ROOT

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (ROOT / "app/model/t2m_ema.safetensors").exists(), reason="no exported model bundle")]
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def fresh():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def go(at):
    next(b for b in at.button if b.label == "Generate & Play").click()
    at.run()
    return at


def test_valid_prompt_produces_a_level_and_a_badge():
    at = fresh()
    at.text_area(key="prompt").set_value("full floor with two pipes and a few enemies")
    go(at)
    assert not at.exception, at.exception
    texts = " ".join(s.value for s in at.success) + " ".join(c.value for c in at.caption)
    assert "Route found" in texts or any("No route found" in w.value for w in at.warning)
    assert any("columns" in c.value and "seed" in c.value for c in at.caption)   # the timing / settings line


def test_empty_prompt_shows_a_readable_error_not_a_stack_trace():
    at = fresh()
    at.text_area(key="prompt").set_value("   ")
    go(at)
    assert not at.exception
    assert any("describing the level" in e.value for e in at.error)


def test_overlong_prompt_is_rejected():
    at = fresh()
    at.text_area(key="prompt").set_value("pipe " * 200)
    go(at)
    assert not at.exception
    assert any("characters" in e.value for e in at.error)


def test_unrecognised_words_warn_but_do_not_crash():
    at = fresh()
    at.text_area(key="prompt").set_value("xyzzy plugh frobnicate")
    go(at)
    assert not at.exception, at.exception
    assert any("No recognised level words" in w.value for w in at.warning)


def test_example_button_fills_the_prompt():
    at = fresh()
    ex = next(b for b in at.button if b.label.startswith("mostly pits"))
    ex.click()
    at.run()
    assert at.text_area(key="prompt").value.startswith("mostly pits")
