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


def test_dice_never_raises_widget_already_instantiated():
    """Regression (audit P1): Dice wrote st.session_state['seed'] after the number_input existed -> StreamlitWidgetAlreadyInstantiatedError.
    Fixed with an on_click callback.  30 consecutive clicks, then examples, then a generation: no exception, seeds stay valid and change."""
    at = fresh()
    seen = set()
    for _ in range(30):
        next(b for b in at.button if b.label == "Dice").click()
        at.run()
        assert not at.exception, at.exception
        seed = at.number_input(key="seed").value
        assert 0 <= seed <= 10**9
        seen.add(seed)
    assert len(seen) > 20, "Dice must produce different seeds"
    for b in [b for b in at.button if b.key and b.key.startswith("ex")]:
        b.click()
        at.run()
        assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Dice").click()
    at.run()
    go(at)
    assert not at.exception, at.exception


def _stub_sampler(monkeypatch, draws_pipes: bool):
    """Replace the diffusion sampler (the app still runs for real: validation, normalizer, planner, checker, rendering, exports)."""
    import numpy as np

    from text2mario.data import tiles as T
    from text2mario.generate import full_level as FL

    def fake(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = np.zeros((16, 16), np.uint8)
        g[15, :] = T.GROUND
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        n = 2 if "two pipes" in caption else 1 if "one pipe" in caption else 0
        for i in range(n if draws_pipes else 0):
            c = 9 + 4 * i if known_cols is not None else 2 + 6 * i
            g[11, c], g[11, c + 1] = T.PIPE_TL, T.PIPE_TR
            g[12:15, c], g[12:15, c + 1] = T.PIPE_BL, T.PIPE_BR
        return g

    monkeypatch.setattr(FL, "sample_window", fake)
    monkeypatch.setattr(FL, "sample_window_batch", lambda b, cap, w, st, r, seed, kc, dev, n: np.stack([fake(b, cap, w, st, r, seed + i, kc, dev) for i in range(n)]))


def test_app_says_the_counts_match_when_they_do(monkeypatch):
    _stub_sampler(monkeypatch, draws_pipes=True)
    at = fresh()
    at.text_area(key="prompt").set_value("two pipes")
    at.run()
    go(at)
    assert not at.exception, at.exception
    texts = " ".join(s.value for s in at.success)
    assert "Counts match your sentence" in texts and "pipe: asked 2, got 2" in texts and "never edited" in texts


def test_app_says_which_counts_were_not_met_and_never_calls_it_a_success(monkeypatch):
    _stub_sampler(monkeypatch, draws_pipes=False)                                    # a model that cannot draw pipes at all
    at = fresh()
    at.text_area(key="prompt").set_value("two pipes")
    at.run()
    go(at)
    assert not at.exception, at.exception
    assert not any("Counts match" in s.value for s in at.success)
    warn = " ".join(w.value for w in at.warning)
    assert "Could not match every count" in warn and "pipe: asked 2, got 0" in warn and "not edited" in warn


def test_app_refuses_contradictions_and_impossible_counts_with_a_message(monkeypatch):
    _stub_sampler(monkeypatch, draws_pipes=True)
    at = fresh()
    for prompt, needle in (("one pipe and no pipes", "rules out the same thing"), ("500 pipes", "will not fit")):
        at.text_area(key="prompt").set_value(prompt)
        at.run()
        go(at)
        assert not at.exception, at.exception
        assert any(needle in e.value for e in at.error), (prompt, [e.value for e in at.error])


def test_app_rejects_negative_numbers_before_generating():
    at = fresh()
    at.text_area(key="prompt").set_value("-3 pipes")
    at.run()
    assert any("Negative numbers" in e.value for e in at.error) and not at.exception


def test_app_shows_the_per_window_captions_and_no_raw_warning_codes(monkeypatch):
    _stub_sampler(monkeypatch, draws_pipes=True)
    at = fresh()
    at.text_area(key="prompt").set_value("three towers")                              # the caption can only say 'two': the user is told in words
    at.run()
    body = " ".join(c.value for c in at.caption)
    assert "CLAMPED" not in body and "BARE_PLURAL" not in body and "_" not in "".join(c.value for c in at.caption if c.value.startswith("Notes"))


def test_app_survives_a_redeploy_into_a_running_process_with_stale_modules():
    """Production incident found by re-testing the deployed app: Streamlit Cloud keeps ONE process across deployments and only reloads app/, so the new
    script met the OLD text2mario.app.service (no explain_warning) -> ImportError on every page load.  Simulate that process: a stale module already in
    sys.modules and no signature stored; the app must drop the stale modules, re-import the current ones and work.  Everything is restored afterwards so
    other tests keep the module objects they imported."""
    import sys
    import types

    mine = lambda k: k == "text2mario" or k.startswith("text2mario.")
    saved = {k: v for k, v in sys.modules.items() if mine(k)}
    saved_sig = getattr(sys, "_t2m_src_signature", None)
    stale = types.ModuleType("text2mario.app.service")                                       # an old version: has validate_prompt but NOT explain_warning
    stale.validate_prompt = lambda text: (True, "")
    sys.modules["text2mario.app.service"] = stale
    del sys._t2m_src_signature                                                                # the previous script never stored one
    try:
        at = fresh()                                                                          # fresh() asserts there is no exception
        assert at.text_area(key="prompt").value
        assert sys.modules["text2mario.app.service"] is not stale and hasattr(sys.modules["text2mario.app.service"], "explain_warning")
        go(at)
        assert not at.exception, at.exception
    finally:
        for k in [k for k in sys.modules if mine(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys._t2m_src_signature = saved_sig


def test_unchanged_sources_do_not_drop_modules_on_every_rerun():
    """The guard must not reload the model on every interaction: with an unchanged src/ the second run keeps the loaded modules."""
    import sys
    at = fresh()
    mod = sys.modules["text2mario.generate.full_level"]
    at.run()
    assert sys.modules["text2mario.generate.full_level"] is mod


def test_an_edited_sentence_marks_the_old_level_as_belonging_to_an_earlier_sentence(monkeypatch):
    """Mutation audit (U3): this behaviour was only covered by the browser audit, not by a pytest."""
    _stub_sampler(monkeypatch, draws_pipes=True)
    at = fresh()
    at.text_area(key="prompt").set_value("one pipe")
    at.run()
    go(at)
    assert not any("earlier sentence" in i.value for i in at.info)         # right after generating: the level belongs to the sentence in the box
    at.text_area(key="prompt").set_value("two enemies and one cannon")
    at.run()
    assert any("earlier sentence" in i.value for i in at.info)
    assert any(c.value == "one pipe" for c in at.code)                      # the old sentence is shown verbatim as plain text
    go(at)
    assert not any("earlier sentence" in i.value for i in at.info)         # regenerating for the new sentence clears the notice
