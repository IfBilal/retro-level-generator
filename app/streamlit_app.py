"""text2mario - "Type it, play it": a sentence in, a checked, playable level out (doc 28).

Run:  streamlit run app/streamlit_app.py        (set T2M_CKPT=path/to/step_XXXX.pt, or write app/model/ckpt.json)
Explain-me:
  What: sidebar prompt -> normalizer preview ("what the model understood") -> generate (outpainting, sanitizer, checker,
        resampling) -> verification badge + playable game (Phaser, offline) + exports.  CPU fast mode is automatic without CUDA.
  Previous: src/text2mario/app/service.py.  Next: the downloaded HTML runs without this app.
  Try: change the CFG slider (guidance w) or the width and watch the verification log and the time per level.
"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

st.set_page_config(page_title="text2mario", page_icon=":video_game:", layout="wide")

from text2mario.utils import appcfg

CFG = appcfg.load()                    # configs/generate/app.yaml
EXAMPLES = CFG["examples"]


@st.cache_resource(show_spinner="Loading the model...")
def get_service():
    from text2mario.app.service import LevelService
    return LevelService()


from text2mario.app.service import explain_warning, validate_prompt


def embed_game(html: str, height: int = 620) -> None:
    """Render the self-contained game HTML in an iframe that can take keyboard focus (doc 28 §2)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height, tab_index=0, alt="Playable generated level")
    else:  # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def adherence_notice(level: dict) -> None:
    """Say plainly whether the counts the sentence asked for are in the generated tiles (judged on the RAW grid; nothing is edited)."""
    a = level["verification"].get("adherence")
    if not a or a["status"] == "not_applicable":
        return
    rows = a["constraints"]
    text = "; ".join(f"{r['topic']}: asked {r['lo'] if r['lo'] == r['hi'] else str(r['lo']) + '-' + str(r['hi'])}, got {r['got']}" for r in rows)
    if a["status"] == "met":
        st.success(f"Counts match your sentence ({text}). Checked on the generated tiles; {a['attempts']} re-sample(s).")
    else:
        bad = "; ".join(f"{r['topic']}: asked {r['lo'] if r['lo'] == r['hi'] else str(r['lo']) + '-' + str(r['hi'])}, got {r['got']}" for r in rows if not r["met"])
        st.warning(f"Could not match every count after {a['attempts']} re-sample(s): {bad}. The level is shown as generated (not edited).")


def badge(level: dict) -> None:
    v = level["verification"]
    if v["status"] == "verified":
        st.success("Route found: the playability checker found a path from start to flag (not a guarantee).")
    else:
        st.warning(f"No route found by the checker ({v.get('reason') or 'search exhausted'}). You can still play it.")
    st.caption("'Route found' means our checker found a path under the game's physics (enemies assumed stompable). It matches the game engine on "
               "micro-levels, but an automatic test player finishes only part of generated levels end to end (see the report), so treat it as "
               "'probably beatable', not proven. See the About tab.")


def _set_state(key: str, value) -> None:
    """Button callback: write a widget's value before the rerun instantiates the widget."""
    st.session_state[key] = value


def _roll_seed() -> None:
    """Dice callback: a fresh random seed (the number_input with key='seed' picks it up on the rerun)."""
    st.session_state["seed"] = random.randint(0, 10**9)


def studio() -> None:
    try:
        svc = get_service()
    except Exception as e:  # noqa: BLE001 - no checkpoint yet: say so plainly instead of crashing
        st.error(f"No model is available yet: {e}")
        st.info("Train or select a checkpoint, then set T2M_CKPT. The Gallery tab still shows pre-made levels.")
        return
    with st.sidebar:
        st.header("Describe a level")
        if "prompt" not in st.session_state:
            st.session_state.prompt = EXAMPLES[0]
        for i, ex in enumerate(EXAMPLES):
            # on_click callbacks run BEFORE the script reruns, so they may set a widget's key; setting it after the widget exists raises
            # StreamlitWidgetAlreadyInstantiatedError (the Dice crash).
            st.button(ex, key=f"ex{i}", width="stretch", on_click=_set_state, args=("prompt", ex))
        st.text_area("Your sentence", key="prompt", height=110)
        w = st.slider("Prompt strength (CFG w)", float(CFG["cfg"]["min_w"]), float(CFG["cfg"]["max_w"]), float(CFG["cfg"]["default_w"]), 0.5)
        width = st.select_slider("Width (tiles of generated content)", options=list(CFG["widths"]), value=CFG["default_width"])
        if "seed" not in st.session_state:
            st.session_state.seed = 1234
        c1, c2 = st.columns([3, 1])
        c1.number_input("Seed", min_value=0, max_value=2**53 - 1, key="seed", step=1)
        c2.button("Dice", on_click=_roll_seed)
        fast = not st.toggle("High quality (50 steps, resampling: slower)", value=False, disabled=svc.fast_forced,
                             help="Fast mode (25 steps) is the default and is forced without a GPU; high quality takes about a minute or more.")
        go = st.button("Generate & Play", type="primary", width="stretch")
    st.title("text2mario: type it, play it")
    norm = svc.preview(st.session_state.prompt)
    with st.expander("What the model will understand", expanded=not go):
        if not norm.scenes:
            st.write("No recognised level words. An unconditioned level will be generated.")
        for i, s in enumerate(norm.scenes, 1):
            st.markdown(f"**Scene {i}:** `{s.caption}`" + (f"  - avoid: {', '.join(s.forbid)}" if s.forbid else ""))
            notes = sorted({explain_warning(w) for w in s.warnings} - {""})
            if notes:
                st.caption("Notes: " + " ".join(notes))
        notes = sorted({explain_warning(w) for w in norm.warnings} - {""})
        if notes:
            st.caption("Overall: " + " ".join(notes))
    ok, msg = validate_prompt(st.session_state.prompt)
    if not ok:
        st.error(msg)
    elif go and not norm.scenes:
        st.warning("No recognised level words in that sentence; the model will make an unconditioned level. "
                   "Try words like pipe, enemy, coin, platform, staircase, gap, cannon, question block.")
    if go and ok:
        with st.spinner("Generating, checking, and re-sampling if needed..."):
            try:
                res = svc.run(st.session_state.prompt, seed=int(st.session_state.seed), w=w, width=width, fast=fast)
                st.session_state.result = res
                st.session_state.result_prompt = st.session_state.prompt
            except ValueError as e:                 # validation problems are shown, never a stack trace
                st.error(str(e))
            except Exception as e:  # noqa: BLE001 - unexpected failure: say so, keep the app alive
                st.error(f"Generation failed ({type(e).__name__}). Please try another seed or a shorter prompt.")
    res = st.session_state.get("result")
    if res is None:
        st.info("Pick an example or type a sentence, then press Generate & Play.")
        return
    lvl = res.level
    shown = st.session_state.get("result_prompt")
    if shown is not None and shown != st.session_state.prompt:
        st.info("The level below was generated for an earlier sentence (shown here as plain text). Press Generate & Play to make one for the current sentence.")
        st.code(shown, language=None)           # plain text: the user's words are never interpreted as markdown
    adherence_notice(lvl)
    badge(lvl)
    t = res.seconds
    st.caption(f"{lvl['size']['cols']} columns | {t['generate_and_verify']} s to generate and verify | seed {lvl['generation']['seed']} "
               f"| steps {lvl['generation']['ddim_steps']} | attempts {lvl['verification']['attempts_total']}")
    embed_game(res.files["html"].decode())
    tab_a, tab_b, tab_c = st.tabs(["Picture", "Details", "Download"])
    with tab_a:
        st.image(res.files["png"], width="stretch")
    with tab_b:
        st.markdown("**Caption each 16-column window was actually given** (the model describes windows, not whole levels):")
        for k, sw in enumerate(lvl["prompt"]["scenes"]):
            st.markdown(f"- window {k + 1} (columns {sw['col_from']}-{sw['col_to']}, scene {sw.get('scene', 0) + 1}): `{sw['caption']}`")
        st.json({k: lvl[k] for k in ("prompt", "sanitizer", "verification", "generation")}, expanded=False)
    with tab_c:
        lid = lvl["level_id"]
        for label, key, mime, ext in (("Playable HTML (offline)", "html", "text/html", "html"), ("Level JSON", "json", "application/json", "level.json"),
                                      ("PNG", "png", "image/png", "png"), ("CSV tile ids", "csv", "text/csv", "ids.csv"),
                                      ("VGLC text", "vglc", "text/plain", "vglc16.txt")):
            st.download_button(label, res.files[key], file_name=f"t2m_{lid}.{ext}", mime=mime)


def gallery() -> None:
    st.title("Showcase levels")
    man = ROOT / "app/showcase/manifest.json"
    if not man.exists():
        st.info("No showcase levels have been generated yet (scripts/make_showcase.py).")
        return
    from text2mario.render.game_html import build_game_html
    items = json.loads(man.read_text())["levels"]
    pick = st.selectbox("Level", [f"{i['prompt']}  (seed {i['seed']})" for i in items])
    it = items[[f"{i['prompt']}  (seed {i['seed']})" for i in items].index(pick)]
    lvl = json.loads((ROOT / "app/showcase" / it["file"]).read_text())
    adherence_notice(lvl)
    badge(lvl)
    embed_game(build_game_html(lvl))


def about() -> None:
    st.title("About")
    st.markdown(
        "**Model:** a small U-Net denoising-diffusion model (18.3 M parameters) over 16x16 tile windows with a frozen "
        "MiniLM text encoder; levels are built by sliding-window outpainting with a stride of 8 columns.\n\n"
        "**Data:** the MarioDiffusion corpus (Super Mario Bros 1 and 2, VGLC). Idea and caption grammar credit: Lee & Simo-Serra "
        "(MIT). The caption detector and this checker were written from the descriptions, not ported.\n\n"
        "**Art:** Kenney Pixel Platformer (CC0) and Phaser 3.90.0 (MIT). No Nintendo sprites are included.\n\n"
        "**Honest limits:** the model sees 31 training levels; 'route found' means *our checker found a path under the game's "
        "physics*, with enemies assumed killable by a stomp; prompts outside the training vocabulary are clamped or ignored "
        "(the preview tells you).")
    for f in ("FROZEN.json", "reports/final/RESULTS.md"):
        p = ROOT / f
        if p.exists():
            st.subheader(f)
            st.code(p.read_text()[:3000])


pages = [st.Page(studio, title="Studio", icon=":material/sports_esports:", default=True),
         st.Page(gallery, title="Gallery", icon=":material/collections:"),
         st.Page(about, title="About", icon=":material/info:")]
st.navigation(pages).run()
