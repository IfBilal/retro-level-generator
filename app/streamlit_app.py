"""text2mario - "Type it, play it": a sentence in, a verified playable level out (doc 28).

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

import streamlit as st  # noqa: E402

st.set_page_config(page_title="text2mario", page_icon=":video_game:", layout="wide")

EXAMPLES = [
    "full floor with two pipes and a few enemies",
    "flat ground, one pipe, then several coins and a staircase",
    "an underground level: three platforms, then a cannon and two enemies",
    "mostly pits with two islands of ground, then a coin line",
]


@st.cache_resource(show_spinner="Loading the model...")
def get_service():
    from text2mario.app.service import LevelService
    return LevelService()


from text2mario.app.service import validate_prompt  # noqa: E402


def embed_game(html: str, height: int = 620) -> None:
    """Render the self-contained game HTML in an iframe that can take keyboard focus (doc 28 §2)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height, tab_index=0, alt="Playable generated level")
    else:  # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def badge(level: dict) -> None:
    v = level["verification"]
    if v["status"] == "verified":
        st.success("Route found: the playability checker found a path from start to flag (not a guarantee).")
    else:
        st.warning(f"Not verified ({v.get('reason') or 'checker found no path'}). You can still play it.")
    st.caption("'Route found' means our checker found a path under the game's physics (enemies assumed stompable). It matches the game engine on "
               "micro-levels, but an automatic test player finishes only part of generated levels end to end (see the report), so treat it as "
               "'probably beatable', not proven. See the About tab.")


def studio() -> None:
    try:
        svc = get_service()
    except Exception as e:  # no checkpoint yet: say so plainly instead of crashing
        st.error(f"No model is available yet: {e}")
        st.info("Train or select a checkpoint, then set T2M_CKPT. The Gallery tab still shows pre-made levels.")
        return
    with st.sidebar:
        st.header("Describe a level")
        if "prompt" not in st.session_state:
            st.session_state.prompt = EXAMPLES[0]
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", width="stretch"):
                st.session_state.prompt = ex
        st.text_area("Your sentence", key="prompt", height=110)
        w = st.slider("Prompt strength (CFG w)", 1.0, 10.0, float(st.session_state.get("default_w", 5.0)), 0.5)
        width = st.select_slider("Width (tiles of generated content)", options=[16, 48, 80, 112], value=48)
        if "seed" not in st.session_state:
            st.session_state.seed = 1234
        c1, c2 = st.columns([3, 1])
        c1.number_input("Seed", min_value=0, max_value=2**53 - 1, key="seed", step=1)
        if c2.button("Dice"):
            st.session_state.seed = random.randint(0, 10**9)
            st.rerun()
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
            if s.warnings:
                st.caption("notes: " + ", ".join(sorted(set(s.warnings))))
        if norm.warnings:
            st.caption("overall: " + ", ".join(norm.warnings))
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
            except ValueError as e:                 # validation problems are shown, never a stack trace
                st.error(str(e))
            except Exception as e:                  # unexpected failure: say so, keep the app alive
                st.error(f"Generation failed ({type(e).__name__}). Please try another seed or a shorter prompt.")
    res = st.session_state.get("result")
    if res is None:
        st.info("Pick an example or type a sentence, then press Generate & Play.")
        return
    lvl = res.level
    badge(lvl)
    t = res.seconds
    st.caption(f"{lvl['size']['cols']} columns | {t['generate_and_verify']} s to generate and verify | seed {lvl['generation']['seed']} "
               f"| steps {lvl['generation']['ddim_steps']} | attempts {lvl['verification']['attempts_total']}")
    embed_game(res.files["html"].decode())
    tab_a, tab_b, tab_c = st.tabs(["Picture", "Details", "Download"])
    with tab_a:
        st.image(res.files["png"], width="stretch")
    with tab_b:
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
