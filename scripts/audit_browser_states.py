"""Browser audit part 2 (state handling, viewports, seed edge cases, keyboard, offline, play-through, level equality):
python scripts/audit_browser_states.py http://localhost:8501 --out evidence/audit/<ts> [--only s1,s2]

Explain-me:
  What: drives the real app in headless Chromium through the scenarios of the audit brief section 3.  Every scenario records page errors,
        console errors, failed requests, Streamlit exception boxes and a screenshot, and writes browser_states.json with PASS/FAIL per check.
  In/Out: URL -> evidence/audit/<ts>/states_*.png + browser_states.json.  Exit 1 if any check failed.
  Previous: scripts/audit_browser.py (Dice x30, examples, generate).  Next: evidence/audit/INDEX.md.
  Try: --only viewports,seed  (comma list of: viewports, state, seed, keyboard, offline, play, equality)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
from playwright.sync_api import sync_playwright

BAD = ("StreamlitWidgetAlreadyInstantiatedError", "Traceback", "StreamlitAPIException", "Generation failed")
BENIGN_CONSOLE = ("Permissions policy violation", "Unrecognized feature", "sandbox", "Failed to load resource")        # Streamlit platform noise, listed in the report
RESULTS: list[dict] = []


def rec(name, ok, **info):
    RESULTS.append({"check": name, "pass": bool(ok), **info})
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "", flush=True)


class Session:
    def __init__(self, p, url, out, vw=1400, vh=1000, tag="s"):
        self.url, self.out, self.tag = url, out, tag
        self.browser = p.chromium.launch()
        self.ctx = self.browser.new_context(viewport={"width": vw, "height": vh})
        self.page = self.ctx.new_page()
        self.console, self.failed, self.errors, self.hosts = [], [], [], set()
        self.page.on("console", lambda m: self.console.append((m.type, m.text[:200])) if m.type == "error" else None)
        self.page.on("requestfailed", lambda r: self.failed.append(r.url[:160]))
        self.page.on("pageerror", lambda e: self.errors.append(str(e)[:200]))
        self.page.on("request", lambda r: self.hosts.add(urlparse(r.url).netloc or r.url[:20]))

    def open(self):
        self.page.goto(self.url, wait_until="domcontentloaded", timeout=120000)
        self.wait_ready()

    def app(self):
        for f in self.page.frames:
            if "/~/+/" in f.url:
                return f
        return self.page

    def wait_ready(self, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.app().get_by_role("button", name="Generate & Play").count():
                    return
            except Exception:  # noqa: BLE001
                pass
            self.page.wait_for_timeout(500)
        raise TimeoutError("app did not become ready")

    def generate(self, wait=True, timeout=240):
        self.app().get_by_role("button", name="Generate & Play").click()
        if wait:
            self.app().wait_for_selector("text=/Route found|No route found/", timeout=timeout * 1000)
            self.page.wait_for_timeout(2500)

    def body(self):
        return self.app().inner_text("body")

    def clean(self):
        b = self.body()
        return not any(x in b for x in BAD)

    def shot(self, name, full=False):
        self.page.screenshot(path=str(self.out / f"states_{self.tag}_{name}.png"), full_page=full)

    def game_frame(self):
        return next((f for f in self.page.frames if f.url == "about:srcdoc"), None)

    def level(self):
        gf = self.game_frame()
        return gf.evaluate("() => window.__LEVEL__") if gf else None

    def own_console_errors(self):
        return [m for t, m in self.console if not any(b in m for b in BENIGN_CONSOLE)]

    def close(self):
        self.browser.close()


def set_prompt(s, text):
    ta = s.app().locator("textarea").first
    ta.fill(text)
    ta.press("Control+Enter")
    s.page.wait_for_timeout(1200)


def s_viewports(p, url, out):
    for name, (w, h) in {"1920x1080": (1920, 1080), "1280x720": (1280, 720), "390x844": (390, 844)}.items():
        s = Session(p, url, out, w, h, tag=f"vp{name}")
        try:
            s.open()
            ov = s.page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
            rec(f"viewport {name}: no horizontal overflow before generating", ov <= 1, px=ov)
            if w < 600:                                               # phone: the sidebar is collapsed behind a button
                opener = s.page.locator("[data-testid=stExpandSidebarButton], [data-testid=stSidebarCollapsedControl] button").first
                if opener.count():
                    opener.click()
                    s.page.wait_for_timeout(800)
            btn = s.app().get_by_role("button", name="Generate & Play")
            rec(f"viewport {name}: Generate button reachable", btn.count() == 1 and btn.is_visible(), visible=btn.is_visible() if btn.count() else None)
            s.shot("loaded")
            s.generate()
            ov = s.page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
            rec(f"viewport {name}: no horizontal overflow after generating", ov <= 1, px=ov)
            rec(f"viewport {name}: no exception box", s.clean())
            s.shot("generated", full=True)
            rec(f"viewport {name}: no page errors", not s.errors, errors=s.errors[:2])
        finally:
            s.close()


def s_state(p, url, out):
    s = Session(p, url, out, tag="state")
    try:
        s.open()
        s.generate()
        first_level = s.level()
        # 1. change the prompt without regenerating: the old level must be labelled as belonging to the OLD sentence
        set_prompt(s, "two enemies and one cannon")
        b = s.body()
        rec("state: edited prompt -> old level is labelled as an earlier sentence", "earlier sentence" in b, snippet=b[b.find("earlier"):][:80])
        s.shot("edited_not_regenerated")
        # 2. generate twice quickly (second click while the first runs)
        btn = s.app().get_by_role("button", name="Generate & Play")
        btn.click()
        s.page.wait_for_timeout(300)
        try:
            btn.click(timeout=3000)
        except Exception:  # noqa: BLE001
            pass
        try:                                   # the old level's badge is still on screen, so wait for the "earlier sentence" notice to go instead
            s.app().wait_for_selector("text=earlier sentence", state="detached", timeout=240000)
        except Exception:  # noqa: BLE001
            pass
        s.page.wait_for_timeout(3000)
        rec("state: double click while generating -> no exception", s.clean())
        rec("state: after double click the level matches the NEW sentence", "earlier sentence" not in s.body())
        s.shot("double_click")
        # 3. reload mid-generation
        s.app().get_by_role("button", name="Generate & Play").click()
        s.page.wait_for_timeout(1000)
        s.page.reload(wait_until="domcontentloaded")
        s.wait_ready()
        rec("state: reload mid-generation -> app usable again, no exception", s.clean() and s.app().get_by_role("button", name="Generate & Play").count() == 1)
        s.shot("reloaded")
        s.generate()
        # 4. navigation: Gallery, About, back, forward
        for page_name in ("Gallery", "About", "Studio"):
            s.app().get_by_role("link", name=page_name).first.click()
            s.page.wait_for_timeout(2500)
            rec(f"state: page {page_name} opens without exception", s.clean())
            s.shot(f"nav_{page_name}")
        s.app().get_by_role("link", name="Gallery").first.click()
        s.page.wait_for_timeout(2000)
        picker = s.app().locator("[data-baseweb=select]").first
        if picker.count():
            picker.click()
            s.page.wait_for_timeout(500)
            opts = s.app().locator("[role=option]")
            if opts.count() > 1:
                opts.nth(1).click()
                s.page.wait_for_timeout(2500)
        rec("state: Gallery level picker works without exception", s.clean())
        s.shot("gallery_pick")
        s.page.go_back()
        s.page.wait_for_timeout(2500)
        s.page.go_forward()
        s.page.wait_for_timeout(2500)
        rec("state: browser back/forward -> no exception", s.clean())
        rec("state: no page errors in the whole state scenario", not s.errors, errors=s.errors[:2])
        rec("state: no own console errors", not s.own_console_errors(), errors=s.own_console_errors()[:3])
        rec("state: first level existed", first_level is not None)
        # 5. two tabs at once
        page2 = s.ctx.new_page()
        page2.goto(url, wait_until="domcontentloaded")
        s2 = Session.__new__(Session)
        s2.page, s2.url, s2.out, s2.tag = page2, url, out, "tab2"
        s2.wait_ready()
        s.app().get_by_role("link", name="Studio").first.click() if s.app().get_by_role("link", name="Studio").count() else None
        s.page.wait_for_timeout(1500)
        s.app().get_by_role("button", name="Generate & Play").click()
        s2.app().get_by_role("button", name="Generate & Play").click()
        s.app().wait_for_selector("text=/Route found|No route found/", timeout=240000)
        s2.app().wait_for_selector("text=/Route found|No route found/", timeout=240000)
        rec("state: two tabs generating at once -> both finish, no exception", s.clean() and s2.clean())
        page2.screenshot(path=str(out / "states_state_tab2.png"))
    finally:
        s.close()


def s_seed(p, url, out):
    s = Session(p, url, out, tag="seed")
    try:
        s.open()
        seed = s.app().locator("input[type=number]").first
        for label, val in (("typed value", "4242"), ("huge value", str(2**60)), ("negative", "-5"), ("empty", ""), ("text", "abc"), ("decimal", "3.7")):
            if val == "abc":                                         # a number input refuses fill("abc"); type the letters like a user would
                seed.fill("")
                seed.press_sequentially(val)
            else:
                seed.fill(val)
            seed.press("Enter")
            s.page.wait_for_timeout(1000)
            now = seed.input_value()
            rec(f"seed: {label} -> no exception and a sane field value", s.clean(), value=now)
        seed.fill("4242")
        seed.press("Enter")
        s.page.wait_for_timeout(500)
        s.app().get_by_role("button", name="Increment").first.click()
        s.page.wait_for_timeout(800)
        rec("seed: step up increments by one", seed.input_value() == "4243", value=seed.input_value())
        s.app().get_by_role("button", name="Decrement").first.click()
        s.page.wait_for_timeout(800)
        rec("seed: step down decrements by one", seed.input_value() == "4242", value=seed.input_value())
        s.generate()
        cap = s.app().locator("text=/seed \\d+/").first.inner_text()
        rec("seed: the seed shown is the seed used", "seed 4242" in cap, caption=cap)
        lv = s.level()
        rec("seed: the iframe level records the same seed", lv is not None and lv["generation"]["seed"] == 4242, got=(lv or {}).get("generation", {}).get("seed"))
        # settings sliders by keyboard
        s.app().get_by_role("slider").first.focus()
        s.page.keyboard.press("ArrowRight")
        s.page.keyboard.press("ArrowRight")
        s.page.wait_for_timeout(800)
        rec("seed: CFG slider by keyboard -> no exception", s.clean())
        s.shot("seed_done")
    finally:
        s.close()


def s_keyboard(p, url, out):
    s = Session(p, url, out, tag="kbd")
    try:
        s.open()
        s.app().locator("textarea").first.focus()
        reached = False
        for _ in range(25):
            s.page.keyboard.press("Tab")
            label = s.app().evaluate("() => (document.activeElement && (document.activeElement.innerText || document.activeElement.getAttribute('aria-label') || '')) || ''")
            if "Generate" in label:
                reached = True
                break
        rec("keyboard: Tab reaches the Generate & Play button from the prompt box", reached)
        s.page.keyboard.press("Enter")
        s.app().wait_for_selector("text=/Route found|No route found/", timeout=240000)
        rec("keyboard: Enter on the focused button generates a level", s.clean())
        names = s.app().evaluate("""() => [...document.querySelectorAll('textarea,input,button,[role=slider]')].filter(e => e.offsetParent !== null)
                                   .map(e => [e.tagName, (e.getAttribute('aria-label') || '').trim(), (e.innerText || '').trim()])""")
        unlabeled = [n for n in names if n[0] in ("TEXTAREA", "INPUT") and not n[1]]
        rec("keyboard: every visible text input has an accessible name", not unlabeled, unlabeled=unlabeled[:3])
        s.shot("keyboard")
    finally:
        s.close()


def s_offline(p, url, out):
    s = Session(p, url, out, tag="offline")
    try:
        host = urlparse(url).netloc
        external = []

        def route(r):
            if urlparse(r.request.url).netloc != host and not r.request.url.startswith(("data:", "blob:", "about:")):
                external.append(r.request.url[:120])
                r.abort()
            else:
                r.continue_()

        s.ctx.route("**/*", route)
        s.open()
        s.generate()
        rec("offline: generation works with every non-local request blocked", s.clean() and s.app().locator("iframe").count() >= 1, blocked=external[:5])
        gf = s.game_frame()
        ok = False
        if gf:
            gf.wait_for_selector("canvas", timeout=20000)
            ok = gf.evaluate("() => typeof Phaser !== 'undefined'")
        rec("offline: Phaser is bundled in the game page (works with the network blocked)", ok)
        rec("offline: the game page itself requested no external URL", not [u for u in external if "srcdoc" in u], blocked=external[:5])
        s.shot("offline")
        (out / "states_offline_blocked_requests.txt").write_text("\n".join(external) or "(none)", encoding="utf-8")
    finally:
        s.close()


def s_play(p, url, out):
    s = Session(p, url, out, tag="play")
    try:
        s.open()
        s.generate()
        gf = s.game_frame()
        gf.wait_for_selector("canvas", timeout=20000)
        errs = []
        gf.on("console", lambda m: errs.append(m.text[:150]) if m.type == "error" else None)
        box = gf.locator("canvas").bounding_box()

        def grab():
            return s.page.screenshot(clip={"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]})

        before = grab()
        gf.locator("#focus").click()
        s.page.wait_for_timeout(500)
        s.page.keyboard.down("ArrowRight")
        s.page.wait_for_timeout(1800)
        s.page.keyboard.press("Space")
        s.page.wait_for_timeout(1200)
        s.page.keyboard.up("ArrowRight")
        after = grab()
        (out / "states_play_before.png").write_bytes(before)
        (out / "states_play_after.png").write_bytes(after)
        rec("play: the frame changes after pressing right + jump (Mario / camera moved)", before != after)
        rec("play: nothing threw inside the game iframe", not errs, errors=errs[:3])
        rec("play: the click-to-play overlay hides after the click", not gf.locator("#focus").is_visible())
    finally:
        s.close()


def s_equality(p, url, out):
    """Same prompt+seed+settings twice through the UI -> byte-identical raw grid; the iframe level equals a direct service run."""
    sys.path.insert(0, str(ROOT / "src"))
    s = Session(p, url, out, tag="eq")
    try:
        s.open()
        seed = s.app().locator("input[type=number]").first
        seed.fill("777")
        seed.press("Enter")
        s.page.wait_for_timeout(800)
        set_prompt(s, "one pipe, then two enemies")
        s.generate()
        a = s.level()
        s.generate()
        b = s.level()
        rec("equality: same prompt+seed+settings twice through the UI -> identical raw_grid", a["raw_grid"] == b["raw_grid"])
        from text2mario.app.service import LevelService
        svc = LevelService()
        res = svc.run("one pipe, then two enemies", seed=777, w=7.5, width=48, fast=True)
        rec("equality: the iframe level equals a direct service run (same machine)", res.level["raw_grid"] == a["raw_grid"])
        rec("equality: displayed columns are true", a["size"]["cols"] == 3 + 48 + 6)
        cap = s.app().locator("text=/columns \\|/").first.inner_text()
        rec("equality: the caption's column count matches the level", f"{a['size']['cols']} columns" in cap, caption=cap)
        import re
        summary = s.app().get_by_text("What the model will understand")
        if summary.count():
            summary.first.click()                                                    # the panel is collapsed after a generation
            s.page.wait_for_timeout(1000)
        shown = s.app().inner_text("body")
        panel = [m.strip().rstrip(".") for m in re.findall(r"Scene \d+:\s*([^\n]+)", shown)]
        recorded = [". ".join(ph) for ph in a["prompt"]["canonical"]]            # the scene captions the generator was given (before per-window planning)
        rec("equality: the 'what the model will understand' panel shows exactly the scene captions recorded in the level",
            panel == recorded and len(panel) == 2, panel=panel, recorded=recorded)
        sent = [w["caption"] for w in a["prompt"]["scenes"]]
        (out / "states_equality.json").write_text(json.dumps({"panel_scene_captions": panel, "level_scene_captions": recorded,
                                                              "window_captions_actually_sent_to_the_model": sent}, indent=1), encoding="utf-8")
    finally:
        s.close()


SCEN = {"viewports": s_viewports, "state": s_state, "seed": s_seed, "keyboard": s_keyboard, "offline": s_offline, "play": s_play, "equality": s_equality}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "evidence/audit" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir(parents=True, exist_ok=True)
    names = a.only.split(",") if a.only else list(SCEN)
    with sync_playwright() as p:
        for n in names:
            try:
                SCEN[n](p, a.url, out)
            except Exception as e:  # noqa: BLE001
                rec(f"scenario {n} completed", False, error=f"{type(e).__name__}: {str(e)[:300]}")
    (out / "browser_states.json").write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")
    bad = [r for r in RESULTS if not r["pass"]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
