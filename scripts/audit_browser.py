"""Real-browser audit of the running app (local or deployed): python scripts/audit_browser.py URL [--out DIR] [--dice N] [--viewport WxH]

Explain-me:
  What: opens the app in headless Chromium and drives it like a user: clicks Dice N times (the audit P1 crash), clicks every example
        button, generates a level, waits for the badge, checks the nested game iframe, and records every page error, console error,
        failed request and Streamlit exception box.  Screenshots + a JSON report are written to --out.
  In/Out: URL -> evidence/audit/<UTC>/browser_<name>.json and PNGs.  Exit code 1 if any defect was recorded.
  Previous: scripts/gate_p8.py (five smoke checks).  Next: evidence/audit/INDEX.md.
  Try: python scripts/audit_browser.py http://localhost:8501 --dice 30 --viewport 390x844
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from playwright.sync_api import sync_playwright

BAD_TEXT = ("StreamlitWidgetAlreadyInstantiatedError", "Traceback", "StreamlitAPIException", "Generation failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dice", type=int, default=30)
    ap.add_argument("--viewport", default="1400x1000")
    ap.add_argument("--name", default="run")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--offline-after-load", action="store_true", help="block all non-local requests after the first load")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "evidence/audit" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir(parents=True, exist_ok=True)
    vw, vh = (int(x) for x in a.viewport.split("x"))
    defects, log = [], {"url": a.url, "viewport": a.viewport, "steps": []}

    def step(name, **kw):
        log["steps"].append({"step": name, "t": round(time.time() - t0, 1), **kw})

    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": vw, "height": vh})
        pg = ctx.new_page()
        console, failed = [], []
        pg.on("pageerror", lambda e: defects.append(f"pageerror: {e}"))
        pg.on("console", lambda m: console.append((m.type, m.text[:300])) if m.type in ("error", "warning") else None)
        pg.on("requestfailed", lambda r: failed.append((r.url[:200], str(r.failure))))
        t0 = time.time()
        pg.goto(a.url, wait_until="domcontentloaded", timeout=120000)

        def app():
            for f in pg.frames:
                if "/~/+/" in f.url:
                    return f
            return pg

        end = time.time() + a.timeout
        while time.time() < end and app().locator("text=Generate & Play").count() == 0:
            pg.wait_for_timeout(1000)
        step("loaded")
        pg.screenshot(path=str(out / f"{a.name}_01_loaded.png"))

        def check(tag):
            body = app().inner_text("body")
            for bad in BAD_TEXT:
                if bad in body:
                    defects.append(f"{tag}: page shows '{bad}'")
                    pg.screenshot(path=str(out / f"{a.name}_DEFECT_{tag}.png"))
                    return False
            return True

        # P1: Dice, repeatedly
        dice_ok = 0
        for i in range(a.dice):
            app().get_by_role("button", name="Dice").click()
            pg.wait_for_timeout(700)
            dice_ok += check(f"dice{i}")
        step("dice", clicks=a.dice, ok=dice_ok)
        seeds = app().locator("input[type=number]").first.input_value()
        step("seed_after_dice", seed=seeds)
        # examples
        for i in range(4):
            btn = app().locator("button[kind=secondary]").nth(i)
            try:
                label = " ".join(btn.inner_text().split())
                btn.click()
                pg.wait_for_timeout(900)
                check(f"example{i}")
                filled = app().locator("textarea").first.input_value()
                if " ".join(filled.split()) != label:               # the chip must put exactly its own sentence into the prompt box
                    defects.append(f"example{i}: prompt box holds '{filled}' instead of '{label}'")
            except Exception as e:
                defects.append(f"example{i}: {e}")
        step("examples")
        # generate
        app().get_by_role("button", name="Generate & Play").click()
        t1 = time.time()
        try:
            app().wait_for_selector("text=/Route found|No route found/", timeout=int(a.timeout * 1000))
            step("generated", seconds=round(time.time() - t1, 1))
        except Exception:
            defects.append("generate: no result badge before timeout")
        pg.wait_for_timeout(3000)
        check("after_generate")
        pg.screenshot(path=str(out / f"{a.name}_02_generated.png"), full_page=True)
        n_if = app().locator("iframe").count()
        step("game_iframe", count=n_if)
        if n_if < 1:
            defects.append("no game iframe")
        else:
            gf = next((f for f in pg.frames if f not in (pg.main_frame,) and "/~/+/" not in f.url and "statuspage" not in f.url), None)
            if gf is not None:
                try:
                    gf.wait_for_selector("canvas", timeout=20000)
                    nonblank = gf.evaluate("""() => { const c = document.querySelector('canvas'); const t = document.createElement('canvas');
                      t.width = 64; t.height = 64; const x = t.getContext('2d'); x.drawImage(c, 0, 0, 64, 64);
                      const d = x.getImageData(0, 0, 64, 64).data; let s = new Set(); for (let i = 0; i < d.length; i += 4) s.add(d[i] + ',' + d[i+1] + ',' + d[i+2]);
                      return s.size; }""")
                    step("canvas_distinct_colours", n=nonblank)
                    if nonblank < 3:
                        defects.append("game canvas looks blank")
                except Exception as e:
                    defects.append(f"game canvas: {e}")
        # horizontal overflow check
        ov = pg.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        step("horizontal_overflow_px", px=ov)
        b.close()
    log["console"] = console[:40]
    log["failed_requests"] = failed[:40]
    for t, m in console:
        if t == "error":
            defects.append(f"console error: {m}")
    for u, why in failed:
        defects.append(f"failed request: {u} ({why})")
    log["defects"] = defects
    (out / f"browser_{a.name}.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({"defects": defects, "steps": log["steps"]}, indent=1)[:3000])
    return 1 if defects else 0


if __name__ == "__main__":
    sys.exit(main())
