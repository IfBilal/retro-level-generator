"""Type the audit-2 prompts into a running app (local or the deployed URL) and record what the page says: python scripts/audit2_deployed_prompts.py URL

Explain-me: what = open the app in headless Chromium (waking a sleeping Community Cloud app), type each prompt, press Generate & Play, wait for
the page to settle and keep the visible text; in/out = URL -> evidence/audit2/deployed_prompts.json; try = add a prompt to PROMPTS.
Each prompt has a substring the page must show (a refusal sentence or the count line) and one it must not (a red traceback box)."""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    ("three pipes and 5 enemies, 2 enemies before the first pipe and the rest after the 4th pipe", "no 4th pipe"),
    ("flatland", "Flat level confirmed"),
    ("no enemies", "enemy: asked none"),
    ("castle with lava", "could not find anything"),
    ("tall pipe at the start and cannon at the end", "cannon (part 2): asked 1"),
    ("full floor with two pipes and a few enemies", "enemy: asked 3-4"),
]


def main():
    url = sys.argv[1]
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(url, wait_until="domcontentloaded", timeout=120000)
        wake = pg.get_by_role("button", name="Yes, get this app back up!")
        try:
            wake.wait_for(state="visible", timeout=8000)
            wake.click()
        except Exception:
            pass

        def app():
            return next((f for f in pg.frames if "/~/+/" in f.url), pg)
        t_end = time.time() + 300
        while time.time() < t_end and app().get_by_role("button", name="Generate & Play").count() == 0:
            pg.wait_for_timeout(1000)
        for text, must in PROMPTS:
            ta = app().locator("textarea").first
            ta.fill(text)
            ta.press("Control+Enter")
            app().get_by_role("button", name="Generate & Play").click()
            body, t0 = "", time.time()
            while time.time() - t0 < 120:
                pg.wait_for_timeout(1500)
                body = app().inner_text("body")
                if must in body or "Traceback" in body:
                    break
            ok = must in body and "Traceback" not in body
            rows.append({"prompt": text, "expect": must, "ok": ok, "seconds": round(time.time() - t0, 1)})
            print(("PASS " if ok else "FAIL ") + text[:70], rows[-1]["seconds"], "s")
        b.close()
    (ROOT / "evidence/audit2/deployed_prompts.json").write_text(json.dumps({"url": url, "rows": rows}, indent=1), encoding="utf-8")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
