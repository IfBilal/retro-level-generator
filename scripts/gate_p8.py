"""Gate G-P8: the DEPLOYED app works (doc 24 §G-P8).  Run it against your Streamlit Community Cloud URL after deploying.

Explain-me:
  What: opens the URL in headless Chromium, waits for the app, clicks Generate & Play with the default example, waits for the result badge,
        and checks: the page loads, a result appears within the time limit, the embedded game iframe exists, there are no uncaught page errors, an empty
        prompt shows the validation message, and a screenshot is saved.  Load time (cold start after sleep) and generation time are recorded.
  In/Out: python scripts/gate_p8.py https://YOUR-APP.streamlit.app  -> reports/gates/P8/RESULT.json + evidence/deployment/screenshot.png
  Try: python scripts/gate_p8.py http://localhost:8501 --out-dir reports/gates/_p8_local   (a local dry run is not a deployment result)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from playwright.sync_api import sync_playwright  # noqa: E402

from text2mario.utils.gates import git_sha, write_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else ROOT / "reports/gates/P8"
    evid = out if a.out_dir else ROOT / "evidence/deployment"
    evid.mkdir(parents=True, exist_ok=True)
    errs, res = [], {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 1000})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        t0 = time.time()
        pg.goto(a.url, wait_until="domcontentloaded", timeout=120000)
        # Streamlit Community Cloud serves the app inside an iframe whose URL contains "/~/+/"; locally there is no such frame.
        def app():
            for f in pg.frames:
                if "/~/+/" in f.url:
                    return f
            return pg

        deadline = time.time() + a.timeout
        while time.time() < deadline and app().get_by_role("button", name="Generate & Play").count() == 0:
            pg.wait_for_timeout(1000)
        app().get_by_role("button", name="Generate & Play").wait_for(timeout=5000)
        res["load_seconds"] = round(time.time() - t0, 1)
        app().get_by_role("button", name="Generate & Play").click()
        t1 = time.time()
        try:
            app().wait_for_selector("text=/Route found|No route found/", timeout=int(a.timeout * 1000))
            res["generate_seconds"] = round(time.time() - t1, 1)
            res["badge"] = True
        except Exception:
            res["badge"] = False
        pg.wait_for_timeout(2500)
        res["iframe"] = app().locator("iframe").count() >= 1
        pg.screenshot(path=str(evid / "screenshot.png"))
        app().locator("textarea").first.fill("   ")
        app().get_by_role("button", name="Generate & Play").click()
        pg.wait_for_timeout(2500)
        res["validation_message"] = "describing the level" in app().inner_text("body")
        b.close()
    checks = [{"id": "P8.01", "name": "app loads", "value": res.get("load_seconds"), "pass": "load_seconds" in res},
              {"id": "P8.02", "name": "generation produces a result badge", "value": res.get("generate_seconds"), "pass": bool(res.get("badge"))},
              {"id": "P8.03", "name": "game iframe present", "value": res.get("iframe"), "pass": bool(res.get("iframe"))},
              {"id": "P8.04", "name": "empty prompt shows the validation message", "value": res.get("validation_message"), "pass": bool(res.get("validation_message"))},
              {"id": "P8.05", "name": "no uncaught page errors", "value": len(errs), "pass": not errs}]
    write_result("P8", all(c["pass"] for c in checks), {"url": a.url, **res, "git_sha": git_sha()}, checks, out_dir=out, page_errors=errs[:5])
    for c in checks:
        print(c["id"], c["value"], "PASS" if c["pass"] else "FAIL")


if __name__ == "__main__":
    main()
