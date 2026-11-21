"""Build slides/slides.pdf (12 slides) from the saved results: python scripts/build_slides.py

Explain-me:
  What: 12 landscape slides (problem, motivation, solution, technique, implementation, testing, results qualitative + quantitative,
        limitations, conclusion) as HTML, printed to PDF with headless Chromium.  Numbers come from build_report.values().
  In/Out: report/figures + reports/** -> slides/slides.html, slides/slides.pdf.  Previous: scripts/make_figures.py, build_report.py.
  Try: edit SLIDES below; every {{key}} is looked up in build_report.values().
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_report as BR  # noqa: E402

FIG = "../report/figures/"
SLIDES = [
    ("text2mario", """<p class=big>Type a sentence, play a verified Mario level</p>
      <p>A text-conditioned diffusion model, a playability checker, a browser game and a Streamlit app.</p>
      <p class=small>AI2002 Software Category, Path A. All numbers on these slides are generated from saved result files.</p>"""),
    ("1. The problem and who it is for", """<ul><li>A platformer level is a tile grid; a first draft by hand is slow.</li>
      <li>Random generators are hard to steer; learned ones ignore words and rarely check that the level can be finished.</li>
      <li><b>Users:</b> indie/hobby level designers (steerable drafts, export), students of procedural generation (inspectable pipeline), demo users (play it).</li>
      <li><b>Input:</b> one sentence + width + guidance + seed. <b>Output:</b> framed 16-row level, canonical caption, badge, PNG, playable HTML, JSON/CSV/VGLC.</li>
      <li class=small>No user study was run: the needs are stated, not measured.</li></ul>"""),
    ("2. Objectives and scope", """<ul><li>O-1 content follows the prompt (C-score) &nbsp; O-2 valid structures (no broken pipes/cannons)</li>
      <li>O-3 verify beatability <i>and measure how far to trust it</i> &nbsp; O-4 do not copy training windows</li>
      <li>O-5 honest evaluation on unseen levels (level-grouped split, one-shot test) &nbsp; O-6 deployed Streamlit app</li>
      <li>Scope: one game family (SMB1/SMB2 tiles), 16 rows, widths 16-112, five quantity buckets, no positions.</li></ul>"""),
    ("3. Data and leakage prevention", """<ul><li>MarioDiffusion corpus: 7,687 windows (16x16, 13 tile classes) from 42 levels; captions in a fixed grammar.</li>
      <li>Split <b>by level</b>: 31 train / 5 dev / 6 test (windows overlap by 15 of 16 columns); near-duplicate families kept together; hash-locked.</li>
      <li>Our caption detector reproduces <b>7,687 / 7,687</b> stored captions (written from the grammar, not ported).</li>
      <li>Tuning on dev only; test used once (guards in code and tests).</li></ul><img src="{FIG}fig11_tiles.png" style="height:70px">"""),
    ("4. The technique: diffusion + text + guidance", """<ul><li>x_t = &radic;&#7869;<sub>t</sub> x<sub>0</sub> + &radic;(1-&#7869;<sub>t</sub>) &epsilon;; a U-Net predicts &epsilon; (MSE); DDIM-50 sampling with x<sub>0</sub> clipping.</li>
      <li>Frozen MiniLM tokens &rarr; trainable adapter (66,048 params) &rarr; cross-attention in the U-Net (18.3 M params).</li>
      <li>10 % caption dropout &rarr; classifier-free guidance &epsilon;<sub>u</sub> + w(&epsilon;<sub>c</sub> - &epsilon;<sub>u</sub>).</li>
      <li>Long levels: stride-8 windows with RePaint known-region replacement; framed start and flag.</li></ul>"""),
    ("5. System pipeline", """<img src="{FIG}fig01_pipeline.png" style="width:100%"><ul class=small><li>Normalizer: 30/30 on the dev prompts, 100 % passthrough of all stored captions.</li>
      <li>Failed check &rarr; re-sample the stuck window (max 5 per window, 20 per level). Same prompt + seed &rarr; same level.</li></ul>"""),
    ("6. Playability checker and game", """<ul><li>A* with jump arcs simulated by the game's own integrator; physics in one shared file; conservative margins; enemies killable.</li>
      <li>Real train/dev levels solved: <b>{{cal_levels}}</b> (full), <b>{{cal_windows}}</b> (windows) &mdash; target 0.90 not reached.</li>
      <li>Reference-solver agreement <b>{{p5b_agree}}</b> (&ge; 0.85 needed): <b>gate failed</b>, so selection uses S<sub>noplay</sub>.</li>
      <li>Evidence instead: game-engine contract test ({{game_unsound}} unsound of {{game_cases}}), ghost-bot replay {{replay_rate}} of {{replay_n}} real windows.</li></ul>"""),
    ("7. The Streamlit app", """<ul><li>Prompt box + examples &rarr; <i>what the model will understand</i> (scenes, clamps, ignored words) &rarr; generate &rarr; badge + embedded offline game + downloads.</li>
      <li>Input validation with readable errors; CPU fast mode (25 steps) automatic without a GPU; cloud profile with a time budget.</li>
      <li>Measured (median): {{lat_cuda_48_fast_p50}} s for 48 columns and {{lat_cuda_112_fast_p50}} s for 112 columns in fast mode on the RTX 3060 Ti; full quality takes {{lat_cuda_48_p50}} s / {{lat_cuda_112_p50}} s; CPU fast mode {{lat_cpu_48_p50}} s / {{lat_cpu_112_p50}} s. The 10 s target is not met at 112 columns.</li></ul>"""),
    ("8. Quantitative results (unseen test levels, one shot)", """<table><tr><th></th><th>P-test</th><th>P-rand</th></tr>
      <tr><td>C-score</td><td>{{t_c_score}}</td><td>{{r_c_score}}</td></tr><tr><td>exact match</td><td>{{t_exact_match}}</td><td>{{r_exact_match}}</td></tr>
      <tr><td>broken rate</td><td>{{t_broken_rate}}</td><td>{{r_broken_rate}}</td></tr><tr><td>floor validity</td><td>{{t_floor_valid}}</td><td>{{r_floor_valid}}</td></tr>
      <tr><td>copy rate</td><td>{{t_copy_rate}}</td><td>{{r_copy_rate}}</td></tr></table>
      <p class=small>Prompt-ignoring reference (null): {{t_ref_null_other_level}}. Intervals resample the {{t_n_levels}} test levels (wide). Dev: a retrieval baseline (look up the best training window) scores C = 0.949 vs the model's 0.794 &mdash; the model's value is novelty (copy rate 0.08 vs 1.0), not adherence.</p>"""),
    ("9. Qualitative results", """<div style="display:flex;gap:12px"><img src="{FIG}fig08_cond_samples.png" style="width:48%"><img src="{FIG}fig12_render_best_000.png" style="width:48%"></div>
      <p class=small>Left: dev samples at w*; right: a generated full level. Contrast test (pipes vs enemies/gaps/stairs) and failure gallery are in the report.</p>"""),
    ("10. What did not work (and what we did about it)", """<ul><li><b>G-P3 failed</b>: broken structures fall with training while copying rises; aux-CE trades one for the other. We chose cosine, documented a waiver, and judged the conditional model on its own metrics.</li>
      <li><b>G-P5b failed</b>: the reference solver is not a usable oracle here; fallback S<sub>noplay</sub>.</li>
      <li>Four bugs found by tests/e2e runs and fixed with regression tests.</li></ul><img src="{FIG}fig03_p3_tradeoff.png" style="height:210px">"""),
    ("11. Limitations and conclusion", """<ul><li>31 training levels; coarse captions (no positions, 5 buckets); one game family; 5 dev / 6 test levels &rarr; wide intervals; no human playability study.</li>
      <li>"Verified" = a conservative physics claim by our checker, not a proof for every player.</li>
      <li>Cloud deployment needs the team's Streamlit/GitHub login; the repo is prepared (DEPLOY.md).</li>
      <li><b>Conclusion:</b> a complete, honestly evaluated text-to-level system; failures are reported with evidence and fallbacks.</li></ul>"""),
]
CSS = """@page{size:1280px 720px;margin:0}body{margin:0;font-family:Arial,sans-serif;color:#1b2a49}
.slide{width:1280px;height:720px;box-sizing:border-box;padding:42px 60px;page-break-after:always;position:relative;background:#fff;overflow:hidden}
h1{font-size:34px;margin:0 0 18px;border-bottom:3px solid #4477aa;padding-bottom:8px}li{font-size:23px;margin:9px 0;line-height:1.3}
.big{font-size:44px;font-weight:bold;margin-top:120px}.small,li.small{font-size:17px;color:#444}table{border-collapse:collapse;font-size:21px}
td,th{border:1px solid #99a;padding:6px 14px}th{background:#eef}.num{position:absolute;bottom:14px;right:24px;font-size:14px;color:#888}img{display:block;margin-top:12px}"""


def main():
    vals = BR.values()
    best = sorted((ROOT / "report/figures").glob("fig12_render_best_*.png"))
    best_name = best[0].name if best else "fig08_cond_samples.png"
    parts = []
    for i, (title, body) in enumerate(SLIDES, 1):
        body = body.replace("{FIG}", FIG).replace("fig12_render_best_000.png", best_name)
        parts.append(f"<div class=slide><h1>{title}</h1>{BR.render(body, vals)}<div class=num>{i}/{len(SLIDES)}</div></div>")
    out = ROOT / "slides"
    out.mkdir(exist_ok=True)
    (out / "slides.html").write_text(f"<!doctype html><html><head><meta charset=utf-8><title>text2mario slides</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.goto((out / "slides.html").as_uri())
        pg.wait_for_timeout(600)
        pg.pdf(path=str(out / "slides.pdf"), width="1280px", height="720px", print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    print("wrote slides/slides.pdf", (out / "slides.pdf").stat().st_size // 1024, "KB,", len(SLIDES), "slides")


if __name__ == "__main__":
    main()
