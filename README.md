# text2mario: type a sentence, play a verified level

**Live app (Streamlit Community Cloud): https://retro-level-generator-24i301524i3166-seb-aiproj.streamlit.app/**

Text-to-Mario-level diffusion project (Path A, Streamlit): a typed sentence becomes a full, framed, **checked** and **playable** level.
Specs and the day-by-day plan are in [docs/](docs/) (start at [docs/README.md](docs/README.md)); the grading rubric is
[docs/Rubrics_SoftwareCategory.pdf](docs/Rubrics_SoftwareCategory.pdf); the running log is [PROGRESS.md](PROGRESS.md).
Report: `report/report.pdf` · slides: `slides/slides.pdf` · evidence of evaluation: `evidence/` · model/data cards: [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md).

## What is here
| Part | Where |
|---|---|
| Data audit, level-grouped locked split, caption detector (7,687/7,687 captions), dataset | `src/text2mario/data`, `captions`, `scripts/audit_dataset.py`, `data/splits/` |
| Diffusion maths, U-Net (18.3 M), EMA, trainer, text conditioner, CFG sampler | `src/text2mario/diffusion`, `models`, `train`, `text` |
| Prompt normalizer (free text -> canonical captions) | `src/text2mario/text/normalizer.py` |
| Full-level generation (outpainting, framing, sanitizer, verify loop) | `src/text2mario/generate/` |
| Playability checker (A*, simulated jumps) + reference-solver bridge | `src/text2mario/playability/`, `scripts/derive_physics.py` |
| Browser game (Phaser, one offline HTML file) and level schema/exports | `app/static/game/`, `src/text2mario/render/` |
| Streamlit app (Studio, Gallery, About) | `app/streamlit_app.py`, `src/text2mario/app/service.py` |
| Evaluation (C-score, structure metrics, seams, level-clustered CIs), gates, freeze, one-shot test | `src/text2mario/eval/`, `scripts/gate_p*.py`, `freeze.py`, `eval_test.py` |
| Tests (fast suite + headless-browser end-to-end) | `tests/` |

## Run the app
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu    # or the CUDA wheel of the same minor
uv pip install -r requirements-dev.txt streamlit
streamlit run app/streamlit_app.py --server.address 127.0.0.1   # (or scripts/run_app.sh) uses the exported bundle in app/model/ (or set T2M_CKPT=path/to/step_XXXX.pt)
```
Without a GPU the app switches to CPU fast mode automatically. The exported bundle in `app/model/` already contains the (fp16) MiniLM encoder, so the app needs no download and runs offline.

## Rebuild everything
```bash
python scripts/fetch_data.py && python scripts/build_index.py && python scripts/fetch_models.py && python scripts/fetch_assets.py && python scripts/make_atlas.py
python scripts/audit_dataset.py && python scripts/gate_p1.py && python scripts/gate_p2.py && python scripts/verify_env.py
pytest -q -m "not slow"                                   # fast suite;  pytest -q -m slow  = headless-browser end-to-end
PYTHONPATH=src python -m text2mario.train.train --config configs/cond_main.yaml --run-dir runs/C-main
python scripts/gate_p4.py --ckpt runs/C-main/step_XXXXXXX.pt && python scripts/gate_p5.py --ckpt runs/C-main/step_XXXXXXX.pt
python scripts/freeze.py --ckpt runs/C-main/step_XXXXXXX.pt --w 5 && python scripts/eval_test.py          # one shot
python scripts/export_cloud_ckpt.py runs/C-main/step_XXXXXXX.pt && python scripts/make_figures.py && python scripts/build_report.py
```
Deployment to Streamlit Community Cloud: see [DEPLOY.md](DEPLOY.md) (needs the team's login).

## Honest status
Several gates failed and are documented with their evidence and fallbacks, not hidden: G-P3 (unconditional baseline; waiver),
G-P5b (checker vs reference solver; `S_noplay` used for selection), and the checker's real-level solve-rate target. See the report (sections 5-8) and `reports/gates/*/RESULT.json`.

## Credits
Builds on the MarioDiffusion dataset and idea (Schrum et al., arXiv 2507.00184, MIT). Phaser 3 (MIT), Kenney Pixel Platformer (CC0), all-MiniLM-L6-v2 (Apache-2.0).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/) for what is external and what is our own work.
