# Deploying to Streamlit Community Cloud (needs the team's GitHub and Streamlit login)

The build prepared everything that does not need a login. What is left is clicking through the Streamlit Community Cloud form.

1. **Export the model bundle** from the frozen checkpoint (about 120 MB, each file below GitHub's 100 MB limit):
   `python scripts/export_cloud_ckpt.py runs/C-main/step_XXXXXXX.pt` -> `app/model/{t2m_ema.safetensors, config.json, minilm/, EXPORT.json}`.
2. **Commit** `app/model/`, `app/static/vendor/phaser.min.js`, `assets/atlas*.png|json`, `data/prompts/{lexicon.yaml,clamp_table.json}`, `physics.json`, and `app/requirements.txt`
   (CPU torch). Do *not* commit `runs/`, `data/raw/`, checkpoints (`step_*.pt`).
3. On https://share.streamlit.io: *New app* -> repository `T361/retro-level-generator` (or the public deployment mirror), branch `main`, main file `app/streamlit_app.py`,
   *Advanced settings* -> Python 3.12, secrets none. Environment variable `T2M_PROFILE=cloud` (adds the 45 s time guard).
4. After the first build, run the deployment gate from any machine with Playwright: `python scripts/gate_p8.py https://YOUR-APP.streamlit.app` -> `reports/gates/P8/RESULT.json` and `evidence/deployment/screenshot.png` (load time, generation time, game iframe present, empty prompt shows the validation message, no page errors). A local dry run of this script passed all five checks; the deployed result is yours to record. Also note the Streamlit build log and the first cold-start time in `evidence/deployment/`.
   Deployed app: https://retro-level-generator-24i301524i3166-seb-aiproj.streamlit.app/ (repository `T361/retro-level-generator`, branch `main`, main file `app/streamlit_app.py`). `gate_p8.py` against it: see `reports/gates/P8/RESULT.json` (the audit also drives it with `scripts/audit_browser.py`).
5. Offline fallback for the demo: `streamlit run app/streamlit_app.py` on every laptop (the zip contains the exported bundle), and the Gallery page works without generation.

Measured before deployment (CPU, fast mode, a clean venv from `app/requirements.txt`): 16 columns in about 1 s and 48 columns in about 5 s on this machine; the cloud CPU will be slower, which is why the cloud profile has a time budget.
