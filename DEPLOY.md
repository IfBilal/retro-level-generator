# Deploying to Streamlit Community Cloud (needs the team's GitHub and Streamlit login)

The build prepared everything that does not need a login. What is left is clicking through the Streamlit Community Cloud form.

1. **Export the model bundle** from the frozen checkpoint (about 120 MB, each file below GitHub's 100 MB limit):
   `python scripts/export_cloud_ckpt.py runs/C-main/step_XXXXXXX.pt` -> `app/model/{t2m_ema.safetensors, config.json, minilm/, EXPORT.json}`.
2. **Commit** `app/model/`, `app/static/vendor/phaser.min.js`, `assets/atlas*.png|json`, `data/prompts/{lexicon.yaml,clamp_table.json}`, `physics.json`, and `app/requirements.txt`
   (CPU torch). Do *not* commit `runs/`, `data/raw/`, checkpoints (`step_*.pt`).
3. On https://share.streamlit.io: *New app* -> repository `T361/retro-level-generator` (or the public deployment mirror), branch `main`, main file `app/streamlit_app.py`,
   *Advanced settings* -> Python 3.12, secrets none. Environment variable `T2M_PROFILE=cloud` (adds the 45 s time guard).
4. After the first build: open the URL, generate a 48-column level, then run `scripts/gate_p8.py <url>` is not implemented here; record the build log, cold-start time and one generation time
   in `evidence/deployment/` by hand (the rubric asks for the deployed URL).
5. Offline fallback for the demo: `streamlit run app/streamlit_app.py` on every laptop (the zip contains the exported bundle), and the Gallery page works without generation.

Measured before deployment (CPU, fast mode, a clean venv from `app/requirements.txt`): 16 columns in about 1 s and 48 columns in about 5 s on this machine; the cloud CPU will be slower, which is why the cloud profile has a time budget.
