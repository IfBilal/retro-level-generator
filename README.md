# Retro Level Generator

Text-to-Mario-level diffusion project (Path A, Streamlit): type a sentence, get a verified, playable level.
Full plan, specs and status are in [docs/](docs/) (start at [docs/README.md](docs/README.md) and [docs/STATUS.md](docs/STATUS.md)).
Grading rubric: [docs/Rubrics_SoftwareCategory.pdf](docs/Rubrics_SoftwareCategory.pdf). Day-by-day log: [PROGRESS.md](PROGRESS.md).

## Where we are
The pre-training phase is built and gated: data audit, level-grouped locked split, caption detector, schedulers, U-Net,
text conditioner, sampler and trainer (gates G-P1 and G-P2 pass). **No model has been trained yet.**

## Setup (Python 3.12)
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel of the same minor
uv pip install -r requirements-dev.txt
python scripts/fetch_data.py        # pinned MarioDiffusion + VGLC, hash-verified
python scripts/build_index.py       # data/processed/*
python scripts/fetch_models.py      # frozen MiniLM (once; everything else runs offline)
python scripts/audit_dataset.py && python scripts/gate_p1.py && python scripts/gate_p2.py
python scripts/verify_env.py
pytest -q -m "not slow"
```

## Layout
`src/text2mario/` (data, captions, diffusion, models, text, train, eval, render, utils) · `scripts/` (every figure/gate comes from a
script) · `configs/` (all numbers live here) · `data/splits` + `data/prompts` (committed, locked) · `reports/` (gate results, audit,
references) · `tests/` · `reference/ref_unet.py` (oracle for the U-Net).

## Credits
Builds on the MarioDiffusion dataset and idea (Schrum et al., arXiv 2507.00184, MIT); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [DATA_CARD.md](DATA_CARD.md) for what is external and what is our own work.
