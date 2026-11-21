# Third-party notices

This project builds on published work; what is external and what is the group's own work is stated in
`docs/docs/02_SCOPE_AND_REQUIREMENTS.md` §15 and ledger §11b. None of the third-party data files below are committed to
this repository (only hashes and the manifest are).

| Item | Licence | Notice |
|---|---|---|
| **MarioDiffusion** (Schrum et al., arXiv 2507.00184): dataset of 16x16 Mario windows with rule-based captions, and the idea of caption-conditioned tile diffusion. Pinned commit `561c76d4634027fa89ff8c1cfb58a24e8c089197` | MIT | Copyright (c) 2025 Jacob Schrum |
| **TheVGLC** (cleaned fork used to build the dataset), commit `e91316a0b53955e981e179c5ec4f8d989d02b2bb`; cite Summerville et al., "The VGLC: The Video Game Level Corpus", arXiv 1606.07487 | MIT | Copyright (c) 2016 Adam Summerville, Santiago Ontanon, Sam Snodgrass |
| Underlying level designs (Super Mario Bros., Super Mario Bros. 2 Japan) | (c) Nintendo | Used for non-commercial academic research only; the raw corpus is not redistributed and Nintendo-derived sprites are never committed |
| **all-MiniLM-L6-v2** (frozen text encoder), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 (per its model card) | Downloaded by `scripts/fetch_models.py`, not committed (re-check the licence text when it is bundled for the cloud profile, ledger CL4) |

The MarioDiffusion source code is **not** imported or transcribed: the caption detector (`src/text2mario/captions/detector.py`)
is implemented from the grammar description and checked against the stored captions (ledger O3).

## Runtime assets (fetched by `scripts/fetch_assets.py`, recorded with hashes in `assets/ASSET_MANIFEST.json`)
| Item | Licence | Notice |
|---|---|---|
| **Phaser 3.90.0** (`app/static/vendor/phaser.min.js`, 1,196,122 bytes, sha256 `e92ddef1...eba7a7`) | MIT | Copyright (c) 2024 Richard Davey, Phaser Studio Inc.; licence text in `LICENSES/PHASER_MIT.md`. |
| **Kenney Pixel Platformer** (tilemap and character sprites in `assets/kenney/`; `assets/atlas.png` is built from them) | CC0 1.0 | Kenney.nl; the licence file is `LICENSES/KENNEY_CC0.txt`. Pipes and cannons in the atlas are drawn by us with Pillow. No Nintendo sprites are used. |
| **MarioEval.jar** (Baumgarten A*, used only as an external reference solver for the G-P5b agreement study; never imported, never shipped in the app) | as distributed with MarioDiffusion (MIT repository) | Run via `subprocess`; not part of the submission zip. |
| Python packages (torch, numpy, transformers, safetensors, Pillow, PyYAML, streamlit, playwright for tests) | BSD / Apache-2.0 / MIT | pinned in `requirements*.txt`, `app/requirements.txt`. |
