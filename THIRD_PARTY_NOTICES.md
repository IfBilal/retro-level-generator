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

Assets (Phaser 3, Kenney sprites) will be listed here when they are fetched (doc 26).
