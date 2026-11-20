# Data card

**Source.** MarioDiffusion `Mar1and2_LevelsAndCaptions-regular.json` at commit `561c76d4634027fa89ff8c1cfb58a24e8c089197`
(content sha256 `6a15264a...de2c`, recorded in `data/MANIFEST.json`), plus the VGLC level files used only to name the levels.
Licences and credits: `THIRD_PARTY_NOTICES.md`.

**What it is.** 7,687 windows of 16 x 16 tiles (13 tile classes) cut at stride 1 from 42 levels (20 SMB1, 22 SMB2 "Lost
Levels"); each window has a rule-based caption of 1-10 phrases from a closed grammar of 62 phrases / 44 words. The corpus has
no level id; we reconstruct it from the stride-1 overlap (`text2mario.data.index`), and the 42 reconstructed levels match
the VGLC files byte for byte. All ledger facts were re-measured (`reports/eda/AUDIT.json`, 39 / 39 checks pass).

**Known properties that shape the project.**
- Sky is 87.1 % of all tiles; some concepts are rare (descending staircases occur in 5 levels, ceilings in 4).
- Windows overlap by 15 of 16 columns, so a random window split leaks: the upstream test split covers all 42 levels, with a
  median distance of 8 tiles to the nearest training window (F17, re-measured).
- 7.49 % of windows have a near-copy (<= 2 tiles) in a different level; levels form two near-duplicate families (15 levels).
- Captions carry no position information; all enemies are one symbol; levels are only 14-16 rows tall (padded to 16 with sky).

**Our split (`data/splits/split_v1.json`, locked by `data/splits/LOCK`).** By level, never by window: 31 train / 5 dev / 6 test
levels (5,709 / 905 / 1,073 windows). The two near-duplicate families stay in train. The test levels are touched only by
`scripts/eval_test.py`, once, after `FROZEN.json` exists (`MarioWindows` raises `PermissionError` otherwise). Remaining
dev/test windows within 2 tiles of a train window (1.7 % / 1.9 %) are listed in `data/splits/reused_segments.json`.

**Intended use / limits.** Training a text-conditioned generator of SMB-style tile levels for coursework. 42 levels is a tiny
number of independent units, so every held-out statistic has wide uncertainty; results are reported per level and with
level-clustered intervals. Not suitable for claims about other games or about commercial level design.
