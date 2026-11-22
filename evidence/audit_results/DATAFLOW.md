# Data flow: sentence -> level, and every function that touches the COUNT of an object

Written from reading the code (before the adherence fix; the "NEW" rows are what the audit added). File:line numbers are in `src/text2mario/` unless noted.

## 1. Path of one request
```
app/streamlit_app.py studio()            sidebar prompt, seed, width, mode  -> validate_prompt()  -> svc.preview() (normalize) -> svc.run()
  app/service.py LevelService.run()      validate_prompt + validate_settings -> generate() -> exports (png/json/csv/vglc/html)
    text/normalizer.py normalize()       sentence -> Normalized(scenes[Scene(caption, topics, forbid, warnings, want NEW)])
      _split_scenes()  split on "then", ";" ... -> segments           (ordering = order of segments)
      _canonical_scene()  a segment already in the training grammar is passed through
      _scene()  tokens -> nouns (_match_nouns) -> quantity (_qty_before) -> counts{topic:int} -> topics{topic: bucket word} + want NEW
      _caption()  bucket words -> clamp table (_clamp) -> "two pipes. one enemy."        <- the model's conditioning string (COARSE: one/two/a few/several/many)
    generate/full_level.py generate()    scenes -> n windows (16 wide, stride 8) -> sample each window -> assemble -> frame -> checker -> resample
      scene_for_windows()  spreads scenes over windows in order                          (OLD: every window of a scene got the SAME scene caption)
      adherence.check_feasible() NEW       contradiction / impossible count -> ValueError
      adherence.plan_caption() NEW         per-window caption = only what the window must ADD; returns plan{topic:n}
      sample_window()                      one 16x16 window, CFG, optional known columns (RePaint)
      adherence.window_miss() NEW          redraw the window until its NEW columns hold exactly the plan (best of N)
      _assemble(), frame()                 stitch windows; add 3 spawn + 6 end-template columns (NOT generated)
      adherence.evaluate() NEW             count the RAW content per scene region -> verification.adherence
      adherence.window_to_resample() NEW   which window to redraw for an unmet count
      sanitizer.repair()                   DISPLAY copy only (broken pipes/cannons removed); raw_grid is never edited; adherence is judged on raw
    render/schema.py make_level/validate_level   level dict (raw_grid, display_grid, prompt.scenes[caption per window], verification)
    render/game_html.py build_game_html  inlines the level JSON (escaped) into the offline Phaser page
```

## 2. Where a count can change (every touch)
| # | function | what it does to a count | audit verdict |
|---|---|---|---|
| 1 | `normalizer._qty_before` / `NUMBERS` / `VAGUE` | words -> integer ("several" -> 5, "a few" -> 3, "many" -> 12) | numerals fine; "-3" mis-tokenised (negative sign dropped) -> fixed by `validate_prompt` |
| 2 | `normalizer._scene` | `counts[topic]`; `neg`/`q==0` -> `forbid` | bare negation produced NO scene (forbid lost); contradiction silently emptied -> fixed |
| 3 | `normalizer.bucket` / `_clamp` | integer -> bucket word -> training-supported bucket (clamp table) | LOSSY: "three" -> "a few" (3-4), "three towers" -> "two"; the exact number was thrown away -> `Scene.want` keeps it |
| 4 | `grammar.counted_phrase` / `quantity_word` | bucket -> words in the caption | only 5 buckets exist |
| 5 | `full_level.scene_for_windows` + `sample_window` | the scene caption goes to EVERY window of the scene | **ROOT CAUSE of P2**: a 16-column caption "one pipe" over 5 windows = one pipe per window |
| 6 | the model | draws objects for a window caption | loose: "one pipe" exact in ~55% of windows, "two pipes" ~10-30%, "a few enemies" never exactly 3-4 (draws 4-10) (`reports/audit/window_exactness.json`) |
| 7 | RePaint known columns | the right half of window k is the left half of window k+1 | a truthful caption of the known half DOUBLES the count (measured: 19/24 windows add another pipe) -> caption names only what is NEW |
| 8 | `full_level._forbidden_hit` (old) | caption-text check for forbidden words | only checked windows, not counts; replaced by `adherence.evaluate` |
| 9 | checker loop (old) | resamples for PLAYABILITY only | accepted any count: hypothesis (e) true |
| 10 | `sanitizer.repair` | removes broken pipes/cannons from the DISPLAY grid | does not add objects (hypothesis f false); metrics use raw |
| 11 | `renderer` / game | draws display_grid | does not change counts (hypothesis g partly: the panel showed the scene caption, not the per-window captions the model got; the Details tab now lists the real per-window captions) |
| 12 | `captions.detector.detect` | defines what a count IS (pipes = connected components, enemies/coins = cells, platforms = runs, staircases = diagonal runs) | the audit's counter uses the same definition plus an independent re-count (`eval/count_audit.independent_counts`) |

## 3. What "one pipe" means in the audit
A pipe = one connected component of pipe tiles (cap + body), counted by the detector; an upside-down pipe and a *broken* pipe also count as "a pipe" (a person sees a pipe). Cross-check: number of vertical runs of left-half pipe tiles. Hand-built-grid tests: `tests/test_count_audit.py` (two touching pipes, pipe next to a tower, upside-down pipe, scene ownership, ordering).
Quantity words follow ledger F15 (`grammar.quantity_word`): one = 1, two = 2, a few = 3-4, several = 5-9, many >= 10; numerals are exact. "then" = left-to-right order of the scenes; an object belongs to the scene whose windows first own its leftmost column (window 0 owns columns 0-15, window k>0 owns the 8 columns it adds).
