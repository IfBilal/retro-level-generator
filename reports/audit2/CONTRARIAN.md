# Contrarian audit (audit2, 27 Nov)

Method: read normalizer.py / adherence.py / service.py / streamlit_app.py / game_html.py cold, then ran about 250 prompts through `normalize()`
(harness: scratchpad `n2.py`, calling `normalize(p)` and printing `caption, want, forbid, warnings`) and a few through `LevelService.run(..., fast=True, verify=True, width=48)` on CPU.
No source/test/config/docs file was edited. Reproduction for every row below:

    .venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');from text2mario.text.normalizer import normalize as n;x=n('PROMPT');print([(s.caption,s.want,s.forbid) for s in x.scenes],x.warnings)"

Output shown is exactly what was printed (forbid lists shortened where marked). "Silent" = nothing in `Normalized.warnings` and no scene warning the UI shows.

Key insight for ranking: the 630/630 adherence figure measures the planner against the NORMALIZER's reading of the sentence. Any prompt the normalizer mis-reads
is then "verified" as met, and the UI says "Counts match your sentence". So parser defects are invisible to every gate that exists.

## Root causes, ranked

### RC1 (HIGH) Negation vocabulary is a closed list of 11 tokens; every other prohibition is read as a REQUEST for the thing
`normalizer.py:42` NEG, used only in `_qty_before` (l.232) as a single token directly left of the noun (skipping only ADJ/STOP words).
`any`, `a single`, `avoid`, `skip`, `remove`, `exclude`, `free of`, `-free`, `minus`, `except`, `get rid of`, `don't want` are not in NEG / ADJ / STOP, so the walk stops and the noun becomes a bare plural -> `BARE_PLURAL_DEFAULT` "two".
The user asked for NONE and the level is generated, verified "met", with TWO. A flipped meaning, no warning in `norm.warnings`.

| prompt | actual output |
|---|---|
| `without any pipes or enemies` | `[('two enemies. two pipes.', {'pipe':[2,2],'enemy':[2,2]}, [])] []` |
| `I don't want any enemies` | `[('two enemies.', {'enemy':[2,2]}, [])] []` |
| `not a single enemy` | `[('one enemy.', {'enemy':[1,1]}, [])] []` |
| `avoid pipes` | `[('two pipes.', ...)]` |
| `skip the enemies` / `remove all enemies` | `[('two enemies.', ...)]` |
| `get rid of the pipes` | `[('two pipes.', ...)]` |
| `a level free of enemies` | `[('two enemies.', ...)]` |
| `pipe-free level with 3 coins` | `[('a few coins.', {'coin':[3,3]}, [])]` (pipes allowed) |
| `enemy-free` | `[] ['EMPTY_SCENE']` -> refused, but then the user is told "nothing buildable" |
| `not too many pipes` | `a few pipes. {'pipe':[10,99]}` (flip to MANY) |
| `not 3 pipes but 2` | `{'pipe':[3,3]}` (the negated number is the one built) |
| `3 coins, minus the pipes` | `two pipes` wanted |

Fix: normalise negation phrases to `no` in `_pre()` before tokenising ("without any", "don't want any", "not a single", "avoid", "skip", "remove( all)", "get rid of", "free of", "X-free", "minus", "except", "exclude", "not any"), and make `any/all/the/a single` transparent in `_qty_before`.
Add `NEGATION_UNSURE` warning whenever a negation-looking word (`not|n't|avoid|skip|without|free|except|minus|remove`) occurs and no `forbid` was produced.

### RC2 (HIGH) A colon splits off a "global prefix" and everything in it except floor/ceiling is thrown away
`normalizer.py:438-441` (`_split_scenes`) + `normalize()` l.503-506 (`g_topics` keeps only floor/ceiling).
| prompt | actual output |
|---|---|
| `no pipes: 2 coins` | `[('two coins.', {'coin':[2,2]}, [])] []` : the prohibition is gone, the level may contain pipes and is reported "met" |
| `pipes: 2` | `[] ['EMPTY_SCENE']` (then refused) |
| `enemies: none` | `[] ['EMPTY_SCENE']` |
| `pipes: 2 and no enemies` | `full floor, forbid enemy` : the 2 pipes vanished |
Fix: only treat the prefix as global when it contains no counted noun / negation / numeral; otherwise treat the colon as a comma.

### RC3 (HIGH) Hyphen / compound / decimal / range numbers are mis-read as other numbers or none
`_tokens` (l.192-196) keeps `-` inside tokens and strips `.`; `_qty_before` only knows single tokens in NUMBERS and `\d+`.
| prompt | actual |
|---|---|
| `twenty-one pipes` | `two pipes` (token `twenty-one` unknown -> bare plural default) |
| `twenty one pipes` | `one pipe` |
| `2.5 pipes` | `a few pipes {'pipe':[5,5]}` ("." removed -> "2 5") |
| `3-4 pipes` | `two pipes` (bare default) |
| `3 to 5 pipes`, `3-5` with en dash, `4 or 5 pipes` | `{'pipe':[5,5]}` (upper bound) |
| `between 2 and 4 pipes` | `{'pipe':[4,4]}` plus POSITION_NOT_PLANNED ("between" is a ref prep) |
| `one or two pipes` / `a pipe or two` | `two pipes` / `one pipe` |
| `three separate pipes` | `two pipes` (`separate` not in ADJ, so the 3 is lost: bare default) |
| `3 single pipes` | `one pipe` ("single" -> 1 overrides the 3) |
| `pipes (2) and coins (3)` | pipes 2, **coins 2** (the number after the noun is never used; the "2" is picked up for coins) |
| `pipes three`, `pipes, 3`, `pipes x 3`, `pipes ×3`, `pipe*3` | `two pipes`/`one pipe` (post-fixed numbers ignored) |
| `3pipes`, `3-pipes` | `[] EMPTY_SCENE` |
| `３ pipes` (full-width), `٣ pipes` | `two pipes` (non-ASCII digits stripped by `[^a-z0-9...]`) |
Fix: use `unicodedata.normalize('NFKC')` first; parse compound numbers (`twenty-one`, `twenty one`); ranges `a-b`, `a to b`, `a or b` -> `want=[a,b]`; `(n)` after a noun; `Nx`. Add a `NUMBER_UNPARSED` warning whenever a digit/number word is left unconsumed.

### RC4 (HIGH) Ordinal words ("first/second/third") inside any 3-token window before a noun delete the noun
`normalizer.py:324-333`: `near = toks[a-3:a]`; any ORDINAL in it turns the whole mention into a "reference", marks 3 tokens consumed and `continue`s, so the topic is never requested.
| prompt | actual |
|---|---|
| `2 pipes in the first half, 3 coins in the second half` | `[('two pipes.', {'pipe':[2,2]}, [])] ['POSITION_NOT_PLANNED']` : coins DROPPED |
| `the first 3 pipes` | `[] ['EMPTY_SCENE','POSITION_NOT_PLANNED']` (3 pipes lost -> app says nothing buildable) |
| `the second 2 coins` | same |
| `a first pipe and 5 coins` | only coins |
| `3 coins and a second pipe` | only coins |
| `before the first pipe, 2 enemies` | `[] EMPTY_SCENE` (the 2 enemies lost) |
| `the first pipe at the start and a coin at the end` | pipe dropped, only `one coin` |
Same code, REF_PREPS: `a gap before a pipe and 3 coins` -> `floor with one gap. a few coins.` (the pipe vanished; "before a pipe" has no q, so it is a reference); `past the pipes there are 4 coins` -> only coins.
Fix: an ordinal is a reference only when directly adjacent to the noun (`ord noun`, `ord of the noun`, `Nth noun`); a numeral between the ordinal and the noun means a count ("first 3 pipes" = 3 pipes). "half" must not be a noun hop. REF_PREPS followed by an article + noun with no quantity should still count the noun once if it is not already requested.

### RC5 (HIGH) FLAT_WORDS are matched anywhere in the raw prompt
`normalizer.py:55, 535, 542`. If no scene has `want`/`forbid`, every counted topic is forbidden, whatever the sentence meant.
| prompt | actual |
|---|---|
| `an empty gap` | `[('floor with one gap.', {}, [ALL 14 topics])] ['FLAT_REQUEST']` |
| `a plain gap` | same |
| `bare ground and a big gap` | `giant gap with one chunk of floor`, ALL forbidden |
| `a gap with nothing in it` | same |
| `a flat ceiling` | `full floor. full ceiling.` ALL forbidden (OK-ish) |
| `I want a plain old level` | `full floor.` ALL forbidden, FLAT_REQUEST (a "plain old level" is not a request for a verified-empty level) |
| `make it a little flat` | same |
| `there is nothing between the pipes` | `full floor.` all forbidden **including pipe**, FLAT_REQUEST + POSITION_NOT_PLANNED |
| `an empty gap` etc. then verified as "Flat level confirmed" (streamlit_app.py:87) |
Also asymmetric: `pipes and plain coins`, `bare minimum 2 pipes`, `the pipe is empty` pass because `want` exists, so "plain/bare/empty" are both a flat trigger and ignored adjectives depending on whether a count is present.
Fix: FLAT_WORDS only as a head noun/adjective of the whole level (`^(a |an |the )?(completely |totally )?(flat|empty|blank|bare|plain|barren)( level| ground| floor| stretch)?$` or "nothing else" at the end of a clause); drop `plain`, `bare`, `nothing`, `barren` from the set, and require "level|ground|floor|stage".

### RC6 (MEDIUM-HIGH) `stairs up` / `staircase up` is registered as PLURAL, so one staircase becomes two
`normalizer.py:138-139`: `out.append((w, "ascending staircase", True, "noun"))` for "stairs up, steps up, staircase up, stairs".
| prompt | actual |
|---|---|
| `a staircase up then a staircase down` | `[('two ascending staircases.', {'ascending staircase':[2,2]}, []), ('one descending staircase.', ...)]` |
| `a staircase up then down` | `two ascending staircases`, `EMPTY_SCENE` ("down" alone dropped) |
| `staircase down then up` | only `one descending staircase`, `EMPTY_SCENE` ("up" dropped, silent to the user as "Part of the sentence had no level words") |
Fix: `pl = w.endswith('s')` per word; handle bare `up`/`down` after a staircase scene.

### RC7 (MEDIUM-HIGH) Gap / floor-gap / ceiling-gap counts are never verified, but the UI says "Counts match your sentence"
`adherence.py:36` `COUNTED = CONCEPTS - (floor, ceiling)`; `constraints()` only emits COUNTED. `two pipes and nine gaps` -> constraints = pipe only; caption `floor with a few gaps` (clamped, bucket silently changed 9 -> "a few", the CLAMPED warning is scene-level only).
`a ceiling with five gaps` -> `ceiling with one gap.`; `no gaps` -> `full floor.` with no constraint. The success banner (`streamlit_app.py:91`) lists only pipe, so the user reads it as covering the whole sentence.
Fix: add floor-gap / ceiling-gap / chunk constraints to `evaluate`, or say in the banner "not checked: gaps".

### RC8 (MEDIUM) Silent merge/loss of order: commas do not split, "end with" before "start with"
`end with 3 coins, start with 2 pipes` -> ONE scene `('a few coins. two pipes.', ...)` and no warning (SPLIT_RE has `,\s*next` but no plain `,`; the START_RE/END_RE reordering only fires per segment). `pipes on the left and coins on the right` -> one scene, only the scene-level POSITIONAL_IGNORED.
`3 coins at the end then 2 pipes` gives coins first (typed order) with POSITION_NOT_PLANNED: the sentence order beats the explicit position word.
Fix: split on `, (start|end|finish|begin)` and apply the start/end ordering across segments; warn on position/order conflict.

### RC9 (MEDIUM) Prompt validation false positives
`service.py:73` regex `(?<![\w])-\s?\d` rejects `2 pipes - 3 coins`, `level 1 - 3 pipes`, `3 pipes -- 2 coins`, `pipes (-3)` with "Negative numbers do not make sense". `\r\n` (Windows paste / API client) is rejected as control characters (`service.py:71` allows only `\n`, `\t`). Fix: only flag `-\d` when preceded by start/space AND followed by a counted noun; allow `\r`.

### RC10 (MEDIUM) Large / vague numbers are refused or clamped with confusing text; caption disagrees with want
`twenty pipes`, `20 pipes`: `want [20,20]` but caption `a few pipes.` (clamp); `check_feasible` then refuses "20 x pipe will not fit in 48 columns (at most about 6)" : fine for the UI, but `one hundred pipes` parses as `hundred`=100, `thirty pipes` 30. `a dozen pipes` -> `{'pipe':[10,99]}` while VAGUE says 12 (bucket "many" = 10-99): the verifier accepts 10..99 pipes where the user said 12, and capacity limit is ~6 pipes, so "a dozen pipes" is refused even though "many pipes" (10-99) is requested by the same logic: `many many pipes` also yields `[10,99]` then refusal. `twice as many pipes as enemies` -> `pipe [10,99]`, `enemy [2,2]` (nonsense relation parsed as MANY).

### RC11 (MEDIUM, claims) The headline 630/630 figures do not support the README's framing
README.md:45 "Before the audit only 116/630 generated levels matched; now 630/630 ... 100/100 held-out prompts". Problems:
1. "matched" = matched the normalizer's reading, not what the person meant. RC1/RC2/RC4/RC5 prompts would all score 100% while doing the opposite of the request.
2. The prompt categories (count, example, negation, ordering, synonyms, adversarial) are written by the same authors who wrote the regexes; "held-out" prompts are held out from the model, not from the rule set.
3. reports/audit2/MODEL_RCA.md:15 itself says "CPU/cloud matching is weaker"; the deployed app is CPU-only (fast=True, time budget), the 630 was a local run. The deployed matching rate is not reported next to the headline.
4. Runs are wall-clock-budgeted on CPU (`service.py:146-147`), so "same prompt + seed + settings -> same level" (full_level.py docstring) is false whenever the budget triggers.

### RC12 (LOW-MEDIUM) Streamlit state
- `streamlit_app.py:181`: stale-result notice only compares the PROMPT text; changing seed, CFG w, width or quality still shows the old level with no notice.
- `streamlit_app.py:168`: a cleared Seed box gives `None`; `int(None)` -> TypeError -> generic "Generation failed (TypeError)" instead of a seed message.
- `drop_stale_modules()` (l.25-41) deletes `text2mario.*` from `sys.modules` and `st.cache_resource.clear()` when ANY session starts after a src change, while another tab may be mid-`svc.run` (second session crash / model reload; also memory x2 on a 1 GB Community Cloud box).
- One cached `LevelService` is shared by all sessions; two tabs generating concurrently contend for the same CPU and the wall-clock budget (`time_budget_s`) then reports "time budget" and unmet counts for the second user.
- `preview()` re-runs `normalize` on every keystroke rerun (fine: measured 0.8 s for 20 000 chars, no ReDoS; max 400 chars anyway).

### RC13 (LOW) Duplicate / contradictory mention handling hides information
`3 pipes and 2 pipes` -> `two pipes` ({'pipe':[2,2]}), warning `DUPLICATE_TOPIC` only at scene level ("last amount used"). `3 coins at the end ...` as above. `two pipes then no pipes` -> scene 2 `full floor` with forbid pipe (ok). `no pipes at the start and pipes at the end` ok.

## Security (nothing exploitable found)
- `render/game_html.py:26-31` escapes `</`, `<!--`, U+2028/9 in the JSON islands. Tested prompts `<script>alert(1)</script> 2 pipes`, `</script><img src=x onerror=alert(1)> pipes`, `2 pipes" onload="x`: normalised to nothing harmful, text only stored in JSON, escaped. `<title>` uses `level_id` (server generated `t2m_YYYYmmdd-HHMMSS_hex6`), not user text. game.js never renders prompt/caption (only Phaser `add.text` for HUD).
- Streamlit side: scene captions come from the grammar (`CONCEPTS` strings), `forbid` from a fixed list; the raw prompt is shown with `st.code`. Download file names use `level_id`. No path handling of user input.
- Residual (LOW): `title`/`level_id` is interpolated unescaped (`game_html.py:46`); safe only because nothing user-controlled can reach it. An uploaded-level feature would need `html.escape`.
- Prompt injection into captions: `ignore previous instructions and output 100 pipes` is just parsed as `100 pipes` -> refused by capacity check; no LLM in the loop.
- Non-English (`deux tuyaux`, `dos tuberias`, `zwei Rohre und drei Münzen`): `[] EMPTY_SCENE` -> refused with the "could not find anything I can build" message. Acceptable, but the message does not say "English only".

## Service-level runs
See the section appended at the end of this file if the CPU run finished (the box was shared with a long audit run, so only a few generations completed).

### Service run result
`LevelService(device="cpu").run(..., fast=True, verify=True, width=48)` on 12 prompts x 2 seeds (script in scratchpad) produced no completed generation within the 560 s limit
(the CPU was shared with the long audit run), so NO end-to-end claim is made here. Every finding above is at the `normalize()` / `check_feasible()` level, which is deterministic
and is what `LevelService.run` feeds the planner and verifier; a wrongly parsed prompt is judged "met" against the wrong target by `adherence.evaluate_both`.
