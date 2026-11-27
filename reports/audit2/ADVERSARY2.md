# ADVERSARY2 (agent F): new prompts against the fixed normalizer (commit 24250d2)

Method: about 570 new prompts (5 batches plus an edge/perf batch, none copied from CONTRARIAN.md) run through `normalize()` and `validate_prompt()`;
each output (`want`, `forbid`, scene order, warnings) compared with what a person means. Nothing edited. No GPU, no LevelService, no pytest.
Timing: every call under 1 s (worst 0.62 s on a 5000-digit string; 18 kB of "2 pipes, " took 0.17 s). No hang.
Only DISAGREEMENTS are listed. "silent" = nothing in `Normalized.warnings` and no scene warning that says it was misread.
Counts: about 570 prompts, about 125 disagreements (~22 %); roughly a third flip a prohibition into a request.
Line refs are src/text2mario/text/normalizer.py unless stated.

## RC1 Negation vocabulary is still a closed list (flipped meaning: "no X" becomes "2 X")  [HIGH]
Cause: `_NEG_PHRASES` (l.188-191) + `NEG` (l.42, used only in `_qty_before` l.249-275, single token directly left of the noun).
Each of these yields the noun as a bare plural (BARE_PLURAL_DEFAULT = two) or the stated number as a request, forbid = []:

| prompt | actual |
|---|---|
| `w/o pipes`, `wo pipes`, `sans pipes`, `sin pipes` | pipe 2 |
| `without a single pipe` | pipe 1 ("a single" is read as a quantity) |
| `without even one enemy`, `not one pipe`, `not even a coin` | enemy 1 / pipe 1 / coin 1 |
| `no 2 pipes`, `not 2 pipes`, `not 3 pipes but 2 coins` | pipe 2 / pipe 2 / pipe 3 + coin 2 |
| `3 coins, but not pipes`, `3 coins, not pipes`, `3 coins and not 2 pipes` | coin 3 + pipe 2 |
| `neither pipes nor coins`, `nor pipes`, `neither pipes nor enemies, just 3 coins` | pipes/coins/enemies all requested |
| `I'd rather not have enemies` | enemy 2 |
| `don't use cannons`, `never use pipes`, `stay away from pipes`, `steer clear of enemies`, `nothing with pipes` | requested (2) |
| `ban the enemies`, `leave the enemies out`, `keep the pipes out`, `excluded enemies`, `exclusive of pipes` | requested (2) |
| `everything but pipes`, `anything but enemies`, `all but enemies`, `besides / other than / apart from / not including pipes` | pipe 2 / enemy 2 (the opposite) |
| `goomba free level` (space, not hyphen; `_HYPHEN_FREE` l.194 needs "-") | enemy 1 |
| `pipes - none`, `pipes (none)` (`_NOUN_PAREN_NUM` l.196 takes digits only) | pipe 2 |
| `less pipes`, `fewer enemies`, `hardly any enemies`, `barely any pipes`, `less the pipes` | 2 requested |
| `not many coins`, `not a lot of enemies`, `not too many goombas` | want (10,99) = MANY: the opposite of the sentence |
| `no-pipes`, `nopipes`, `NoPipes` | EMPTY_SCENE (loud, but `pipe-free` works and `no-pipes` does not) |

`no pipez`, `no enemys`, `dont add enemys` give EMPTY_SCENE only (see RC3).

## RC2 Comma inside a number / word number after a colon / fractions  [HIGH for the first]
- `2,000 coins` -> want {} forbid ['coin'] (NEGATION_ONLY): "2,000" is split at the comma and the lone "000" = 0 forbids coins. Same family: `2,5 pipes` -> 5 pipes, `1,5 pipes` -> 5 (validate_prompt l.73 only checks `\d\.\d`, not `\d,\d`). Cause: the scene/clause split at "," then `_qty_before` reads the last digit group.
- `pipes: three, coins: two` -> pipe 2, coin 3 (SWAPPED); `pipes = 3, coins = 2` -> pipe 2, coin 3 (swapped); `pipes=3` -> 2; `pipes x3`, `x3 pipes` -> 2; `pipes 2 coins 3` -> pipe 2, coin 2. Cause: `_NOUN_COLON_NUM` l.195 accepts only `\d+|none|zero|no`; word numbers, "=" and the "x3" prefix/suffix forms are not handled, so the bare noun takes the number of the next noun phrase.
- `hard: 3 goombas, 2 pipes` -> enemy 2 (3 lost), `easy: 3 goombas` likewise. Cause: `_NOUN_COLON_NUM` matches "hard: 3" and rewrites it to "3 hard goombas"; "hard"/"easy" are not in ADJ (l.45) so the quantity walk stops. (`hard 3 goombas` without the colon is right.)
- `1e3 pipes` -> 2, `1/2 pipes` -> 2, `3 1/2 pipes` -> 2, `2 sets of 3 pipes` -> 3 (expected 6), `2 groups of 3 coins` -> 3, `2 pairs of pipes` -> 2 (expected 4), `a couple dozen coins` -> 3-4 (expected 24 = many), `twice as many pipes` -> (10,99). Silent.

## RC3 Typos / hyphen compounds / "a second X" silently drop an object  [HIGH where silent]
`two pipez and a coin` -> only {coin 1}, NO warning at all (the pipes vanish; the UI will say the counts match). `2 pipez` alone is EMPTY_SCENE (loud). `3-pipe level`, `a 3-coin line` -> EMPTY_SCENE. `a plain floor`, `ground only` -> EMPTY_SCENE. `a second pipe`, `the second pipe`, `a second coin and a third coin` -> EMPTY_SCENE (a "second pipe" usually means another pipe; the ordinal-reference rule around l.355 treats any ordinal + noun as a reference). Cause: no fuzzy matching in `_match_nouns` l.229; no "unknown word" warning.

## RC4 Synonyms of one topic overwrite instead of adding  [HIGH, silent at the page level]
`a goomba and a koopa` -> enemy 1 (expected 2); `2 goombas and 1 koopa then 3 coins` -> enemy 1; `three goombas and two koopas` -> 2 (expected 5); `2 pipes and 3 tubes` -> 3; `a turret and a cannon` -> 1; `3 coins plus 3 collectibles` -> 3; `2 coins and 2 gold coins` -> 2; `a coin and a gold coin` -> 1; `a ledge and a floating platform` -> 1. Repetition too: `pipe pipe pipe` -> 1, `coin coin coin coin` -> 1, `pipes pipes` -> 2.
Cause: `counts[topic] = val` (last wins) l.388-391; the only trace is `DUPLICATE_TOPIC` in the SCENE warnings, and `normalize()` (l.569-570) copies up only CONTRADICTION / POSITION_NOT_PLANNED / GAPS_NOT_VERIFIED, so `Normalized.warnings` is [] for all of these. Contrast `3 coins and 3 more coins` = 6, which works.

## RC5 Distribution / group words are ignored  [MEDIUM]
`3 pipes with 2 goombas each` -> enemy 2 (expected 6); `2 pipes each with a coin` -> coin 1 (expected 2); `a goomba on every pipe` -> 1/1; `three of each: pipes, coins and goombas` -> 2/3-4/2 (expected 3 each); `3 each of pipes and coins`; `one of each`, `2 of each`, `everything` -> EMPTY_SCENE (loud). Cause: no multiplier grammar; "each/every/all" sit in ADJ (l.45) and are skipped.

## RC6 Bounds and ranges worded in words are accepted as exact numbers  [MEDIUM]
`2 or 3 pipes`, `two to three pipes`, `two or three coins` -> exactly 3 (validate_prompt l.73 rejects `2 to 3` with digits but not the word forms, so the same sentence is refused or silently accepted depending on how the numbers are typed); `between 2 and 4 coins` -> 4 (plus a bogus POSITION_NOT_PLANNED because of "between"); `at most 2 pipes`, `no more than 3 coins`, `up to 3 enemies` -> exactly 2/3/3 (the adherence check will then fail a level with 1 pipe that the user would accept); `at least 3 coins` -> exactly 3.

## RC7 Order words other than "then" collapse scenes  [MEDIUM]
`first 2 pipes, second 3 coins, third no enemies` -> ONE scene (pipe 2, coin 3, forbid enemy over the whole level; the user meant the third part); `firstly 2 pipes secondly 3 coins`; `1st: 2 pipes, 2nd: 3 coins`; `2 pipes in the first half, 3 coins in the second half` (no warning at all) and `... on the left ... on the right` (POSITIONAL_IGNORED in the scene only). `ORD_RE` l.65 handles only first/second/... at the start of a segment; `firstly`, `1st` are missing, and nothing splits on them.
`stairs down then stairs up` -> 1 descending, 2 ascending; `steps up` -> 2; `3 stairs` / `three steps` -> 3 staircases (a person means one staircase). The earlier "stairs up/steps up plural" decision still misreads the bare phrase.

## RC8 validate_prompt false positives  [MEDIUM]
Rejected as "ranges and decimals" (service.py:73, `\d\.\d|\d-\d|\d\s+to\s+\d`): `level 1-2 with pipes`, `world 1-1 with 2 pipes`, `2 pipes in 1-1`, `v2.0 level with 2 pipes`, `version 1.2 with coins`, `3 pipes at 4.30`, `3 pipes at 2-3 speed`. The "-3" rule (l.75) rejects `2 pipes -3 coins` and `2 pipes, -3 coins` (dash typed without spaces) though `2 pipes - 3 coins` passes.
Accepted although it should be refused: `2,5 pipes`, `1,5 pipes`, `two to three pipes`, `1e3 pipes`, `3 1/2 pipes` (RC2/RC6).

## RC9 Direct `normalize()` can raise  [LOW; not reachable through the service]
`normalize("9"*5000 + " pipes")` (also `the 999..9th pipe`, `pipes: 999..9`, `2 pipes (999..9)`) -> `ValueError: Exceeds the limit (4300 digits)` at l.266 `int(w.rstrip("+"))` (and l.355). `validate_prompt` blocks it only because `max_prompt_chars` is 400 (configs/generate/app.yaml:15); `full_level.py:130` and `service.py:139` call `normalize` on whatever they get, so any caller that skips validation crashes. Clamp the digit run before `int()`.

## RC10 Misc silent clamp / odd readings  [LOW]
`99 pipes`, `thirty pipes`, `four pillars` keep `want` = the stated count while the caption is clamped ("a few pipes", "two towers"); the clamp warning (`CLAMPED:pipe:many->a few`) lives only in the scene, not in `Normalized.warnings`.
`lots of coins` -> caption "several coins" but `many coins` -> "many coins" (same want (10,99), different caption). `two gaps and a chasm` -> 1 gap. `3 packs of goombas` -> 2.

## Held up (do not re-test)
digits/words 0-6 with plurals and uppercase, full-width digits, `3x pipes`, `3 coins and 3 more coins`, `no X, no Y`, `no X or Y / nor Y`, `without any X or Y`, `with no/zero/0 X`, `X (no Y)`, `X / no Y`, `5 coins | no pipes`, trailing `!!!`/`...`, tabs/NBSP/zero-width space, `a few pipes and lots of coins`, 18 kB inputs (under 0.2 s), contradictions (`3 pipes and no pipes` refused), flat family, gap warnings.

## 10 worst findings
1. `w/o pipes`, `sans pipes`, `wo pipes` -> 2 pipes requested (normalizer.py:188 `_NEG_PHRASES` closed list).
2. `not many coins` / `not too many goombas` -> MANY (10,99), the exact opposite (`_qty_before` l.249).
3. `everything but pipes`, `anything but enemies`, `besides/other than/apart from pipes`, `don't use cannons`, `stay away from pipes`, `keep the pipes out`, `ban the enemies` -> requested 2 (l.188).
4. `without a single pipe`, `without even one enemy`, `not one pipe`, `no 2 pipes`, `3 coins, but not pipes` -> the noun is REQUESTED (l.189 vs VAGUE "a single" l.39).
5. `2,000 coins` -> forbids coins (comma split, lone "000" = 0); `2,5 pipes` -> 5 and validate_prompt passes it (service.py:73).
6. `two pipez and a coin` -> pipes silently vanish, no warning at all (l.229 `_match_nouns`).
7. `a goomba and a koopa`, `2 goombas and 1 koopa`, `2 pipes and 3 tubes`, `3 coins plus 3 collectibles` -> synonyms overwrite instead of add; DUPLICATE_TOPIC stays in scene warnings (l.388, l.569).
8. `pipes: three, coins: two` / `pipes = 3, coins = 2` -> counts SWAPPED; `hard: 3 goombas` -> 2 (l.195 `_NOUN_COLON_NUM`).
9. `neither pipes nor coins`, `nor pipes`, `I'd rather not have enemies` -> requested (l.188).
10. `3 pipes with 2 goombas each` -> 2 goombas; `2 sets of 3 pipes` -> 3; `at most 2 pipes` / `2 or 3 pipes` -> exactly 2 / 3 (no multiplier or bound grammar).

Overall disagreement rate: about 125 / 570 = ~22 % of the invented prompts (roughly 80 are a prohibition read as a request or a count read as something else with no warning; the rest are loud refusals or false-positive rejections). Exceptions: 1 class (RC9, only when validate_prompt is skipped). Slow calls (over 1 s): none.
