# Adherence targets (written BEFORE the first measurement; changes are logged at the bottom with reason + timestamp)

Committed 2026-11-21 (UTC), branch `audit/prompt-adherence`, before any baseline run.

## What is measured
The RAW generated grid (`level["raw_grid"][:, 3:3+width]`, i.e. without the 3 spawn + 6 end-template columns), through the same
path the app uses (`LevelService.run`, fast mode, default CFG 7.5, width 48), 10 fixed seeds per prompt (1000..1009).
Counts are decided by the repo's own caption detector (the definition the training captions were made with) AND cross-checked
by an independent counter (`text2mario.eval.count_audit`). Quantity words use ledger F15: one=1, two=2, a few=3-4, several=5-9, many>=10.
"three" is judged as exactly 3 (what a user means), even though the caption grammar can only say "a few" (3-4): that is a
structural limit of the frozen model's conditioning and it is reported, not hidden.

## Targets (pass = share of the 10 seeds whose raw level satisfies the prompt)
| id | scope | target | justification |
|---|---|---|---|
| T1 | exact counts 0-3, pipe / enemy / cannon / question block / staircase / tower / platform, per prompt | >= 90% of seeds | what a user sees: "one pipe" must be one pipe |
| T2 | coin counts (cells) 0-3 | >= 90% | same |
| T3 | scene ordering ("X, then Y") | 100% of seeds | ordering is positional and checkable |
| T4 | negation ("no X", "without X") | 100% of seeds | absence is as visible as presence |
| T5 | the example prompt (one pipe, then several coins and a staircase) | 10/10 seeds | the reported bug |
| T6 | quantities with the same object in two scenes | >= 90% | same as T1 |
| T7 | synonyms (tube, warp pipe, chimney...) keep the count | >= 90% | lexicon claim |
| T8 | unsatisfiable / malicious / junk input | 100%: clear message or rejection, never a stack trace, never injected markup, never a silent clamp | safety |
| T9 | determinism (same prompt+seed+settings -> byte-identical level) | 100% | repo claim |
| T10 | zero exceptions / red boxes / console errors of our own / failed requests of our own in the browser pass, incl. 30 Dice clicks | 0 | P1 |

The frozen model was never trained to count exactly, so T1/T2/T6/T7 may be unreachable by the model alone. The plan (only if the
baseline shows it) is: level-level count planning + deterministic verification of the RAW level against the parsed constraints with
visible, logged resampling attempts, and a clear message when the budget is exhausted. Any grid edit would be a labelled REPAIR and
never counted as adherence.

## Change log
(none yet)
