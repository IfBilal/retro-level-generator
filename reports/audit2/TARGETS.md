# audit2 targets (written before the fixes of 2026-11-27; the audit-1 targets in `reports/audit/TARGETS.md` still apply and were never loosened)

Source: the 27 Nov mission file (`that.md`). Measured on current `main` (8fb5fb4) before any audit2 change, 2026-11-27.

| id | requirement | measured before | target |
|---|---|---|---|
| A2-1 | headline prompt ("three pipes and 5 enemies, 2 enemies before the first pipe and the rest after the 4th pipe") keeps 3 pipes / 5 enemies and says "no 4th pipe" | normalised to "two enemies. one pipe." (counts dropped, no message) | refused with a clear sentence, counts kept |
| A2-2 | flat / flatland / empty level / just a floor / no enemies produce what they say | flatland, empty level, castle with lava = unconditioned model; "a completely flat level" kept enemies allowed | 100% of 8 seeds each meet "nothing but floor" on the raw level |
| A2-3 | unmapped sentence (castle with lava, junk) never generates a random level | generated one | refused with examples |
| A2-4 | position words never silently dropped | dropped | honoured ("X at the start and Y at the end" = ordered scenes, 10/10 seeds) or listed in the UI as NOT controlled |
| A2-5 | what the player sees has the counts the page claims | 1 of 15 levels showed 2 pipes for 3 raw (the display repair deleted a pipe cut by the end template) | 0 of 15; counts judged on raw AND displayed grid |
| A2-6 | model RCA H1, H3, H4, H6, H8 answered with numbers | none | `MODEL_RCA.md` |
| A2-7 | fresh-seed adherence audit after the changes | 630/630 (audit 1) | no regression (>= 95%) |
| A2-8 | browser pass at 1920x1080, 1280x720, 390x844, 50 Dice clicks, no console error | 30 Dice, one viewport | all three viewports clean |
| A2-9 | deployed app re-tested after the push | audit 1 | gate_p8 5/5 + headline + flat on the deployed URL |
