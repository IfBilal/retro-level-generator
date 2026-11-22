# Audit report: prompt adherence and app robustness (branch `audit/prompt-adherence`)

Everything below is generated from saved files by `scripts/build_audit_report.py`; the evidence index is `evidence/audit/INDEX.md`. Nothing here is copied from memory.

## 1. Bottom line

* **The reported bug was real and has a root cause.** On the unmodified app, **{{base_overall}}** generated levels ({{base_pct}} %) matched what the sentence asked for (counts, ordering, negation); the example prompt
  "flat ground, one pipe, then several coins and a staircase" matched in 3 of 40. After the fix **{{A_overall}}** ({{A_pct}} %) match on the same prompts and seeds, and **{{B1_overall}}** on 10 fresh seeds
  never used while tuning, and **{{B2_overall}}** on 10 held-out prompts written before tuning. No target in `reports/audit/TARGETS.md` was lowered.
* **The Dice crash was real** (`StreamlitWidgetAlreadyInstantiatedError`, reproduced on the deployed app and locally, third click) and is fixed with `on_click` callbacks; 30 consecutive clicks, mixed with
  generations, give zero exceptions in the browser ({{br_dice}} clicks without a problem).
* **It is not "perfect".** What is still wrong is in section 7 (levels are sparser, "tower" needed a definition change, count matching on the CPU/cloud is weaker, the checker/model limits of the project remain,
  four older gates still fail).

## 2. Diagnosis: which of the suspected causes were true

| | suspect | verdict | evidence |
|---|---|---|---|
| a | the normalizer mishandles or drops numbers | **partly true** | numerals and vague words parse correctly (30/30 dev prompts, 100 % passthrough), but: bare negation ("no pipes") produced NO scene so the prohibition vanished; "-3 pipes" became "two pipes"; "one pipe and no pipes" was silently emptied; "500 pipes" was silently clamped. All fixed (`tests/test_normalizer_want.py`, `tests/test_adherence_plan.py`) |
| b | the caption is count-blind | **true, partly** | the caption can only say five buckets (one / two / a few / several / many), so "three" became "a few" (3-4) and "three towers" "two"; the exact request is now kept in `Scene.want` |
| c | the diffusion model cannot control exact counts | **true, partly** | inside one 16-column window it hits "one pipe" {{we_pipe1}}, "two pipes" {{we_pipe2}}, "one enemy" {{we_enemy1}}, "one cannon" {{we_cannon1}}, "a few enemies" {{we_enemy3}} (it draws 4-10) of the sampled windows (`reports/audit/window_exactness.json`). The fix works around this; it does not remove it |
| d | multi-scene stitching duplicates objects | **TRUE: the root cause** | the scene caption was given to EVERY 16-column window of the scene, so "one pipe" over a 48-column level (5 windows) asked for one pipe per window |
| e | the verify/resample loop checks playability, not adherence | **true** | `generate()` resampled only for route failures and forbidden words; it accepted any count |
| f | the sanitizer adds or duplicates objects | **disproved** | `repair` only deletes cells; property test over random grids: every changed cell is SKY and no tile type increases (`test_display_repair_only_removes_cells_never_adds_them`); metrics and adherence use the raw grid (schema enforces `judged_on == "raw"`) |
| g | renderer / "what the model will understand" panel differ from what the generator received | **partly true** | the panel showed the scene captions of the CURRENT text box, so an edited sentence showed a different panel than the level beside it; the generator actually sent per-window captions. Now the old level is labelled "generated for an earlier sentence", the Details tab lists the per-window captions that were sent, and the browser audit checks panel == recorded captions |

Full data flow, naming every function that touches a count: `reports/audit/DATAFLOW.md`.

## 3. The fix in one paragraph

Counts are planned across windows and verified on the raw level (ledger section 7b, AD1-AD9, all numbers in `configs/generate/app.yaml`): each window is told only what it must ADD in its 8 new columns
(describing an object already in the known half made the model add another: {{we_known_one}} new-half pipes with the truthful caption vs {{we_known_none}} without), a window is redrawn until its new columns match
(GPU: 32 candidates in one pass; CPU: one draw, then a small batch on a miss), the whole raw level is counted per scene region and the window the plan gave the missing object to is resampled, within hard
budgets (16 whole-level resamples, 40 extra window draws, the 45 s cloud guard). Contradictions, impossible counts, negative numbers and more scenes than windows are refused with a message. A count that cannot
be met is reported in the page; **no tile is ever edited** for adherence.

## 4. Before / after (the same 630 rows: 61 prompts x 10 seeds = 610 levels, plus 20 input-safety checks)

Baseline = the unmodified code (git `{{base_git}}`, UTC {{base_utc}}; its input-safety rows were re-run with the corrected injection check on the same code). Round C = final code (git `{{A_git}}`, UTC {{A_utc}}), same prompts and seeds as the baseline.

{{table_base_A}}

Prompts below target in round C: {{misses_A}}

**Round D, fresh seeds** (7000-7009, used only for this round; git `{{B1_git}}`, UTC {{B1_utc}}; code unchanged since round C):

{{table_B1}}

Prompts below target in round D: {{misses_B1}}

**Round D, held-out prompts** (written before any tuning, `configs/audit/adherence.yaml: prompts_fresh`, the pristine seeds): {{B2_overall}} levels match.

{{table_B2}}

Held-out prompts below target: {{misses_B2}}

Stop condition: rounds C and D are two consecutive full rounds with no code change between them (`git log` shows no `src/` or `configs/` commit after `{{A_git}}` and before round D ended; the working tree was clean apart from
generated reports). Round time per level (median / p95 / max seconds, GPU): baseline {{base_t50}} / {{base_t95}} / {{base_tmax}}, round C {{A_t50}} / {{A_t95}} / {{A_tmax}}, round D {{B1_t50}} / {{B1_t95}} / {{B1_tmax}}.
Levels with a route found by the playability checker: baseline {{base_verified}} of {{base_levels}}, round C {{A_verified}} of {{A_levels}} (the fix did not make the checker's job worse).

**The earlier confirmation attempt (rounds A and B, history).** Round A (baseline seeds) matched 630/630 and round B (fresh seeds 5000-5009) 629/630 with 100/100 on the held-out prompts:

{{table_dev}}

The one miss in round B ("three staircases" produced four because an unrequested descending staircase appeared; within the per-prompt target, but a real defect) was fixed (D18) and the confirmation was
repeated from scratch on pristine seeds as rounds C and D above, because a code change invalidates "no change between rounds".

## 5. Every defect found (severity, root cause, fix, regression test)

| # | severity | defect | root cause | fix | regression test |
|---|---|---|---|---|---|
| D1 | high | Dice crashes the page (`StreamlitWidgetAlreadyInstantiatedError`), also on the deployed app | `st.session_state.seed` written after the `number_input(key="seed")` existed | `on_click` callbacks for Dice and the example chips | `tests/test_app_ui.py::test_dice_never_raises_widget_already_instantiated` (fails on the old code with the exact error) |
| D2 | critical | counts, ordering and negation are not followed ({{base_pct}} % of levels matched) | scene caption sent to every window; 5-bucket captions; loose per-window counts; the loop never checked counts | level-level planning + raw-level verification (section 3) | `tests/test_full_level.py::test_one_pipe_gives_one_pipe_not_one_per_window` (documents the old behaviour, then the fix), `tests/test_adherence_*.py`, property tests |
| D3 | high | "no pipes" alone vanished (an unconditioned level, free to contain pipes) | `normalize` dropped scenes with no positive topic | a neutral "full floor." scene carries the prohibition, verified on the raw level | `tests/test_adherence_plan.py::test_bare_negation_survives_as_a_scene_with_a_forbid` |
| D4 | medium | "one pipe and no pipes" silently produced an arbitrary level | the empty scene was dropped | refused: "asks for and rules out the same thing" | `test_contradiction_is_refused_with_a_clear_message`, UI test |
| D5 | medium | "-3 pipes" became "two pipes"; "500 pipes" was silently clamped to "a few" | sign dropped by the tokenizer; clamp table only warned in a scene note | rejected / refused with a message | `test_negative_numbers_are_rejected_by_validation_but_ranges_are_not`, `test_impossible_count_is_refused_not_clamped_silently` |
| D6 | medium | an edited prompt left the old level on screen labelled as if for the new text | no link between result and sentence | "generated for an earlier sentence" notice (plain text, never markdown) | browser audit: `state: edited prompt -> old level is labelled` |
| D7 | low/medium | a prompt containing `<!--<script>` could break the game page | `<!--` not escaped in the inlined level JSON | escape `<!--` as well as `</` | `tests/test_level_schema.py::test_a_prompt_cannot_break_out_of_the_game_page` (2 payloads fail on the old code) |
| D8 | high (latency) | one level took 177 s | 16 whole-level resamples x 8 redraws per window for a count the seed could not reach | hard cap `adherence.max_extra_draws` (deterministic) | `test_unsatisfiable_request_has_a_hard_bounded_cost` |
| D9 | medium | levels stopped with a count unmet after the route budget ran out ("underground" example 6/10) | the loop broke when the route budget was spent | counts keep being fixed after the route budget; `attempts_total` capped at the ledger limit (a 21st attempt would have made the level invalid) | `test_counts_keep_being_fixed_after_the_route_budget_is_spent`, `test_attempts_total_never_exceeds_the_ledger_cap...` |
| D10 | medium | scenes beyond the number of windows were silently dropped yet judged | `scene_for_windows` maps them all to scene 0 | refused: "has room for N scenes" | `test_more_scenes_than_windows_is_refused_not_silently_dropped` |
| D11 | medium | "two towers" matched while the picture showed one tower and a staircase | the caption detector counts a staircase's narrow top as a tower and merges pillars (round 5, generated before the rule: the detector said {{tw_r5_det}} of {{tw_r5_n}} "N towers" levels match, a person {{tw_r5_vis}}; round C, after it: a person {{tw_A_vis}} of {{tw_A_n}}, detector {{tw_A_det}}) | towers counted as a person sees them (AD9); detector kept as cross-check | `tests/test_count_audit.py` (7 tower tests), `reports/audit/NOTES_towers.md` |
| D12 | high (found by the test ordering) | after the app ran, a later test touched the GPU and got `illegal memory access` | the CUDA-graph U-Net replay captured bf16 weight copies from the autocast cache, freed when the autocast block exited | capture with `cache_enabled=False` + a lock for shared sessions | `test_capture_under_autocast_survives_the_autocast_cache_being_cleared`, `tests/test_zz_cuda_state.py` sentinel |
| D13 | low | `Permissions policy violation: focus-without-user-activation` console error on every game load | Phaser's boot-time `window.focus()` inside the sandboxed iframe | `autoFocus: false` (the click-to-play overlay focuses the canvas) | `tests/e2e/test_game_smoke.py::test_game_embedded_in_a_sandboxed_iframe_logs_no_console_error` |
| D14 | low | Wilson interval excluded its own point estimate for k = n (float rounding) | `0.9999999999999999 < 1.0` | clamp to the estimate | `test_wilson_interval_properties` (found by hypothesis) |
| D15 | low | cryptic warning codes shown to users (`CLAMPED:tower:a few->two`) | raw codes in the UI | plain-English texts | `tests/test_normalizer_want.py` |
| D16 | tool | `gate_p8.py` could not see the Streamlit Cloud app (it lives in an iframe) and clicked the hint text instead of the button | selector | frame-aware, role-based selector | run against the deployed URL |
| D18 | low | "three staircases" matched the generator's check but a person saw four (an unrequested staircase going down) | the check counted ascending staircases only | asking for staircases of one direction rules out the other direction in that scene | `test_asking_for_staircases_of_one_direction_rules_out_the_other_direction_in_that_scene`, `test_a_stray_descending_staircase_makes_three_staircases_unmet` |
| D19 | **critical (production)** | right after the audit's push the DEPLOYED app showed `ImportError: cannot import name 'explain_warning'` on every load | Streamlit Community Cloud keeps ONE Python process across deployments and only reloads the script folder (`app/`), never `src/`: the new script met the old `text2mario` modules in memory | `drop_stale_modules()` in `app/streamlit_app.py`: if `src/` changed under a running process, drop the stale modules and the cached model and re-import | `tests/test_app_ui.py::test_app_survives_a_redeploy_into_a_running_process_with_stale_modules` (fails with the exact production error without the fix), `test_unchanged_sources_do_not_drop_modules_on_every_rerun` |
| D17 | hygiene | 259 lint findings repo-wide | accumulated | all fixed or exempted with a written reason (`pyproject.toml`) | `ruff check src scripts app tests` clean |

## 6. Contrarian checks (each run on purpose; outcome)

1. **Is the counter right?** Hand-built grids (two touching pipes, pipe next to a tower, hanging pipe, boundaries): `tests/test_count_audit.py`. An independent re-count of pipes, enemies, coins, cannons and question
   blocks agrees with the detector on {{A_counter_levels}} levels with {{A_counter_disagree}} disagreements. A contact sheet of 12 levels was checked by eye (`evidence/audit/*/visual_check_*.png`): 11 matched; the
   12th exposed the tower definition problem (D11), which is why towers are now counted as a person sees them.
2. **Lucky seeds / easy prompts?** Round D uses 10 seeds and 10 prompts that were never used while tuning (round B used another 10 seeds that were no longer pristine after its miss was analysed). Result above.
3. **Does it hurt variety?** No collapse to one level: distinct grids per prompt {{var_distinct_b}} -> {{var_distinct_a}} of 10 (fewest for any prompt: {{var_min}}, the "no X" prompts, which are flat floors by
   definition). **But the levels are sparser**: mean pairwise cell difference {{var_ham_b}} -> {{var_ham_a}}, share of plain floor/sky cells {{var_plain_b}} -> {{var_plain_a}}. That is the price of windows that must
   hold exactly zero of an unrequested thing (`reports/audit/variety.md`).
4. **Playability / checker.** End to end on identical jobs (`scripts/audit_bot_replay.py`: the checker's route replayed by a plain ghost bot in the real game, no re-planning, enemies present, so a harsh lower bound): route found {{bot_b_solved}}/{{bot_jobs}} -> {{bot_a_solved}}/{{bot_jobs}}, bot reaches the flag {{bot_b_won}} -> {{bot_a_won}} (the new levels are sparser, which is part of why). With the re-planning bot (enemies removed, `scripts/replay_levels.py`): {{rl_sc}} verified levels of the regenerated showcase finished (6-7 of 10 before) and {{rl_p5f}} verified 112-column levels of the final-pipeline G-P5 subset (8 of 20 before, a different sample). The checker itself was not changed; levels with a route: baseline {{base_verified}}/{{base_levels}}, final {{A_verified}}/{{A_levels}}. Gates after the work: {{gates}}. (G-P3, G-P4, G-P5,
   G-P5b are model/checker-quality gates; G-P7 now passes because the sampler got faster.) **G-P5.01 (seams) got no better and its mean got worse:** re-running G-P5 with the final pipeline on the first 20 jobs (same dev prompts
   and seeds, `reports/gates/_p5_adherence_check`, `reports/audit/gate_p5_compare.json`) gives a seam ratio median {{p5_b_med}} -> {{p5_a_med}}, mean {{p5_b_mean}} -> {{p5_a_mean}} (max {{p5_b_max}} -> {{p5_a_max}}; limit 1.3, it failed before too),
   with {{p5_b_ver}} -> {{p5_a_ver}} levels route-verified and 0 broken structures after the display repair in both. Windows now get different captions (a carrier window next to plain floor), which makes a few seams worse; this is a cost of the fix, reported, not fixed.
5. **Cloud situation (45 s guard, CPU).** CPU, no profile variable set (the guard now applies to any machine without a GPU, because Streamlit Cloud may not have `T2M_PROFILE` set; found while planning the deployment), 10 prompts x 5 seeds: {{cloud_overall}} match, median {{cloud_t50}} s, max {{cloud_tmax}} s; the 45 s guard ended {{cloud_budget_hits}} level(s) early
   (all the "underground" example: {{cloud_ex3}} match on CPU). Counts are weaker on CPU because candidates cost real time there. This machine's CPU is not Streamlit Cloud's 2-vCPU box; the deployed app is checked separately.
6. **Numbers re-measured, not remembered:** latency (below), tests ({{tests_total}} passing), determinism (`reports/gates/determinism.json`), every pass rate (artifacts above).
   Latency on the RTX 3060 Ti in the app's default fast mode: 48 columns {{lat_cuda_48_fast_p50}} s median (p95 {{lat_cuda_48_fast_p95}}), 112 columns {{lat_cuda_112_fast_p50}} s (p95 {{lat_cuda_112_fast_p95}});
   CPU: {{lat_cpu_48_p50}} s and {{lat_cpu_112_p50}} s; cold start {{lat_cold_start_s}} s.
7. **Metric corrections** (logged with reasons in `reports/audit/TARGETS.md`): injection check measured inert text instead of a breakout; exclusive scenes; tower definition. Targets themselves were never lowered.

## 7. What is still imperfect (blunt)

* **Sparser levels** (above): the fix makes requested things exact by keeping unrequested things out of their windows.
* **Towers needed a definition change** and the same ambiguity may exist for other structure words; pipes, enemies, coins, cannons and question blocks are tile-defined and unambiguous. Platforms, staircases and
  coin lines use the caption detector's definition unchanged.
* **The model itself is not better at counting.** The planner and the verifier compensate with extra draws; on the CPU/cloud they have fewer draws within 45 s (section 6.5), so the deployed app matches counts less
  often than the GPU numbers in section 4, and the "underground" example may report unmet counts there. The page says so when it happens.
* **Old gates still fail:** G-P3, G-P4 (P4.04), G-P5 (seam ratio), G-P5b (checker vs the reference solver). They are about model/data/checker quality and were out of scope (no retraining, frozen checkpoint).
  "Route found" still means "found by our checker", not "beatable by every player".
* **No human played the levels**, and the visual count check covered 12 levels.
* **Windows clock / Windows asyncio:** the server log shows one benign `ConnectionResetError` when a browser is killed mid-request (Windows); no application traceback appeared in any run.
* `reports/audit/round1_partial_counts`, `round2_noncount`, `round3`-`round5`, `roundA`, `roundB_*` are development rounds kept as history; the numbers of record are `baseline_full`, `roundC_baseline_seeds`, `roundD_*`, `cloud_cpu`.

## 7b. The tests were tested too

A test suite that cannot fail proves nothing, so the new tests were put on trial (`scripts/audit_tests_mutation.py`, `audit_test_hygiene.py`, `audit_test_robustness.py`):

* **Mutation audit:** {{mut_total}} deliberate bugs were injected one at a time into the new code (a flipped comparison, a deleted guard, a changed constant, a removed escape...); the tests caught {{mut_killed}} of them
  (`reports/audit/test_mutation.md`). Survivors: {{mut_survivors}}; proven-equivalent mutants: {{mut_equiv}}. The first pass caught 55 of 58 and the next batch 7 of 11: **every survivor exposed a real gap** (the
  height threshold of a tower, the Wilson interval's lower clamp, the draw cap and best-candidate choice of the sequential redraw loop, scene-specific prohibitions, the "earlier sentence" notice, the draw budget of whole-level
  resamples), each is now pinned by a test that fails when the bug is put back. A mutation that created an infinite loop was caught by the timeout, and the harness restores the source after every mutation and heals itself after a forced kill (it did once).
* **Hygiene scan:** {{hyg_tests}} test functions parsed; flagged: {{hyg_flagged}}. The scan found five weak OLD tests (one `assert ... or True` that could never fail, four fuzz tests whose only check was "does not raise"); they now assert something.
* **Flakiness and order:** the fast suite under three different hypothesis seeds and the whole suite (fast + slow) in reverse file order: {{rob_summary}}.
* **Coverage** (branch coverage of the changed modules, fast + slow tests together): count_audit 100 %, adherence 99 %, full_level 98 %, game_html 100 %, schema 94 %, normalizer 94 %, service 86 % (the rest of `service.py` is model loading, covered by the real-model UI tests).

## 8. Browser audit (real Chromium against the running app)

Evidence directory `evidence/audit/{{br_dir}}/` (screenshots + JSON; index in `evidence/audit/INDEX.md`): 30 Dice clicks {{br_dice}} without error, own console errors / page errors / failed requests: {{br_defects}};
state, viewport (1920x1080, 1280x720, 390x844), seed edge cases, keyboard-only use, offline (every non-local request blocked), play-through, determinism and iframe-equals-service checks: {{br_states}} pass.
The deployed app: `scripts/gate_p8.py` {{deployed_pass}}; the deployed run is in `evidence/deployment/`.

## 9. Reproduce every number

```
python scripts/audit_adherence.py --label NAME --seeds baseline      # or --seeds fresh, --fresh-prompts, --only cat,cat, --device cpu with T2M_PROFILE=cloud
python scripts/merge_audit_rounds.py --out baseline_full --parts baseline:non-adversarial baseline_adv:adversarial
python scripts/audit_variety.py --before baseline_full --after roundC_baseline_seeds
python scripts/audit_towers.py --run baseline_full --run roundC_baseline_seeds
python scripts/audit_visual_check.py --run roundC_baseline_seeds --out evidence/audit/<ts>
python scripts/audit_window_exactness.py                              # per-window exact-count rates
python scripts/audit_browser.py http://localhost:8501 --dice 30 --out evidence/audit/<ts> --name final_local
python scripts/audit_browser_states.py http://localhost:8501 --out evidence/audit/<ts>
python scripts/bench_app.py --prompts 5 --cpu && python scripts/gate_p7.py
python scripts/gate_p8.py <app URL>
pytest -q            # fast + slow;  pytest -q -m "not slow" for the fast suite;  ruff check src scripts app tests
python scripts/build_audit_report.py
```

## 10. Commits on `audit/prompt-adherence`

{{commits}}
