# Commit map (audit branch)

The audit branch was rewritten once before the push to remove local run logs that contained machine addresses (`reports/streamlit_local.log` etc.). Run metadata in `reports/audit/*/results.json` and
`RESULTS.md` quote the short hash of the commit under test as it was BEFORE the rewrite; the tree content of every commit is unchanged except for those removed log files.

| before rewrite | after rewrite | subject |
|---|---|---|
| `6f388a2` | `6f388a2` | fix(app): Dice and example buttons use on_click callbacks (StreamlitWidgetAlreadyInstantiatedError); regressio |
| `ec4d73d` | `ec4d73d` | fix(gate): gate_p8 looks inside the Streamlit Cloud app iframe; feat(audit): real-browser audit script |
| `da3a89d` | `da3a89d` | feat(audit): adherence harness, independent counter with hand-built-grid tests, targets written before measuri |
| `b4c0e5f` | `b4c0e5f` | fix(text): normalizer keeps exact requested counts (want), bare negation, contradiction warning; tests |
| `d0c2505` | `d0c2505` | feat(generate): level-level count planning per window + raw-level verification against the parsed counts; UI s |
| `0337a8c` | `0337a8c` | fix(generate): plan only the NEW objects per window, redraw a window until its new columns match (best of N),  |
| `3119b47` | `3119b47` | fix(render): escape '<!--' in the inlined level JSON (a prompt could break the game page); show the previous s |
| `41b710c` | `41b710c` | feat(audit): baseline of the unmodified code (113/628 levels pass), data-flow notes, held-out prompts, deploye |
| `9793739` | `9793739` | test(adherence): property-based + decision-coverage tests (partition, planner completeness, counter agreement, |
| `fb66b34` | `fb66b34` | feat(audit): browser state/viewport/seed/keyboard/offline/play/equality audit script |
| `da2487f` | `da2487f` | fix(generate): hard cap on extra window draws spent on counts (a seed that could not be satisfied took 177 s); |
| `91dc28e` | `91dc28e` | style: ruff safe auto-fixes (unused imports/noqa, import order); fast suite unchanged (371 passed) |
| `bac6891` | `bac6891` | style: repo-wide ruff clean (unused locals, dead conditions, NaN idiom, blind test exceptions); documented per |
| `a1a0dbc` | `a1a0dbc` | feat(generate): CUDA batched candidate windows for count matching (one forward pass, keep the first exact or c |
| `7ef2618` | `7ef2618` | feat(generate): CPU/cloud draws one window first and a small batch only on a miss (measured 8 candidates ~2.5x |
| `4b60943` | `4b60943` | fix(generate): refuse more scenes than windows (the last scenes were silently dropped yet judged); gallery sho |
| `f0e40e4` | `f0e40e4` | feat(generate): exclusive scenes (then = what is asked elsewhere is ruled out here), redraw the window the pla |
| `28ef9ce` | `28ef9ce` | feat(audit): corrected injection metric (real breakout, not inert text), adversarial-only mode, <!-- and </scr |
| `05702a6` | `05702a6` | feat(slides): editable PPTX built from the same slide data as the PDF; audit results on the slides; viva guide |
| `84c394a` | `84c394a` | fix(generate): keep fixing counts after the route budget is spent (all failing 'underground' levels stopped th |
| `a43cea0` | `a43cea0` | fix(count): towers are counted as a person sees them (the detector counted a staircase top as a tower and merg |
| `3c1649d` | `3c1649d` | docs(audit): targets change log, ledger AD8/AD9, tower definition check script and notes |
| `6a8846f` | `6a8846f` | test: service path/boundary tests, UI notice tests (counts met / unmet / refused / time limit), sanitizer prop |
| `20eb918` | `20eb918` | fix(game): Phaser autoFocus off (a console error inside the sandboxed iframe on every load); regression test e |
| `ddcd869` | `ddcd869` | perf(sampler): CUDA-graph replay of the U-Net forward in the app window sampler (4.6x per window; lock for sha |
| `34a0ed9` | `7bdceae` | fix(count): asking for staircases of one direction rules out the other direction in that scene (a fresh-seed m |
| `cb7d3e2` | `5bd899e` | fix(app): the 45 s time guard applies on any machine without a GPU (Streamlit Cloud is CPU-only and the profil |
| `3471397` | `c12e670` | docs,evidence: audit report (generated from artifacts), final rounds C/D, evidence index, report/slides/pptx r |
| `07a2ad9` | `a06cd06` | chore: stop tracking local run logs (machine addresses, absolute paths); every number is in reports/audit/*/ |
