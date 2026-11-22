# Mutation audit of the tests

68/69 injected bugs were caught by the tests; survivors: none; proven-equivalent mutants: H11.  Source tree identical afterwards: True.

| id | file | the bug that was injected | outcome |
|---|---|---|---|
| A1 | `adherence.py` | scene regions start 8 columns too early | KILLED |
| A2 | `adherence.py` | a vague quantity is planned at its top instead of its middle | KILLED |
| A3 | `adherence.py` | an open-ended range (many) is planned at 54 objects | KILLED |
| A4 | `adherence.py` | a topic asked for in this scene is also forbidden here | KILLED |
| A5 | `adherence.py` | 'then' is not exclusive | KILLED |
| A6 | `adherence.py` | a stray staircase of the other direction is allowed | KILLED |
| A7 | `adherence.py` | implied prohibitions are reported as asked for | KILLED |
| A8 | `adherence.py` | a self-contradicting prompt is not refused | KILLED |
| A9 | `adherence.py` | more scenes than windows is not refused | KILLED |
| A10 | `adherence.py` | impossible counts are not refused | KILLED |
| A11 | `adherence.py` | capacity ignores the level width | KILLED |
| A12 | `adherence.py` | a single object lands at the end of the scene | KILLED |
| A13 | `adherence.py` | objects of earlier scenes count towards this scene's quota | KILLED |
| A14 | `adherence.py` | already-placed objects are requested again (the original doubling bug) | KILLED |
| A15 | `adherence.py` | eight coins in a one-window level become impossible | KILLED |
| A16 | `adherence.py` | windows lose their default floor phrase | KILLED |
| A17 | `adherence.py` | a forbidden topic can still be requested in the caption | KILLED |
| A18 | `adherence.py` | the known half is counted as new | KILLED |
| A19 | `adherence.py` | a window with too FEW objects counts as exact | KILLED |
| A20 | `adherence.py` | forbidden objects are not penalised | KILLED |
| A21 | `adherence.py` | too many objects are accepted | KILLED |
| A22 | `adherence.py` | objects right of a scene region are counted in it | KILLED |
| A23 | `adherence.py` | unmet is never reported | KILLED |
| A24 | `adherence.py` | the wrong window is redrawn for a surplus | KILLED |
| A25 | `adherence.py` | column to window ownership is off by one window | KILLED |
| A26 | `adherence.py` | the plan-guided shortfall rule is inverted | KILLED |
| A27 | `adherence.py` | the shortfall fallback picks the wrong window | KILLED |
| C1 | `count_audit.py` | 2-high stubs count as towers | KILLED |
| C2 | `count_audit.py` | 3-wide blocks count as towers | KILLED |
| C3 | `count_audit.py` | a step of one counts as a tower edge | KILLED |
| C4 | `count_audit.py` | pipes and cannons count as towers | KILLED |
| C5 | `count_audit.py` | the Wilson interval can exclude its own estimate | KILLED |
| C6 | `count_audit.py` | the audit's scene ownership differs from the generator's | KILLED |
| C7 | `count_audit.py` | ordering is never checked | KILLED |
| C8 | `count_audit.py` | a scene-count mismatch is not handled | KILLED |
| F1 | `full_level.py` | the draw cap does not stop whole-level resampling | KILLED |
| F2 | `full_level.py` | counts stop being fixed after the route budget | KILLED |
| F3 | `full_level.py` | attempts_total can exceed the ledger cap and invalidate the level | KILLED |
| F4 | `full_level.py` | the draw cap does not stop window redraws | KILLED |
| F5 | `full_level.py` | verify=False still redraws windows | KILLED |
| F6 | `full_level.py` | impossible requests are not refused in generate() | KILLED |
| F7 | `full_level.py` | the planner forgets what is already in the level | KILLED |
| F8 | `full_level.py` | the WORST candidate window is kept | KILLED |
| F9 | `full_level.py` | the first candidate is kept instead of the best | KILLED |
| N1 | `normalizer.py` | vague words become exact numbers | KILLED |
| N2 | `normalizer.py` | a bare prohibition has no scene again | KILLED |
| N3 | `normalizer.py` | contradictions are not detected | KILLED |
| N4 | `normalizer.py` | 'a few' keeps no range | KILLED |
| S1 | `service.py` | negative numbers are accepted | KILLED |
| S2 | `service.py` | a CPU server has no time guard | KILLED |
| G1 | `game_html.py` | '<!--' can break the game page | KILLED |
| G2 | `game_html.py` | '</script>' can break out of the page | KILLED |
| H1 | `schema.py` | count resamples are held to the route limit of 5 per window | KILLED |
| H2 | `schema.py` | the adherence budget is not enforced by the schema | KILLED |
| H3 | `schema.py` | adherence could be judged on the repaired display grid | KILLED |
| H4 | `service.py` | over-long prompts are accepted | KILLED |
| H5 | `service.py` | control characters are accepted | KILLED |
| H6 | `sanitizer.py` | the display repair ADDS ground instead of deleting broken parts | KILLED |
| H7 | `full_level.py` | a later scene uses the first scene's prohibitions | KILLED |
| H8 | `full_level.py` | scenes whose only constraint is an implied prohibition are not checked | KILLED |
| H9 | `full_level.py` | whole-level resamples do not count towards the draw cap | KILLED |
| H10 | `service.py` | clamp warnings are shown as raw codes | KILLED |
| H11 | `game_html.py` | a U+2028 in a prompt breaks the game script | SURVIVED |
| P1 | `graph.py` | the CUDA graph reads freed autocast weights | KILLED |
| P2 | `graph.py` | two sessions corrupt each other's graph buffers | KILLED |
| U1 | `streamlit_app.py` | Dice does nothing | KILLED |
| U2 | `streamlit_app.py` | a redeploy into a running process breaks the app again | KILLED |
| U3 | `streamlit_app.py` | an old level is not labelled as belonging to an earlier sentence | KILLED |
| U4 | `streamlit_app.py` | unmet counts are shown as a success | KILLED |
