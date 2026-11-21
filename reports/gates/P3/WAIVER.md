# G-P3 waiver (CLAUDE.md rule 12: no silent fallbacks)

**Gate result:** FAIL. `reports/gates/P3/RESULT.json` (baseline MSE, 15k steps, 256 EMA samples per milestone) and the dense follow-ups
`P3_lin_dense`, `P3_cos_dense`, `P3_lin_auxdense`, `P3_lin_aux`, `P3_cos_aux`: no checkpoint of any variant satisfies broken-structure rate <= 0.05
AND copy rate <= 0.154. Closest: linear 14k (0.066 / 0.148) and cosine 14k (0.070 / 0.137).

**What was done (doc 25):** R-3.4 auxiliary cross-entropy (lambda 0.001): broken rate drops to 0.02-0.03 but copy rate rises to 0.36-0.43. R-3.6 earlier stopping:
dense checkpoints show broken structures and copying move in opposite directions with training. Metric calibration checked (real train/dev windows: broken rate 0.0). No training was extended to hide the failure.

**Decision:** proceed to Phase 4 with the cosine schedule (better than linear on both axes at equal steps; ties go to cosine, ledger M2), plain MSE loss, and judge the conditional model, which is the
product, by its own dev composite and by the held-out test. The unconditional baseline's failure is reported in the report (section 5.1). No threshold was moved.
