# One-shot test results (frozen step 18000, CFG w = 7.5)

Checkpoint sha256 `f471e2a56c2c4432...`, git `9f08566421`. 95 % intervals resample the **6 held-out levels** (clusters); with so few levels they are wide.

| Metric | P-test | P-rand |
|---|---|---|
| C-score | 0.819 [0.767, 0.861] | 0.348 [0.348, 0.348] |
| exact match | 0.034 [0.016, 0.051] | 0.007 [0.007, 0.007] |
| broken-structure rate (raw) | 0.102 [0.050, 0.168] | 0.377 [0.377, 0.377] |
| floor validity | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| checker-verified rate (framed windows, diagnostic) | 0.788 [0.700, 0.865] | 0.733 [0.733, 0.733] |
| median nearest-train Hamming | 13.000 | 28.000 |
| copy rate | 0.080 | 0.037 |
| tile JS divergence | 0.011 | 0.034 |
| rare-concept exact match | 0.577 | 0.182 |

References on the same prompts (a generator that ignores the prompt):

| Reference | P-test C-score | P-rand C-score |
|---|---|---|
| null_other_level | 0.463 | nan |
| always_empty | 0.495 | 0.413 |
| always_full_floor | 0.566 | 0.352 |
| always_modal | 0.626 | 0.289 |

P-casual-test (normalizer on, 30 prompts): scene-count accuracy 0.97, caption match vs the expected table 0.63, C-score of generated scenes vs expected 0.748.

Per-level, per-concept tables and the failure gallery are alongside this file.
