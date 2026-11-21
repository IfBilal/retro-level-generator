# Notes on the one-shot test run (read with RESULTS.md)

- **History (also in `test_invocations.jsonl`).** The first invocation computed P-test and then crashed in P-rand because one synthetic prompt needs 49 tokens and the encoder
  refuses to truncate silently (training limit 48). Nothing was saved. The only change before the second invocation: the evaluation encoder is built with `max_len = 64`
  (the number of prompts over 48 tokens is stored in `metrics.json` under `_token_lengths`). Model, seeds and prompts are unchanged and sampling is deterministic; the second
  run's P-test C-score (0.8192) equals the value printed by the crashed first run. No test number influenced any choice.
- **P-rand has no source levels.** Its prompts are random combinations of up to seven topics, not captions of real windows. Therefore (i) a level-clustered interval does not exist (a single cluster:
  the interval collapses to the point estimate, so the report shows no interval), and (ii) the cross-level "null" C-score is undefined (`nan`). The constant-generator rows are still meaningful.
- **P-rand is a stress test, and the model does poorly on it:** C-score 0.348 is below an always-empty generator (0.413). Random seven-topic prompts contain unseen combinations and rare
  concepts; the model also produces broken structures (rate 0.377), each of which costs 2/18 of the C-score (a known property of the metric, doc 22 Q3).
- **P-casual-test:** the "caption match" compares our normalizer (which clamps to the *train-split* supported buckets) with the full-corpus clamp table printed in the specification, so differences are expected.
