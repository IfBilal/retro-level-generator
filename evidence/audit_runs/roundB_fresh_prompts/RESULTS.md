# Adherence audit `roundB_fresh_prompts`

device **cuda**, mode **fast**, profile **local**, width 48, seeds [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009], git ddcd869, UTC 2026-11-22T00:34:03Z

**Overall: 100/100 levels pass (100.0%)** (adversarial rows included)

## By category

| category | pass / n | rate | 95% CI | target |
|---|---|---|---|---|
| fresh | 100/100 | 1.00 | 0.96-1.00 | 0.90 |

## Per prompt

| id | prompt | pass/n | CI | median s | meets target |
|---|---|---|---|---|---|
| fr_mix1 | `one pipe, then three coins, then one enemy` | 10/10 | 0.72-1.00 | 3.315 | yes |
| fr_mix2 | `two platforms and one question block` | 10/10 | 0.72-1.00 | 3.3499999999999996 | yes |
| fr_mix3 | `a staircase and two pipes` | 10/10 | 0.72-1.00 | 3.355 | yes |
| fr_mix4 | `three enemies and no pipes` | 10/10 | 0.72-1.00 | 3.31 | yes |
| fr_mix5 | `one tower, then one cannon` | 10/10 | 0.72-1.00 | 3.3200000000000003 | yes |
| fr_mix6 | `two question blocks, then a coin line` | 10/10 | 0.72-1.00 | 3.32 | yes |
| fr_mix7 | `several coins and one platform` | 10/10 | 0.72-1.00 | 3.3200000000000003 | yes |
| fr_mix8 | `a warp pipe, then two goombas` | 10/10 | 0.72-1.00 | 3.325 | yes |
| fr_mix9 | `make me a level with exactly three pipes` | 10/10 | 0.72-1.00 | 3.325 | yes |
| fr_mix10 | `no enemies, just two platforms` | 10/10 | 0.72-1.00 | 3.315 | yes |

## Counter cross-check: 2 of 100 levels where detector and independent counter disagree

- fr_mix3 seed 5006: {'pipe': [2, 1]} (detector, independent)
- fr_mix9 seed 5002: {'pipe': [3, 4]} (detector, independent)

Tower cross-check (ledger AD9): the visible-tower count differs from the caption detector's tower count in 3 of 100 levels (the detector also calls a staircase's narrow top a tower and merges neighbouring pillars).

verification status counts: {'verified': 99, 'not_verified': 1}
