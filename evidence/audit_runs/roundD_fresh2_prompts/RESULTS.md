# Adherence audit `roundD_fresh2_prompts`

device **cuda**, mode **fast**, profile **local**, width 48, seeds [7000, 7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009], git cb7d3e2, UTC 2026-11-22T02:11:43Z

**Overall: 100/100 levels pass (100.0%)** (adversarial rows included)

## By category

| category | pass / n | rate | 95% CI | target |
|---|---|---|---|---|
| fresh | 100/100 | 1.00 | 0.96-1.00 | 0.90 |

## Per prompt

| id | prompt | pass/n | CI | median s | meets target |
|---|---|---|---|---|---|
| fr_mix1 | `one pipe, then three coins, then one enemy` | 10/10 | 0.72-1.00 | 3.3 | yes |
| fr_mix2 | `two platforms and one question block` | 10/10 | 0.72-1.00 | 3.315 | yes |
| fr_mix3 | `a staircase and two pipes` | 10/10 | 0.72-1.00 | 3.31 | yes |
| fr_mix4 | `three enemies and no pipes` | 10/10 | 0.72-1.00 | 3.29 | yes |
| fr_mix5 | `one tower, then one cannon` | 10/10 | 0.72-1.00 | 3.32 | yes |
| fr_mix6 | `two question blocks, then a coin line` | 10/10 | 0.72-1.00 | 3.32 | yes |
| fr_mix7 | `several coins and one platform` | 10/10 | 0.72-1.00 | 3.315 | yes |
| fr_mix8 | `a warp pipe, then two goombas` | 10/10 | 0.72-1.00 | 3.3099999999999996 | yes |
| fr_mix9 | `make me a level with exactly three pipes` | 10/10 | 0.72-1.00 | 3.31 | yes |
| fr_mix10 | `no enemies, just two platforms` | 10/10 | 0.72-1.00 | 3.3049999999999997 | yes |

## Counter cross-check: 0 of 100 levels where detector and independent counter disagree


Tower cross-check (ledger AD9): the visible-tower count differs from the caption detector's tower count in 6 of 100 levels (the detector also calls a staircase's narrow top a tower and merges neighbouring pillars).

verification status counts: {'verified': 99, 'not_verified': 1}
