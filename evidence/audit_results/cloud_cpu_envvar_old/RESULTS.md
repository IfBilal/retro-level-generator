# Adherence audit `cloud_cpu`

device **cpu**, mode **fast**, profile **cloud**, width 48, seeds [1000, 1001, 1002, 1003, 1004], git 3c1649d, UTC 2026-11-21T22:15:21Z

**Overall: 47/50 levels pass (94.0%)** (adversarial rows included)

## By category

| category | pass / n | rate | 95% CI | target |
|---|---|---|---|---|
| count | 30/30 | 1.00 | 0.89-1.00 | 0.90 |
| example | 7/10 | 0.70 | 0.40-0.89 | 1.00 |
| negation | 5/5 | 1.00 | 0.57-1.00 | 1.00 |
| ordering | 5/5 | 1.00 | 0.57-1.00 | 1.00 |

## Per prompt

| id | prompt | pass/n | CI | median s | meets target |
|---|---|---|---|---|---|
| cnt_pipe_1 | `one pipe` | 5/5 | 0.57-1.00 | 3.21 | yes |
| cnt_pipe_2 | `two pipes` | 5/5 | 0.57-1.00 | 3.17 | yes |
| cnt_enemy_2 | `two enemies` | 5/5 | 0.57-1.00 | 5.8 | yes |
| cnt_coin_3 | `three coins` | 5/5 | 0.57-1.00 | 5.96 | yes |
| cnt_platform_2 | `two platforms` | 5/5 | 0.57-1.00 | 3.16 | yes |
| cnt_tower_2 | `two towers` | 5/5 | 0.57-1.00 | 3.23 | yes |
| ex_button_2 | `flat ground, one pipe, then several coins and a stairca` | 5/5 | 0.57-1.00 | 5.86 | yes |
| ex_button_3 | `an underground level: three platforms, then a cannon an` | 2/5 | 0.12-0.77 | 45.45 | NO |
| neg_two_pipes_no_enemies | `two pipes and no enemies` | 5/5 | 0.57-1.00 | 5.85 | yes |
| ord_three | `one pipe, then two enemies, then a cannon` | 5/5 | 0.57-1.00 | 7.67 | yes |

## Counter cross-check: 0 of 50 levels where detector and independent counter disagree


Tower cross-check (ledger AD9): the visible-tower count differs from the caption detector's tower count in 3 of 50 levels (the detector also calls a staircase's narrow top a tower and merges neighbouring pillars).

verification status counts: {'verified': 46, 'not_verified': 4}
