# Adherence audit `_dev`

device **cuda**, mode **fast**, profile **local**, width 48, seeds [1000, 1001, 1002, 1003], git d0c2505, UTC 2026-11-21T16:02:04Z

**Overall: 16/16 levels pass (100.0%)** (adversarial rows included)

## By category

| category | pass / n | rate | 95% CI | target |
|---|---|---|---|---|
| count | 12/12 | 1.00 | 0.76-1.00 | 0.90 |
| example | 4/4 | 1.00 | 0.51-1.00 | 1.00 |

## Per prompt

| id | prompt | pass/n | CI | median s | meets target |
|---|---|---|---|---|---|
| cnt_pipe_1 | `one pipe` | 4/4 | 0.51-1.00 | 5.300000000000001 | yes |
| cnt_pipe_2 | `two pipes` | 4/4 | 0.51-1.00 | 5.625 | yes |
| cnt_enemy_2 | `two enemies` | 4/4 | 0.51-1.00 | 11.165 | yes |
| ex_button_2 | `flat ground, one pipe, then several coins and a stairca` | 4/4 | 0.51-1.00 | 17.79 | yes |

## Counter cross-check: 0 of 16 levels where detector and independent counter disagree


verification status counts: {'verified': 16}
