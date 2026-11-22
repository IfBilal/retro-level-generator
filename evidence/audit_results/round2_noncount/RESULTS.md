# Adherence audit `round2_noncount`

device **cuda**, mode **fast**, profile **local**, width 48, seeds [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009], git a1a0dbc, UTC 2026-11-21T18:19:14Z

**Overall: 278/290 levels pass (95.9%)** (adversarial rows included)

## By category

| category | pass / n | rate | 95% CI | target |
|---|---|---|---|---|
| example | 30/40 | 0.75 | 0.60-0.86 | 1.00 |
| negation | 40/40 | 1.00 | 0.91-1.00 | 1.00 |
| ordering | 29/30 | 0.97 | 0.83-0.99 | 1.00 |
| phrasing | 80/80 | 1.00 | 0.95-1.00 | 0.90 |
| synonym | 80/80 | 1.00 | 0.95-1.00 | 0.90 |
| two_scene_counts | 19/20 | 0.95 | 0.76-0.99 | 0.90 |

## Per prompt

| id | prompt | pass/n | CI | median s | meets target |
|---|---|---|---|---|---|
| ex_button_1 | `full floor with two pipes and a few enemies` | 10/10 | 0.72-1.00 | 14.465 | yes |
| ex_button_2 | `flat ground, one pipe, then several coins and a stairca` | 10/10 | 0.72-1.00 | 16.17 | yes |
| ex_button_3 | `an underground level: three platforms, then a cannon an` | 1/10 | 0.02-0.40 | 45.42 | NO |
| ex_button_4 | `mostly pits with two islands of ground, then a coin lin` | 9/10 | 0.60-0.98 | 39.445 | NO |
| ph_only_one | `only one pipe` | 10/10 | 0.72-1.00 | 5.375 | yes |
| ph_single | `a single pipe` | 10/10 | 0.72-1.00 | 4.335 | yes |
| ph_exactly_two | `exactly two pipes` | 10/10 | 0.72-1.00 | 4.855 | yes |
| ph_just_2 | `just 2 pipes` | 10/10 | 0.72-1.00 | 4.869999999999999 | yes |
| ph_digit | `1 pipe` | 10/10 | 0.72-1.00 | 4.235 | yes |
| ph_case | `one PIPE` | 10/10 | 0.72-1.00 | 4.425000000000001 | yes |
| ph_spaces | `  one    pipe  ` | 10/10 | 0.72-1.00 | 4.1850000000000005 | yes |
| ph_punct | `One pipe!!!` | 10/10 | 0.72-1.00 | 4.205 | yes |
| neg_no_pipes | `no pipes` | 10/10 | 0.72-1.00 | 4.2 | yes |
| neg_without_pipes | `a flat level without pipes` | 10/10 | 0.72-1.00 | 4.27 | yes |
| neg_two_pipes_no_enemies | `two pipes and no enemies` | 10/10 | 0.72-1.00 | 7.91 | yes |
| neg_coins_no_enemies | `several coins and no enemies` | 10/10 | 0.72-1.00 | 16.095 | yes |
| ord_three | `one pipe, then two enemies, then a cannon` | 10/10 | 0.72-1.00 | 9.0 | yes |
| ord_stair_tower | `a staircase, then a tower` | 9/10 | 0.60-0.98 | 6.865 | NO |
| ord_coin_pipe | `several coins, then one pipe` | 10/10 | 0.72-1.00 | 13.695 | yes |
| two_scene_same | `one pipe, then two pipes` | 9/10 | 0.60-0.98 | 4.345 | yes |
| two_scene_same_enemy | `one enemy, then three enemies` | 10/10 | 0.72-1.00 | 10.415 | yes |
| syn_tube | `one tube` | 10/10 | 0.72-1.00 | 4.275 | yes |
| syn_warp | `one warp pipe` | 10/10 | 0.72-1.00 | 4.425 | yes |
| syn_chimney | `a chimney` | 10/10 | 0.72-1.00 | 4.195 | yes |
| syn_tubes2 | `two tubes` | 10/10 | 0.72-1.00 | 4.4 | yes |
| syn_goomba | `one goomba` | 10/10 | 0.72-1.00 | 6.355 | yes |
| syn_bullet | `a bullet bill` | 10/10 | 0.72-1.00 | 4.455 | yes |
| syn_ledges | `two ledges` | 10/10 | 0.72-1.00 | 4.615 | yes |
| syn_stairway | `a stairway` | 10/10 | 0.72-1.00 | 4.285 | yes |

## Counter cross-check: 1 of 290 levels where detector and independent counter disagree

- ord_coin_pipe seed 1006: {'pipe': [1, 0]} (detector, independent)

verification status counts: {'verified': 277, 'not_verified': 13}
