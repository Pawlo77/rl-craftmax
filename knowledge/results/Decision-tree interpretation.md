---
title: "Decision-tree interpretation"
type: concept
domain: interpretation
parent: "[[2026-10-20-craftax-project]]"
tags: [concept, results, interpretation, decision-trees, craftax-1m]
created: 2026-10-28
updated: 2026-10-28
aliases: [Decision tree interpretation, Distillation results]
related-code:
  - "[[src/wut_murl_craftax/interpretation/distill.py]]"
  - "[[src/wut_murl_craftax/interpretation/rules.py]]"
  - "[[src/wut_murl_craftax/interpretation/tree_policy.py]]"
  - "[[src/wut_murl_craftax/interpretation/run.py]]"
---
# Decision-tree interpretation

Every final Craftax-1M policy distilled into CART decision trees of 2 to 4096 leaves on the states its own stochastic policy visits, scored by agreement with its greedy action on held-out episodes, then read as rules. Measures how complex each method's final decision rule is and what it does in plain words. Package: [[interpretation]]. Plan: [[2026-10-20-craftax-interpretation-and-ruleset]] ("What each tree is asked"). Rulings: R106 in [[Experiment decisions]].

## Scope and sources

- 90 policies: 9 methods x 10 tuned seeds (`final-1m`, `combo-1m`), final checkpoint (update 244); 3 fits each: `base` (memory proxies for recurrent policies, current skill for skill methods), `no-memory` (recurrent policies refit without memory proxies; PPO repeats its base fit), `dagger` (one DAgger round on the 256-leaf tree). 0 policies failed.
- Files, all under `outputs/interpretation/1m/`: `complexity.csv` (27 rows, medians over 10 seeds), `policies.csv` (270 rows), `<method>/rules_16.txt` (16-leaf rules of the lower-median-score seed), `<method>/importance.csv`, `<method>/top_level.json` and `top_level_rules.txt` (skill methods), and per policy `<method>/<run>/fit_<variant>.json`, `failure_<variant>.json`, `test.npz`.
- "Derived" marks a number computed here from those files (a Spearman within a method, a share of test states, a pooled confusion); everything else is a column of the named file.
- Fidelity = top-1 agreement with the network's greedy action on the test states (64 held-out seeds 20000-20063, whole episodes). Scores in % of the 226-point maximum; "final score" is the training score of [[Final results]] (`score` in `policies.csv`, macros `\scoreOneM<Method>`).
- Rule marks: "matches mechanic" = the condition and action agree with [[Craftax action effects]], [[Craftax survival stats]], [[Craftax crafting and placing]], [[Craftax farming and food]] or [[Craftax floors and progression]]; "surprising" = the action cannot succeed there or the condition is not what the game asks for.

### Reading caveats

- Every test state is on floor 0: each of the 260 `failure_<variant>.json` files (270 fits; PPO's no-memory fit repeats its base fit and has no file) has one `by_floor` row, floor 0. "Floors with the lowest fidelity" has a single answer at 1M (floor 0 fidelity = overall fidelity).
- So on every test state: no chest, book, potion, enchantment table or XP exists (chests are on floors 1, 3, 4; XP comes from entering floors 1-8), and `ASCEND` cannot work (floor > 0 needed). `READ_BOOK`, `DRINK_POTION_*`, `ENCHANT_*`, `CAST_*` (no spell learned), `LEVEL_UP_*` and `ASCEND` labels are actions that do nothing.
- While asleep the game replaces any action by `NOOP` ([[Craftax action effects]]); the label of an asleep state has no effect. Derived from the `sleeping` column of `test.npz`: median share of asleep test states 0.150 (PPO), 0.180 (PPO-RNN), 0.099 (PQN-RNN), 0.172 (count-bonus PPO-RNN), 0.134 (count-bonus PQN-RNN), 0.182 (SIL), 0.179 (combination), 0.029 (SMDP skills), 0.095 (Dyna-Q). Leaves on `sleeping` spend tree capacity on states where any action is equivalent.
- L80 and L90 are "> 4096" for 267 of 270 fits (89 of 90 policies per fit variant) and L95 for all 270 (R106): the median fidelity peaks at 32 to 256 leaves between 0.398 (Dyna-Q) and 0.514 (SIL) and falls for larger trees (at 4096 leaves 0.299 to 0.453, `<method>/curve.csv`). The low fidelity is genuine, not a label misalignment (A7 review, R106). The exception is SMDP skills seed 8 in all three fits: L80 = L90 = 2 leaves, depth 1, a collapsed policy whose network plays one action (`MAKE_DIAMOND_PICKAXE`) on 93.3% of test states (below).

## Results at a glance

| Method | Final score | Majority | Fid. 16 / 64 / 256 | Gain over majority 16 / 64 / 256 | Val. peak leaves | L90 | Closed loop 256 | Tree / network greedy score |
|---|---|---|---|---|---|---|---|---|
| ppo | 2.17 | 0.420 | 0.488 / 0.494 / 0.491 | 0.100 / 0.112 / 0.108 | 64 | > 4096 | 0.337 | 0.279 / 0.860 |
| ppo-rnn | 2.19 | 0.363 | 0.474 / 0.497 / 0.505 | 0.118 / 0.150 / 0.144 | 128 | > 4096 | 0.335 | 0.463 / 1.534 |
| pqn-rnn | 1.36 | 0.269 | 0.361 / 0.393 / 0.398 | 0.117 / 0.167 / 0.166 | 128 | > 4096 | 0.397 | 0.542 / 1.486 |
| count-bonus-ppo-rnn | 2.27 | 0.333 | 0.447 / 0.479 / 0.490 | 0.112 / 0.136 / 0.151 | 256 | > 4096 | 0.183 | 0.293 / 1.558 |
| count-bonus-pqn-rnn | 1.48 | 0.233 | 0.370 / 0.402 / 0.392 | 0.144 / 0.185 / 0.157 | 64 | > 4096 | 0.444 | 0.705 / 1.638 |
| sil | 2.14 | 0.389 | 0.483 / 0.510 / 0.512 | 0.092 / 0.121 / 0.131 | 128 | > 4096 | 0.250 | 0.345 / 1.413 |
| combo | 2.13 | 0.344 | 0.472 / 0.494 / 0.507 | 0.125 / 0.163 / 0.162 | 128 | > 4096 | 0.237 | 0.317 / 1.441 |
| smdp-skills | 0.83 | 0.271 | 0.419 / 0.454 / 0.428 | 0.116 / 0.146 / 0.143 | 64 | > 4096 (9 of 10) | 0.490 | 0.352 / 0.853 |
| dyna-q | 1.08 | 0.246 | 0.371 / 0.394 / 0.385 | 0.126 / 0.143 / 0.141 | 64 | > 4096 | 0.379 | 0.466 / 1.157 |

- Base fit, medians over 10 seeds; the final score is the mean over seeds. Source: `complexity.csv` (`majority`, `fidelity_*`, `gain_*`, `peak_val_leaves`, `l90_median`, `l90_censored`, `closed_loop_256`); final score from `report/numbers.tex` (`\scoreOneM<Method>`); tree and network greedy scores (% of max, 64 test seeds) derived as medians of `tree_score` and `network_score` in `fit_base.json`.
- Majority = share of the commonest network action on the test states; gain = fidelity minus it (R106).
- Closed loop = the 256-leaf tree played greedily on the 64 test seeds, its score over the network's greedy score; one SMDP seed (8) and one Dyna-Q seed (9) have a network greedy score <= 0, so their ratio is NaN and the median is over 9 seeds.
- Expected fidelity at 256 leaves (mean probability the stochastic policy gives the tree's action): 0.317 to 0.330 for the PPO-based methods, 0.384 to 0.426 for the PQN-based and skill methods (`expected_fidelity_256`).

## Fidelity by action group

Median 256-leaf recall on the test states where the network chose an action of that group (`complexity.csv`, `group_*`, base; blank = the network never chose it).

| Method | noop | move | do | rest | place | craft | ladder | combat | potion | upgrade |
|---|---|---|---|---|---|---|---|---|---|---|
| ppo | | 0.389 | 0.828 | 0.083 | 0.379 | 0.536 | 0.000 | 0.000 | 0.000 | 0.000 |
| ppo-rnn | | 0.516 | 0.740 | 0.605 | 0.311 | 0.185 | | | | 0.000 |
| pqn-rnn | 0.000 | 0.477 | 0.496 | 0.180 | 0.327 | 0.328 | 0.060 | 0.270 | 0.221 | 0.394 |
| count-bonus-ppo-rnn | | 0.535 | 0.734 | 0.401 | 0.415 | 0.440 | | | 0.000 | |
| count-bonus-pqn-rnn | 0.079 | 0.529 | 0.639 | 0.348 | 0.347 | 0.255 | 0.187 | 0.341 | 0.228 | 0.259 |
| sil | 0.000 | 0.485 | 0.734 | 0.523 | 0.376 | 0.202 | | 0.000 | 0.000 | 0.000 |
| combo | | 0.467 | 0.685 | 0.602 | 0.488 | 0.231 | 0.025 | | 0.000 | 0.000 |
| smdp-skills | 0.000 | 0.539 | 0.664 | 0.032 | 0.394 | 0.160 | 0.000 | 0.020 | 0.026 | 0.309 |
| dyna-q | 0.000 | 0.392 | 0.682 | 0.251 | 0.344 | 0.238 | 0.031 | 0.000 | 0.048 | 0.081 |

- `DO` is the best-fitted group everywhere (0.496 to 0.828); rare groups (ladder, combat, potion, upgrade) are near 0 except in the PQN-based methods and SMDP skills upgrade (0.309); in the 16-leaf trees of PQN-RNN seed 4 and count-bonus PQN-RNN seed 2 those rare labels follow the same previous action (below).
- Network move / tree `DO` is the largest off-diagonal confusion cell in all 9 methods; the reverse (network `DO` / tree move) is among the 4 largest in 8 of 9 (derived: confusion pooled over the 10 `failure_base.json` files of each method).

## PPO

Rule readings in the method sections below come from one seed per method, the lower-median-score seed of `rules_16.txt` (named in each section): they are that seed's tree, not a statement about every seed. Importances, fidelities and group recalls are medians over the 10 seeds.

Feed-forward control. Rules: median-score seed 4 (final score 2.171). Majority action `DO` in all 10 seeds (derived, `test.npz`).

- Top-10 features (median permutation importance = drop in top-1 fidelity, `ppo/importance.csv`, base): `water_dist` 0.137, `tree_dist` 0.059, `food` 0.042, `direction` 0.038, `drink` 0.029, `crafting_table_dist` 0.022, `facing_block` 0.011, `water_col_sign` 0.010, `water_row_sign` 0.008, `stone_dist` 0.005.
- 16-leaf tree (`ppo/rules_16.txt`, base; shares of test states):
  - Next to water and no lava below: `DO` (19.1%, purity 0.93) - drinking. Matches mechanic (`DO` facing water gives +1 drink).
  - Next to water with lava below: `RIGHT` (0.4%, purity 0.88). Surprising: in full Craftax lava cannot be entered and does no damage, so no avoidance is needed (craftax 1.6.1 `game_logic_utils.py` L294-334; [[Craftax episode and time]]).
  - Water 2 tiles away and drink >= 4: step toward it (`RIGHT` if water is right, `DOWN` if below, 7.4%), matches mechanic (approach water); or `DO` (3.0%), surprising (`DO` acts on the facing tile only, so it cannot drink from 2 tiles away).
  - Tree adjacent: `DO` (9.6% and 5.5%). Matches mechanic (collect wood).
  - Crafting table adjacent, food <= 7: `MAKE_WOOD_PICKAXE` (3.5% and 1.8%, purity 0.30 and 0.35). Matches mechanic (table among the 8 neighbours); the low purity is consistent with a second pickaxe failing, which the tree cannot see (no pickaxe feature in the rule).
  - Drink <= 3, food <= 7, no water, stone or table adjacent: `DOWN` (14.8%, purity 0.20), or `UP` when the tile above is blocked (2.4%). Surprising: thirsty and not next to water gives a fixed direction, not a move toward water.
  - Food >= 8 and drink >= 9: `DO` whatever is in front (8.0%, purity 0.81 on 7.0%). Surprising: `DO` as the default when nothing is urgent (on grass it yields a sapling with probability 0.1).
- Key actions: `DO` from adjacency to water or a tree and as the sated default; crafting only next to a table; no 16-leaf rule for `SLEEP`, `PLACE_TABLE`, `DESCEND`, spells or potions (256-leaf recall rest 0.083, ladder 0.000, potion 0.000).
- Failures: rest 0.083 and ladder, combat, potion, upgrade 0.000 at 256 leaves; all test states on floor 0.
- DAgger: fidelity at 256 leaves unchanged (0.491), closed loop 0.337 -> 0.481 (`complexity.csv`, dagger).

## PPO-RNN

Recurrent control. Rules: median-score seed 2 (final score 2.181). Majority `DO` in all 10 seeds.

- Top-10 features (`ppo-rnn/importance.csv`, base): `prev_action` [memory proxy] 0.153, `food` 0.052, `water_dist` 0.042, `direction` 0.025, `sleeping` 0.017, `tree_dist` 0.016, `stone_dist` 0.015, `drink` 0.010, `facing_block` 0.008, `water_col_sign` 0.006.
- 16-leaf tree (`ppo-rnn/rules_16.txt`, base): the first split is the previous action, so the tree reads as "what to do after X".
  - After `RIGHT`: `RIGHT` (10.1%, purity 0.52); after `DOWN`: `DOWN` (5.0%); after `LEFT` with water to the left: `LEFT` (2.9%). Surprising (not a game rule): the policy keeps walking in its last direction.
  - After `UP`: `DO` (6.0%). Matches mechanic (the facing tile is the last move direction; the player acts on what it walked toward).
  - After `DO`: `DO` (22.2% over 3 leaves, purity up to 0.91 when food >= 9). Matches mechanic (drinking and mining need repeated `DO`).
  - After `SLEEP`: `SLEEP` (11.5%). Matches mechanic for the asleep states (the action is ignored) and for falling asleep while energy < max.
  - After any other action, water within 2: `DO` (8.7%, purity 0.53). Matches mechanic (drink) only when the water is adjacent; at distance 2 `DO` cannot drink.
  - After any other action, water far, food <= 5: `MAKE_WOOD_SWORD` (15.9%, purity 0.25). Surprising: hungry with no table check gives a craft that fails without a table.
  - After any other action, food >= 6, awake: `SLEEP` (8.2%, purity 0.23). Surprising: `SLEEP` without an energy condition (it fails at full energy).
- Without memory (`no-memory` section): `direction` takes the role of the previous action (food <= 5 and water not adjacent: facing right gives `RIGHT`, 14.9%; facing up gives `DO`, 13.3%), and `SLEEP` fires when food is 6-8 and neither a tree nor water is adjacent (13.5%, purity 0.31).
- Key actions: `DO` after `UP`/`DO` or near water; `SLEEP` after `SLEEP` or when fed; crafting only as `MAKE_WOOD_SWORD`; no rule for `PLACE_TABLE`, `DESCEND`, spells or potions.
- Failures: craft 0.185 and upgrade 0.000 at 256 leaves.

## PQN-RNN

Recurrent value-based control. Rules: median-score seed 4 (final score 1.380). Majority action `DO` in 5 seeds; in seeds 1, 5 and 8 the greedy majority is an action that cannot succeed given what the policy ever has: `CAST_FIREBALL` 0.483, `LEVEL_UP_DEXTERITY` 0.288, `MAKE_IRON_PICKAXE` 0.177 (no spell learned, no XP, iron never collected per [[PQN-RNN training analysis]]). These are the three lowest-scoring seeds (0.303, 0.790, 1.083; derived, `test.npz` and `policies.csv`).

- Top-10 features (`pqn-rnn/importance.csv`, base): `prev_action` [memory proxy] 0.164, `drink` 0.026, `water_dist` 0.025, `direction` 0.023, `food` 0.014, `energy` 0.007, `facing_block` 0.007, `stone_dist` 0.007, `health` 0.006, `light_level` 0.006.
- 16-leaf tree (`pqn-rnn/rules_16.txt`, base): every leaf starts with a set of previous actions; 9 of 16 leaves read nothing else.
  - After `UP`: `UP` (21.2% over 2 leaves); after `NOOP`, `LEFT` or `RIGHT`: `LEFT` (11.4%). Surprising: persistence, not a game rule.
  - After `DOWN`, `DO`, `SLEEP`, a placement or a wood/stone pickaxe: `DO` (45.0% over 5 leaves, purity 0.18 to 0.49). Matches mechanic loosely (`DO` is the main interaction).
  - After `CAST_FIREBALL`: `CAST_FIREBALL` (1.7%); after `MAKE_ARROW`: `MAKE_ARROW` (1.9%); after `DRINK_POTION_CYAN`: `DRINK_POTION_CYAN` (5.1%, purity 0.51). Surprising: repeating actions that cannot succeed for this policy (no spell or potion exists on floor 0; `make arrow` never unlocked).
  - After a yellow potion, book, enchant, torch or level-up action: `LEVEL_UP_INTELLIGENCE` (3.5%, purity 0.18). Surprising (no XP).
  - The episode start: `SLEEP` (0.4%). Surprising (energy starts at its maximum 9, so `SLEEP` fails).
- Without memory: the tree splits on `direction`, `water_dist`, walkability and `facing_block`; `PLACE_TABLE` takes 16.0% (purity 0.16) when the tile above is walkable and no tree is adjacent.
- Key actions: `DO` after most actions; spells and potions only as repeats; `DESCEND` only in the no-memory tree (0.8%, when lava is in view to the left: surprising, the ladder needs to be under the player).
- Failures: noop 0.000 and ladder 0.060 at 256 leaves.

## Count-bonus PPO-RNN

[[Count-Based Progress Bonus design]] on PPO-RNN. Rules: median-score seed 0 (final score 2.274). Majority `DO` in all 10 seeds.

- Top-10 features (`count-bonus-ppo-rnn/importance.csv`, base): `prev_action` [memory proxy] 0.148, `water_dist` 0.054, `food` 0.050, `sleeping` 0.020, `tree_dist` 0.016, `direction` 0.013, `water_col_sign` 0.010, `stone_dist` 0.008, `facing_block` 0.008, `drink` 0.007.
- 16-leaf tree (`count-bonus-ppo-rnn/rules_16.txt`, base): the first split is `food` (<= 4, 5-8, >= 9).
  - Food <= 4 and next to water: `DO` (6.6%, purity 0.54). Matches mechanic (drink).
  - Food <= 4 and tree adjacent: `DO` (8.7%). Matches mechanic (wood), though food is low and no rule hunts cows.
  - Food <= 4, no tree, stone within 4: `LEFT` (10.5%, purity 0.22); no stone, water to the left: `LEFT`, else `UP` (12.4%). Surprising for the fixed `LEFT`; matches mechanic for "toward water".
  - Food 5-8 after a craft or placement: `PLACE_TABLE` (12.2%, purity 0.37). Surprising: no wood condition, so it fires without the 2 wood a table needs, and repeated tables give no further achievement.
  - Food 5-8 after `DO`, `SLEEP` or `DOWN`: the same action again (12.2%). Surprising: persistence.
  - Asleep: `SLEEP` (9.1%). Matches mechanic (any action is ignored).
  - Food >= 9 facing grass, water, stone, tree, wood or path: `DO` (8.7%, purity 0.69). Matches mechanic (`DO` acts on the facing tile).
- Without memory: 6 of 16 leaves keep the same conditions; `PLACE_TABLE` now fires when food is 5-8 and no tree, water or table is near (16.2%): a sensible place, but surprising for the same reason as the base leaf (no wood condition).
- Key actions: `DO` near water, trees or with full food; `PLACE_TABLE` as the habit after crafting; `SLEEP` only while asleep or after `SLEEP`; no `DESCEND`, spell or potion rule.
- Failures: potion 0.000 and rest 0.401 at 256 leaves; the lowest closed loop of all methods (0.183).

## Count-bonus PQN-RNN

Count bonus on PQN-RNN. Rules: median-score seed 2 (final score 1.603). Majority `DO` in all 10 seeds.

- Top-10 features (`count-bonus-pqn-rnn/importance.csv`, base): `prev_action` [memory proxy] 0.186, `drink` 0.025, `food` 0.015, `water_dist` 0.014, `direction` 0.013, `energy` 0.012, `facing_block` 0.011, `health` 0.007, `sleeping` 0.006, `stone_dist` 0.006.
- 16-leaf tree (`count-bonus-pqn-rnn/rules_16.txt`, base): food (<= 7 or >= 8), then the previous action in every leaf.
  - Food <= 7 after any move or `NOOP`: `DOWN` (19.8% over 2 leaves). Surprising: persistence.
  - After `DO`/`SLEEP`, energy <= 8: `DO`; energy >= 9: `SLEEP` (10.5%, purity 0.27). Surprising: `SLEEP` at full energy fails.
  - After `REST`: `REST` (7.0%, purity 0.69). Matches mechanic (rest continues until health is full; actions are ignored meanwhile).
  - After `DRINK_POTION_GREEN`: `DRINK_POTION_GREEN` (4.6%, purity 0.68); after `ENCHANT_BOW`: `ENCHANT_BOW` (2.3%); after a ladder, spell or high-tier craft: `MAKE_ARROW` (12.0%, purity 0.14). Surprising: repeats of actions that cannot succeed for this policy (no potion or enchantment table on floor 0; `make arrow` never unlocked per [[Count-bonus PQN-RNN training analysis]]).
  - Food >= 8 at the episode start: `CAST_ICEBALL` (0.4%, purity 1.00). Surprising (no spell is known at the start).
  - Food >= 8 after a non-move action: `DO` (16.6%, purity up to 0.77). Matches mechanic loosely.
- Key actions: `SLEEP` at full energy, `REST` as a repeat, potions and enchanting as repeats; no `PLACE_TABLE` rule; crafting only as repeats (`MAKE_WOOD_SWORD` after a sword craft 2.2%, `MAKE_ARROW` 12.0%, `MAKE_TORCH` 4.9%).
- Failures: noop 0.079, ladder 0.187 at 256 leaves; the highest gain at 64 leaves (0.185).

## SIL

[[Self-Imitation Learning design]] on PPO-RNN. Rules: median-score seed 6 (final score 2.135). Majority `DO` in all 10 seeds.

- Top-10 features (`sil/importance.csv`, base): `prev_action` [memory proxy] 0.131, `water_dist` 0.063, `food` 0.040, `sleeping` 0.034, `direction` 0.025, `tree_dist` 0.016, `stone_dist` 0.015, `drink` 0.012, `facing_block` 0.009, `water_col_sign` 0.008.
- 16-leaf tree (`sil/rules_16.txt`, base): the first split is `sleeping`.
  - Asleep: `DO` (18.5% over 2 leaves). Surprising as a label, harmless in play (the action is replaced by `NOOP`); it is the largest asleep share among the 16-leaf trees.
  - Awake after `DO`: `DO` (11.4%, purity 0.60); after `RIGHT`, `UP` or `DOWN`: `DO` (10.7%). Matches mechanic (act on the tile just walked toward).
  - Awake after `NOOP`/`LEFT`: `LEFT` (4.3%); facing down after a move: `DOWN` (5.6%). Surprising: persistence.
  - Drink >= 9 and facing up: `DO` (11.2%, purity 0.82 on 8.1%). Matches mechanic (the player starts facing up; `DO` on the tile ahead).
  - Drink 4-8 after a craft or placement: `UP` (8.2%). Surprising: no target in the condition.
- Without memory: water within 3 gives `DO` (12.5%), stone within 4 gives `DO` (18.5%), else `UP` (16.6%) - approach to water and stone and a fixed exploration direction (surprising for `DO` at distance 2-4: it acts on the facing tile only).
- Key actions: almost only `DO` and moves; `SLEEP` only after `SLEEP`; no crafting, `PLACE_TABLE`, `DESCEND`, spell or potion rule.
- Failures: craft 0.202, combat/potion/upgrade 0.000 at 256 leaves.

## Combination (count bonus + SIL)

Combo of [[Count-Based Progress Bonus design]] and [[Self-Imitation Learning design]] on PPO-RNN. Rules: median-score seed 2 (final score 2.106). Majority `DO` in all 10 seeds.

- Top-10 features (`combo/importance.csv`, base): `prev_action` [memory proxy] 0.138, `food` 0.054, `water_dist` 0.051, `sleeping` 0.041, `direction` 0.024, `tree_dist` 0.015, `facing_block` 0.009, `stone_dist` 0.008, `water_col_sign` 0.008, `water_row_sign` 0.007.
- 16-leaf tree (`combo/rules_16.txt`, base): the first split is `food` (<= 4, 5-8, >= 9).
  - Food <= 4, water within 3: `DO`, or `DOWN` when facing down (14.6%). Matches mechanic (drink) only when the water is adjacent; otherwise an approach at best.
  - Food <= 4, water far, stone within 4: `DO` (13.8%, purity 0.41). Matches mechanic (mining) only when the stone is the facing tile; at distance 2-4 `DO` does nothing to it.
  - Food <= 4, water and stone far, awake: `MAKE_WOOD_PICKAXE` (16.2%, purity 0.28). Surprising: no table condition, and a second pickaxe fails.
  - Food 5-8, water far, after `DO`, `SLEEP`, a placement or a craft: `PLACE_TABLE` (19.1% over 3 leaves). Surprising: repeated tables, as in count-bonus PPO-RNN.
  - Asleep with food 5-8: `SLEEP` in daylight, `DO` when darker (9.0%); asleep with food <= 4: `DO` (3.4%). Matches mechanic only in that the action is ignored.
  - Food >= 9: `DO` (10.0%, purity up to 0.78). Matches mechanic loosely.
- Key actions: the count-bonus `PLACE_TABLE` habit and an unconditioned pickaxe craft; `DO` for drink and stone; no `DESCEND`, spell or potion rule.
- Failures: craft 0.231, ladder 0.025, potion and upgrade 0.000 at 256 leaves.

## SMDP skills

[[SMDP Q-Learning over Hand-Defined Skills design]]; labels are the primitive action of the full policy (top level, then the skill network). Rules: median-score seed 7 (final score 0.625), whose greedy label is `READ_BOOK` on 0.658 of test states; `READ_BOOK` is the majority in this seed only, so the rule bullets below describe one collapsed seed, not the method. In 5 of 10 seeds the greedy majority action cannot succeed given what the policy ever has: seed 8 `MAKE_DIAMOND_PICKAXE` 0.933, seed 7 `READ_BOOK` 0.658, seed 9 `PLACE_STONE` 0.437, seed 3 `PLACE_TORCH` 0.425, seed 5 `MAKE_IRON_PICKAXE` 0.268 (no book on floor 0; `place stone`, `place torch`, `collect iron`, `collect diamond` never unlocked per [[SMDP skills training analysis]]). They are exactly the 5 lowest-scoring seeds (-0.327 to 0.625; the others 1.195 to 1.568). Derived, `test.npz` and `policies.csv`.

- Top-10 features (`smdp-skills/importance.csv`, base): `prev_action` [memory proxy] 0.128, `food` 0.020, `drink` 0.010, `skill_unlock_any` [current skill] 0.009, `facing_block` 0.008, `direction` 0.008, `water_dist` 0.006, `tree_dist` 0.004, `stone_dist` 0.004, `sapling` 0.003.
- 16-leaf tree (`smdp-skills/rules_16.txt`, base):
  - Drink <= 6 after `READ_BOOK`: `READ_BOOK` (61.7% over 3 leaves, purity 0.74 to 0.99). Surprising: books exist only in chests on floors 3 and 4.
  - Drink <= 6 after other actions: `READ_BOOK` (12.7% over 3 leaves). Surprising, same reason.
  - Drink >= 7 after `DO`, no sapling, bright: `PLACE_PLANT` (5.5%, purity 0.53 and 0.75). Surprising: placing a plant needs a sapling, and the rule fires with 0.
  - After `PLACE_PLANT`: `DO` (5.3%, purity 0.77). Surprising: `DO` on an unripe plant does nothing.
  - A sapling in hand right after an unlock (`steps_since_unlock` 0): `PLACE_PLANT` (0.4%). Matches mechanic (collect a sapling, plant it).
  - Drink >= 7 with saplings, later steps: `DO` (3.5%). Matches mechanic loosely.
- Without memory: with drink <= 6, 8 of 9 leaves give `READ_BOOK` (74.4%) whatever the current skill; with drink >= 7 the skill `unlock_any` gives `DO` (18.8% over 4 leaves).
- Key actions: `READ_BOOK` dominates; `PLACE_PLANT` and `DO` around saplings; no `SLEEP` rule except one leaf (0.4%); rest recall 0.032.
- Failures: rest 0.032, ladder 0.000, combat 0.020, potion 0.026 at 256 leaves.

### Top level

Each visited key of the final greedy table is one top-level rule (`smdp-skills/top_level.json`, `top_level_rules.txt`; tried view main, R106 and the [[interpretation]] note).

- Rules (visited keys): median 127.5 over 10 seeds, from 47 to 209; distinct greedy skills median 11; key-tree L90 median 128 (`complexity.csv`, `top_keys`, `top_skills`, `top_l90`).
- Optimistic share 0.984 (median): the default argmax lands on an untried entry for 96-100% of keys in every seed, so the tried view is the one that shows learned choices.
- More visited keys, higher score: derived Spearman 0.85 (p 0.002) between `n_rules` and the final score over the 10 seeds.
- Median-score seed (7): 67 keys, 10 greedy skills (fight 13 keys, descend 10, unlock_any 9, eat 7, upgrade_sword 7, collect_wood 6, drink 6, make_wood_pickaxe 6, sleep 2, place_table 1). The key-to-skill tree reaches 1.0 at 58 leaves (L90 64, depth 8; 16 leaves give 0.522): a lookup table, one leaf per 1.2 keys. Across seeds the same holds: median key-tree L90 128 for a median of 127.5 keys.
- The tree's first splits are `low_drink`, `low_food`, `low_health`, then `wood` and `sapling`; the chosen skill rarely matches the need: low drink and low energy with wood 0 gives `descend` (3.0% of keys), low drink with wood 2, food and health fine gives `upgrade_sword` (3.0%). Surprising: the low-stat flags mostly do not select `drink`, `eat` or `sleep` (some `low_food` leaves give `eat`, two `low_energy` leaves `sleep`).
- Default view (untried entries win): 3 skills (`collect_stone` 42 keys, `collect_wood` 24, `unlock_any` 1), L90 8; a tie broken by skill order, not a learned choice.
- 10 highest-value rules of the median-score seed (tried view), all on floor 0 with the ladder open; values 9.95 to 10.04, within 0.05 of the optimistic start value 10 (across the 10 seeds the top-10 values span 9.89 to 10.11, `top_level.json`):
  - `unlock_any` with 2 wood (10.04).
  - `fight` with 2 wood when food, drink and health are low (10.00).
  - `unlock_any` with 2 wood, a sapling, low health (9.99).
  - `drink` when food is low (9.99).
  - `descend` with a sapling, low drink and low energy (9.99).
  - `descend` with 2 wood, a sapling, low drink and low health (9.98).
  - `make_wood_pickaxe` with 2 wood, low health (9.98).
  - `make_wood_pickaxe` with 1 wood, a sapling, low drink and low health (9.98).
  - `upgrade_sword` with 2 wood, a sapling, low drink (9.96).
  - `descend` with 2 wood, a sapling, low food and low drink (9.95).

## Dyna-Q

[[Dyna-Q over a Skill-Level Progress Model design]]; same labels as SMDP skills. Rules: median-score seed 5 (final score 1.046), whose majority label is `SLEEP` (0.206; in this seed only). In 3 of 10 seeds the greedy majority cannot succeed given what the policy ever has: seed 9 `MAKE_IRON_ARMOUR` 0.747, seed 3 `ASCEND` 0.391, seed 7 `MAKE_DIAMOND_PICKAXE` 0.249 (no iron or diamond collected per [[Dyna-Q training analysis]]; no up ladder on floor 0); final scores -0.043, 0.171 and 0.933, the lowest, second and fourth lowest (derived, `test.npz`, `policies.csv`).

- Top-10 features (`dyna-q/importance.csv`, base): `prev_action` [memory proxy] 0.117, `skill_unlock_any` [current skill] 0.023, `direction` 0.015, `food` 0.012, `water_dist` 0.011, `drink` 0.010, `stone_dist` 0.008, `energy` 0.008, `tree_dist` 0.006, `skill_collect_wood` [current skill] 0.004.
- 16-leaf tree (`dyna-q/rules_16.txt`, base): food (<= 6 or >= 7), then `sleeping`, then energy or the current skill.
  - Food <= 6, awake, energy >= 9: `SLEEP` (20.5%, purity 0.25). Surprising: `SLEEP` needs energy below the maximum (9 at dexterity 1).
  - Asleep: `SLEEP` (16.2% over 2 leaves). Matches mechanic only in that the action is ignored.
  - Food <= 6, drink <= 4, stone within 2, iron far: `MAKE_STONE_SWORD` (9.8%, purity 0.22). Surprising: no table condition; `make stone sword` is never unlocked.
  - Food <= 6, drink <= 4, water within 3: `ENCHANT_SWORD` (6.1%, purity 0.24). Surprising: needs an enchantment table (floors 3 and 4), a gem and 9 mana.
  - Food >= 7 while the skill `unlock_any` runs: `DO` (17.0% over 4 leaves, purity up to 0.89). Partly matches mechanic: of the `unlock_any` targets only collecting a sapling and eating a plant use `DO` ([[Craftax skill library]]); placements and crafts do not.
  - Food >= 7, another skill, after any action but a move, `NOOP` or `DO`: `SLEEP` (5.0%). Surprising: no energy condition.
  - Food <= 6, energy <= 8, drink >= 5: `LEFT` (8.8%). Surprising: a fixed direction.
- Without memory: the tree is nearly the same (the previous-action leaves become `drink` splits); 8 of 16 leaves keep their conditions.
- Key actions: `SLEEP` by energy (inverted), `DO` inside `unlock_any`, crafting and enchanting without their stations; no `PLACE_TABLE` or `DESCEND` rule.
- Failures: noop 0.000, combat 0.000, ladder 0.031 at 256 leaves.

### Top level

- Rules (visited keys): median 129.5 over 10 seeds, from 66 to 355; distinct greedy skills median 11; key-tree L90 median 128; optimistic share 0.989 (`complexity.csv`, `top_level.json`).
- More visited keys, higher score: derived Spearman 0.91 (p < 0.001) between `n_rules` and the final score.
- Median-score seed (5): 112 keys, 11 greedy skills (unlock_any 19, descend 18, fight 15, sleep 13, make_wood_pickaxe 11, collect_wood 9, upgrade_sword 8, drink 7, place_table 7, eat 4, level_up 1). The key tree reaches 1.0 at 98 leaves (L90 128, depth 8; 16 leaves give 0.473): a lookup table, as in SMDP skills.
- Its table holds a key on floor 1 (the top-valued `level_up`, 10.07), while no test or final stochastic eval episode leaves floor 0 ([[Dyna-Q training analysis]]).
- Default view: 3 skills (`collect_stone` 75 keys, `collect_wood` 36, `level_up` 1), L90 16; tie-broken, not learned.
- 10 highest-value rules of the median-score seed (tried view), values 9.92 to 10.07 (across the 10 seeds the top-10 values span 9.05 to 10.22, `top_level.json`):
  - `level_up` on floor 1 with 1 wood and low drink (10.07).
  - `upgrade_sword` with 4 wood, low drink and low health (9.98).
  - `drink` with 1 wood, a sapling, low food and low energy (9.98).
  - `upgrade_sword` with 4 wood and low drink (9.98).
  - `sleep` with 3 wood and low drink (9.97).
  - `place_table` with 3 wood, a sapling and low health (9.96).
  - `upgrade_sword` with 2 wood, a sapling, low drink and low energy (9.96).
  - `sleep` with 4 wood, a sapling and low food (9.96).
  - `make_wood_pickaxe` with 4 wood, a sapling and low health (9.95).
  - `fight` with 3 wood, a sapling, low food, drink and energy (9.92).
- Rules with wood >= 2 that choose `place_table` or `make_wood_pickaxe` match mechanic (a table costs 2 wood, a pickaxe 1); `sleep` with low drink and `upgrade_sword` with low drink and health are surprising.

## Cross-method

### Complexity

- No method's median policy reaches 80% agreement with its network at any size up to 4096 leaves: the L80, L90 and L95 medians are "> 4096" for every method and fit (`complexity.csv`, `l80_median` to `l95_median` = inf; `l90_censored` 10 of 10, SMDP skills 9 of 10). The one policy that does is SMDP skills seed 8 (L90 = 2 leaves), a collapsed one-action policy. The reason is a plateau, not a slow climb: median fidelity peaks at 0.398 (Dyna-Q) to 0.514 (SIL) at 32-256 leaves, near the size that maximises validation fidelity (`peak_val_leaves` 64 to 256), then falls by 0.06 to 0.11 up to 4096 leaves (`<method>/curve.csv`).
- The median tree of every method beats the majority-class baseline: gain at 256 leaves from 0.108 (PPO) to 0.166 (PQN-RNN). Per policy it does not always: 11 of 270 fits at 256 leaves have gain <= 0, among them the two collapsed seeds SMDP skills 8 (0.918 vs majority 0.933) and Dyna-Q 9 (0.685 vs 0.747). The high-majority PPO-based policies have the higher raw fidelity; the gain is similar across methods (0.092 to 0.185 at 16 and 64 leaves).
- Figure: `outputs/results/figures/1m/fidelity_vs_leaves.png` (median base curve per method, the 0.9 level as a solid line; colours in `legend_methods`). Interpretation figures F27 `outputs/results/figures/1m/distil_score.png` (score vs 256-leaf fidelity per policy) and F28 `outputs/results/figures/1m/distil_features.png` (importance by feature) are written by the comparison's figure stage; captions in [[Figures 1M]].

### Is a higher-scoring policy more complex?

- Score vs fidelity at 256 leaves, Spearman over the 90 policies (R106): base 0.308 (p 0.003), no-memory 0.389 (p < 0.001), DAgger 0.288 (p 0.006) (`complexity.csv`, `spearman_score_fid256`, `spearman_fid256_p`). Between methods, higher-scoring policies are easier to imitate, not harder.
- The positive sign is accompanied by the majority share: derived from `policies.csv` (base), score vs majority share 0.277 (p 0.008) and score vs gain over majority -0.027 (p 0.80).
- Within a method the sign is zero or negative (derived, base; descriptive, n = 10 per method): SMDP skills -0.77 (p 0.009), Dyna-Q -0.81 (p 0.005); the other seven between -0.42 and 0.05 (p >= 0.23). In the skill methods the most imitable seeds are the collapsed ones (one infeasible action on most states).
- Score vs L90 (`spearman_score_l90`) is not reported as a result: 89 of 90 base policies have a censored L90, so the statistic rests on one uncensored seed (SMDP seed 8) ranked against ties; R106 replaces it. The same holds for `depth_at_l90_median` = 1 for SMDP skills.

### Memory: what recurrent methods lose without memory proxies

| Method | Fid. 256 base / no-memory | Gain 256 base / no-memory | Closed loop base / no-memory | `prev_action` importance |
|---|---|---|---|---|
| ppo-rnn | 0.505 / 0.452 | 0.144 / 0.098 | 0.335 / 0.384 | 0.153 |
| pqn-rnn | 0.398 / 0.327 | 0.166 / 0.053 | 0.397 / 0.188 | 0.164 |
| count-bonus-ppo-rnn | 0.490 / 0.441 | 0.151 / 0.097 | 0.183 / 0.229 | 0.148 |
| count-bonus-pqn-rnn | 0.392 / 0.322 | 0.157 / 0.073 | 0.444 / 0.214 | 0.186 |
| sil | 0.512 / 0.476 | 0.131 / 0.071 | 0.250 / 0.286 | 0.131 |
| combo | 0.507 / 0.451 | 0.162 / 0.121 | 0.237 / 0.299 | 0.138 |
| smdp-skills | 0.428 / 0.361 | 0.143 / 0.073 | 0.490 / 0.327 | 0.128 |
| dyna-q | 0.385 / 0.316 | 0.141 / 0.067 | 0.379 / 0.198 | 0.117 |

- Sources: `complexity.csv` (base and no-memory rows), `<method>/importance.csv` (median, base).
- Without memory proxies fidelity drops by 0.036 (SIL) to 0.071 (PQN-RNN), and the gain over majority roughly halves in the PQN-based and skill methods.
- `prev_action` is the top feature of every recurrent policy (0.117 to 0.186); the next feature is at most 0.063 (SIL `water_dist`). A tree needs the previous action (a memory proxy) to imitate these policies; "repeat the previous action" is how the trees read, not a measurement of what the network conditions on.
- What the proxy recovers differs by learner: for the PQN-based methods it recovers the rare groups, which are repeats (PQN-RNN combat 0.270 vs 0.002, upgrade 0.394 vs 0.073, potion 0.221 vs 0.029); for the PPO-based ones mainly rest (PPO-RNN 0.605 vs 0.498: `SLEEP` after `SLEEP`).
- Closed loop splits the same way: the memory proxy helps the tree play for the PQN-based and skill methods (PQN-RNN 0.397 vs 0.188) but hurts for the four PPO-based ones (PPO-RNN 0.335 vs 0.384). A plausible, untested reading: a tree that copies its own previous action locks into repeats the network breaks out of.

### DAgger

- One DAgger round lowers fidelity on the network's own states in 8 of 9 methods (PPO unchanged at 0.491) and raises the closed-loop ratio in all 9: PPO 0.337 -> 0.481, count-bonus PPO-RNN 0.183 -> 0.329, SIL 0.250 -> 0.404, SMDP skills 0.490 -> 0.517 (`complexity.csv`, dagger rows). The validation peak moves to larger trees in 8 of 9 methods (128 to 512 leaves; count-bonus PPO-RNN stays at 256).

## Report draft (section 6, about 0.4 page)

Draft for [[2026-10-20-craftax-report]] (Phase 5 copies it into `report/`). Macros are those of `report/numbers.tex` written by the 2026-10-28 comparison run; the figure is the PNG `figures/1m/fidelity_vs_leaves.png`.

```latex
\section{Interpretation}
\label{sec:interpretation}
We distil every final 1M policy (9 methods $\times$ 10 seeds) into CART trees of
2 to 4096 leaves fitted on its own stochastic rollouts and measure top-1 agreement
with its greedy action on 64 held-out episodes (Fig.~\ref{fig:fidelity}).
No method's median policy reaches 90\% agreement: the median L90 is \lNinetyOneMPpo{} leaves
for every method (the one exception is a collapsed SMDP seed playing one action on 93\% of states).
Fidelity peaks at 32--256 leaves (validation peak \peakValLeavesOneMPpo{} for PPO,
\peakValLeavesOneMCountBonusPpoRnn{} for count-bonus PPO-RNN), reaches
\fidTwoFiftySixOneMDynaQ{} (Dyna-Q) to \fidTwoFiftySixOneMSil{} (SIL) at 256 leaves and falls for
larger trees; the median tree of every method beats the majority action by \gainTwoFiftySixOneMPpo{}
(PPO) to \gainTwoFiftySixOneMPqnRnn{} (PQN-RNN). Between methods, higher-scoring policies are easier,
not harder, to imitate (Spearman \spearmanScoreFidTwoFiftySixOneM{} between final
score and fidelity at 256 leaves), together with a larger majority share
(\majorityOneMPpo{} for PPO vs \majorityOneMCountBonusPqnRnn{} for count-bonus PQN-RNN).
Played in closed loop, a 256-leaf tree keeps \closedLoopOneMCountBonusPpoRnn{} to
\closedLoopOneMSmdpSkills{} of its network's greedy score.

The 16-leaf trees of each method's median-score seed read as follows. PPO drinks when next to water, chops adjacent trees,
crafts a pickaxe next to a table and otherwise defaults to \texttt{DO}.
For PPO-RNN and SIL the trees split first on the previous action (a memory proxy) and mostly
repeat it, \texttt{DO} after walking toward a target. The PQN-RNN and count-bonus PQN-RNN trees repeat the previous action even when it
cannot succeed (spells, potions). Count-bonus PPO-RNN and the combination add a habit of
placing a crafting table after any craft. In SMDP skills and Dyna-Q, seeds differ more than methods: the five lowest-scoring SMDP
seeds have a greedy majority action that cannot succeed with what they ever hold
(e.g.\ \texttt{READ\_BOOK} without books), while the others act through \texttt{DO}; in Dyna-Q
the same holds for its lowest, second and fourth lowest seeds. The top levels of both
skill methods are lookup tables (key-tree L90 of the order of the number of visited keys),
and their highest values stay near the optimistic start.

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/1m/fidelity_vs_leaves.png}
  \caption{Test top-1 fidelity of the distilled trees vs leaf count (median over
  10 seeds per method; Craftax-1M). The horizontal line is the L90 level.}
  \label{fig:fidelity}
\end{figure}
```

- Numbers in the draft without a macro are left out (the 93% and the seed counts above are from `policies.csv` and `test.npz`, no macro yet): the Spearman p (0.003), the within-method Spearmans and the top-level key counts (`top_keys`, `top_l90` have no macro); the PQN-RNN, SMDP and Dyna-Q statements are qualitative.
