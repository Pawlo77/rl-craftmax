---
title: "Dyna-Q training analysis"
type: concept
domain: analysis
parent: "[[2026-10-20-craftax-project]]"
tags: [concept, results, analysis, training, craftax-1m]
created: 2026-10-28
related-code:
  - "[[src/wut_murl_craftax/analysis/hard.py]]"
  - "[[src/wut_murl_craftax/analysis/cases.py]]"
  - "[[src/wut_murl_craftax/analysis/obstacles.py]]"
  - "[[src/wut_murl_craftax/analysis/training.py]]"
---
# Dyna-Q training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 1.077 (sd 0.619), stochastic eval score 1.081; only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes.
- Achievements: 50 of 67 never unlocked by any seed at any checkpoint (basic 9, intermediate 17, advanced 15, very advanced 9), 9 rare (final rate < 5%), 8 common.
- Deaths at the final checkpoint: zombie 52.5%, starvation 31.6%, arrow 14.5%, dehydration 1.4% (the untrained policy: zombie 51.6%, starvation 28.1%, arrow 20.3%).
- Stalls: 81 stalls in 640 final eval episodes, 0.127 per episode (seed interval 0.086 to 0.172), median length 328 steps, 1.2% escaped by an unlock; they are started by wake up (27%) and episode start (25%).
- Obstacles at 1M: instability 9/10, collapse 5/10, reward hacking 7/10, skill forgetting 10/10, skill interference 10/10, deep-skill starvation 10/10, survival failure despite flags 10/10, planned-value gap 10/10, stale model 10/10, bonus-driven re-trying 2/10; not logged (R78): value failure.

## Scope

- Craftax-1M results only: `final-1m`, Dyna-Q (Dyna-Q over the skill-level progress model, on the PQN-RNN control), 10 seeds (0-9) of the tuned config; method: [[Dyna-Q over a Skill-Level Progress Model design]], plan [[2026-10-20-craftax-method-dyna-q]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/dyna-q/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-dyna-q`, id `462656641a0e4e61b2cbd4e4237fc82e`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 33 case episodes, all replayed from the stored eval files.
- Related analyses: [[SMDP skills training analysis]] (same skill library and tuned knobs).
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- 1B: pending (final-1b not run, R75); sections marked 1B below are placeholders for the same tables at 1B.
- Metric families the 26 reused runs never logged read "not logged" in the detector tables (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 248, detected 162, not applicable 10, not logged 20; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes. Floor 0 reach rate 100.0% (10 of 10 seeds). Source: `training-analysis-1m-dyna-q`, `hard/floor_reach.csv`, plot `plots/floor_reach.png`.
- Never unlocked (50 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (9): eat plant, make stone pickaxe, make stone sword, collect iron, collect diamond, make iron pickaxe, make iron sword, make torch, place torch; intermediate (17): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-dyna-q`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 9): collect stone 1.88% (5/10 seeds ever), defeat zombie 1.56% (10/10 seeds ever), make wood sword 0.78% (10/10 seeds ever), place furnace 0.31% (3/10 seeds ever), collect coal 0.16% (2/10 seeds ever), place stone 0.00% (2/10 seeds ever), defeat skeleton 0.00% (9/10 seeds ever), make arrow 0.00% (2/10 seeds ever), enter dungeon 0.00% (4/10 seeds ever). Same file.
- Common (8): collect sapling 81.6%, collect wood 67.7%, wake up 53.9%, place plant 50.8%, place table 41.7%, collect drink 21.9%, make wood pickaxe 6.4%, eat cow 5.6%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 336 (52.5%); starvation 202 (31.6%); arrow 93 (14.5%); dehydration 9 (1.4%). At checkpoint 0 (untrained): zombie 51.6%, starvation 28.1%, arrow 20.3%. Source: `training-analysis-1m-dyna-q`, `hard/death_causes.csv`, plot `plots/death_causes_over_training.png`.
- Stalls (>= 300 steps without an unlock): 81 stalls in 640 final eval episodes, 0.127 per episode (seed interval 0.086 to 0.172), median length 328 steps, 1.2% escaped by an unlock. All started on floor 0; started by wake up 22 (27.2%), episode start 20 (24.7%), place plant 14 (17.3%), collect drink 8 (9.9%). Source: `training-analysis-1m-dyna-q`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 1 by an unlock, 80 by death. Longest stall 554 steps (run `3df843868f944bda8e0cc6f8188ec120`, seed 4, eval seed 10029, steps 130 to 684, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-dyna-q` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-dyna-q-stall-1`** (run `dyna-q-1m-74c22d85-s0`, `9e31eadfa50d476c8270d287d992c223`, checkpoint 244 (999,424 env steps), eval seed 10043, stochastic). Situation: a stall of 358 steps from step 46 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory sapling 1. Behaviour during the stall: most chosen actions place table 103, place plant 63, drink potion green 49 over the 358 steps; 2.21% of overworld tiles first seen in the window; combat damage 0, close calls 1, lowest health 0. What broke it: collect wood at step 404 (then health 0, food 0, drink 0, energy 9, inventory wood 1, sapling 6). Outcome: the episode lasted 404 steps. Replay: `gifs/1m-dyna-q-stall-1.gif`, `overlays/1m-dyna-q-stall-1.mp4`, `masters/case-1m-dyna-q-stall-1.mkv`, `rollouts/case-1m-dyna-q-stall-1.npz`.

**`1m-dyna-q-stall-2`** (run `dyna-q-1m-74c22d85-s4`, `3df843868f944bda8e0cc6f8188ec120`, checkpoint 1 (4,096 env steps), eval seed 10048, stochastic). Situation: a stall of 317 steps from step 66 on floor 0; at its start health 9, food 7, drink 7, energy 9, inventory empty. Behaviour during the stall: most chosen actions make iron armour 11, make torch 11, drink potion red 9 over the 317 steps; 1.52% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: collect wood at step 383 (then health 2, food 0, drink 0, energy 9, inventory wood 1). Outcome: the episode lasted 400 steps. Replay: `gifs/1m-dyna-q-stall-2.gif`, `overlays/1m-dyna-q-stall-2.mp4`, `masters/case-1m-dyna-q-stall-2.mkv`, `rollouts/case-1m-dyna-q-stall-2.npz`.

**`1m-dyna-q-stall-3`** (run `dyna-q-1m-74c22d85-s5`, `4a271c8ee77f4395a3ebf7f07d7f003e`, checkpoint 7 (28,672 env steps), eval seed 10012, stochastic). Situation: a stall of 325 steps from step 45 on floor 0; at its start health 9, food 8, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions right 26, sleep 25, make stone sword 24 over the 325 steps; 1.48% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 370 (then health 1, food 0, drink 0, energy 6, inventory wood 1). Outcome: the episode lasted 388 steps. Replay: `gifs/1m-dyna-q-stall-3.gif`, `overlays/1m-dyna-q-stall-3.mp4`, `masters/case-1m-dyna-q-stall-3.mkv`, `rollouts/case-1m-dyna-q-stall-3.npz`.

**`1m-dyna-q-stall-4`** (run `dyna-q-1m-74c22d85-s0`, `9e31eadfa50d476c8270d287d992c223`, checkpoint 148 (606,208 env steps), eval seed 10014, stochastic). Situation: a stall of 343 steps from step 0 on floor 0; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions down 198, enchant bow 46, drink potion green 8 over the 343 steps; 1.82% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: wake up at step 343 (then health 3, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-dyna-q-stall-4.gif`, `overlays/1m-dyna-q-stall-4.mp4`, `masters/case-1m-dyna-q-stall-4.mkv`, `rollouts/case-1m-dyna-q-stall-4.npz`.

### Deepest episodes

The 3 episodes with the deepest floor, ties by achievements unlocked (kind `deep_floor`). No floor beyond 0 is reached at 1M, so these are the overworld episodes with the most achievements.

**`1m-dyna-q-deep_floor-1`** (run `dyna-q-1m-74c22d85-s1`, `91dadb59a31240b6b5eaff024f8dae84`, checkpoint 244 (999,424 env steps), eval seed 10014, stochastic). Situation: an episode with 9 achievements unlocked, none below floor 0; window steps 34 to 184, at its start health 9, food 8, drink 9, energy 8, inventory wood 1. Behaviour: most chosen actions do 60, right 24, up 15; 13.06% of overworld tiles first seen in the window; combat damage 2, close calls 0, lowest health 7. At step 184 (the last unlock) health 8, food 3, drink 3, energy 5, inventory wood 1, sapling 1. Outcome: the episode lasted 200 steps. Replay: `gifs/1m-dyna-q-deep_floor-1.gif`, `masters/case-1m-dyna-q-deep_floor-1.mkv`, `rollouts/case-1m-dyna-q-deep_floor-1.npz`.

**`1m-dyna-q-deep_floor-2`** (run `dyna-q-1m-74c22d85-s4`, `3df843868f944bda8e0cc6f8188ec120`, checkpoint 244 (999,424 env steps), eval seed 10005, stochastic). Situation: an episode with 8 achievements unlocked, none below floor 0; window steps 102 to 252, at its start health 7, food 6, drink 5, energy 9, inventory sapling 2. Behaviour: most chosen actions do 35, left 25, up 17; 12.11% of overworld tiles first seen in the window; combat damage 7, close calls 0, lowest health 4. At step 252 (the last unlock) health 4, food 6, drink 2, energy 8, inventory wood 1, sapling 2. Outcome: the episode lasted 278 steps. Replay: `gifs/1m-dyna-q-deep_floor-2.gif`, `masters/case-1m-dyna-q-deep_floor-2.mkv`, `rollouts/case-1m-dyna-q-deep_floor-2.npz`.

**`1m-dyna-q-deep_floor-3`** (run `dyna-q-1m-74c22d85-s6`, `8bf103fc8486422d8d75b0416ec8989b`, checkpoint 244 (999,424 env steps), eval seed 10038, stochastic). Situation: an episode with 8 achievements unlocked, none below floor 0; window steps 148 to 298, at its start health 9, food 4, drink 3, energy 7, inventory sapling 1. Behaviour: most chosen actions up 92, do 55, left 1; 3.65% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 7. At step 298 (the last unlock) health 7, food 0, drink 9, energy 2, inventory wood 2, stone 1, sapling 1. Outcome: the episode lasted 400 steps. Replay: `gifs/1m-dyna-q-deep_floor-3.gif`, `masters/case-1m-dyna-q-deep_floor-3.mkv`, `rollouts/case-1m-dyna-q-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-dyna-q-rare_unlock-1`** (run `dyna-q-1m-74c22d85-s4`, `3df843868f944bda8e0cc6f8188ec120`, checkpoint 244 (999,424 env steps), eval seed 10008, stochastic). Situation: make wood sword (rare at the final checkpoint) is unlocked at step 66; window steps 0 to 66, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions up 15, do 12, left 6; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 7, drink 7, energy 9, inventory wood 2. Outcome: the episode lasted 582 steps. Replay: `gifs/1m-dyna-q-rare_unlock-1.gif`, `masters/case-1m-dyna-q-rare_unlock-1.mkv`, `rollouts/case-1m-dyna-q-rare_unlock-1.npz`.

**`1m-dyna-q-rare_unlock-2`** (run `dyna-q-1m-74c22d85-s0`, `9e31eadfa50d476c8270d287d992c223`, checkpoint 244 (999,424 env steps), eval seed 10047, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 46; window steps 0 to 46, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 37, drink potion green 4, down 3; combat damage 8, close calls 1, lowest health 1. At the unlock health 1, food 8, drink 7, energy 8, inventory sapling 2. Outcome: the episode lasted 228 steps. Replay: `gifs/1m-dyna-q-rare_unlock-2.gif`, `masters/case-1m-dyna-q-rare_unlock-2.mkv`, `rollouts/case-1m-dyna-q-rare_unlock-2.npz`.

**`1m-dyna-q-rare_unlock-3`** (run `dyna-q-1m-74c22d85-s6`, `8bf103fc8486422d8d75b0416ec8989b`, checkpoint 244 (999,424 env steps), eval seed 10036, stochastic). Situation: collect stone (rare at the final checkpoint) is unlocked at step 81; window steps 0 to 81, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 36, right 16, down 10; combat damage 2, close calls 0, lowest health 7. At the unlock health 9, food 6, drink 6, energy 7, inventory wood 1, stone 1. Outcome: the episode lasted 241 steps. Replay: `gifs/1m-dyna-q-rare_unlock-3.gif`, `masters/case-1m-dyna-q-rare_unlock-3.mkv`, `rollouts/case-1m-dyna-q-rare_unlock-3.npz`.

**`1m-dyna-q-rare_unlock-4`** (run `dyna-q-1m-74c22d85-s1`, `91dadb59a31240b6b5eaff024f8dae84`, checkpoint 148 (606,208 env steps), eval seed 10000, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 43; window steps 0 to 43, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 24, place furnace 17, make stone pickaxe 1; combat damage 2, close calls 0, lowest health 7. At the unlock health 7, food 8, drink 7, energy 8, inventory sapling 3. Outcome: the episode lasted 99 steps. Replay: `gifs/1m-dyna-q-rare_unlock-4.gif`, `masters/case-1m-dyna-q-rare_unlock-4.mkv`, `rollouts/case-1m-dyna-q-rare_unlock-4.npz`.

**`1m-dyna-q-rare_unlock-5`** (run `dyna-q-1m-74c22d85-s1`, `91dadb59a31240b6b5eaff024f8dae84`, checkpoint 244 (999,424 env steps), eval seed 10014, stochastic). Situation: place furnace (rare at the final checkpoint) is unlocked at step 184; window steps 34 to 184, at its start health 9, food 8, drink 9, energy 8, inventory wood 1. Behaviour: most chosen actions do 60, right 24, up 15; combat damage 2, close calls 0, lowest health 7. At the unlock health 8, food 3, drink 3, energy 5, inventory wood 1, sapling 1. Outcome: the episode lasted 200 steps. Replay: `gifs/1m-dyna-q-rare_unlock-5.gif`, `masters/case-1m-dyna-q-rare_unlock-5.mkv`, `rollouts/case-1m-dyna-q-rare_unlock-5.npz`.

**`1m-dyna-q-rare_unlock-6`** (run `dyna-q-1m-74c22d85-s6`, `8bf103fc8486422d8d75b0416ec8989b`, checkpoint 244 (999,424 env steps), eval seed 10045, stochastic). Situation: collect coal (rare at the final checkpoint) is unlocked at step 211; window steps 61 to 211, at its start health 9, food 7, drink 7, energy 8, inventory wood 1, sapling 1. Behaviour: most chosen actions right 44, up 37, do 17; combat damage 0, close calls 0, lowest health 8. At the unlock health 8, food 1, drink 0, energy 3, inventory stone 2, coal 1, sapling 3. Outcome: the episode lasted 242 steps. Replay: `gifs/1m-dyna-q-rare_unlock-6.gif`, `masters/case-1m-dyna-q-rare_unlock-6.mkv`, `rollouts/case-1m-dyna-q-rare_unlock-6.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-dyna-q-death-1`** (run `dyna-q-1m-74c22d85-s4`, `3df843868f944bda8e0cc6f8188ec120`, checkpoint 244 (999,424 env steps), eval seed 10040, stochastic). Situation: death by zombie at step 220; the window (steps 70 to 220) opens with health 9, food 7, drink 6, energy 9, inventory wood 3. Behaviour: most chosen actions right 37, left 21, descend 20; combat damage 9, close calls 0, lowest health 0. At death health 0, food 2, drink 0, energy 7, inventory wood 2. Outcome: the episode ended after 220 steps. Replay: `gifs/1m-dyna-q-death-1.gif`, `masters/case-1m-dyna-q-death-1.mkv`, `rollouts/case-1m-dyna-q-death-1.npz`.

**`1m-dyna-q-death-2`** (run `dyna-q-1m-74c22d85-s4`, `3df843868f944bda8e0cc6f8188ec120`, checkpoint 244 (999,424 env steps), eval seed 10026, stochastic). Situation: death by starvation at step 355; the window (steps 205 to 355) opens with health 9, food 2, drink 1, energy 8, inventory empty. Behaviour: most chosen actions level up strength 129, level up intelligence 12, drink potion red 3; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 3, inventory empty. Outcome: the episode ended after 355 steps. Replay: `gifs/1m-dyna-q-death-2.gif`, `masters/case-1m-dyna-q-death-2.mkv`, `rollouts/case-1m-dyna-q-death-2.npz`.

**`1m-dyna-q-death-3`** (run `dyna-q-1m-74c22d85-s3`, `57a2dac8eb094f2cb2cd97918b456204`, checkpoint 244 (999,424 env steps), eval seed 10013, stochastic). Situation: death by arrow at step 260; the window (steps 110 to 260) opens with health 9, food 5, drink 4, energy 6, inventory wood 1, sapling 2. Behaviour: most chosen actions ascend 115, place furnace 35; combat damage 6, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 1, inventory wood 1, sapling 2. Outcome: the episode ended after 260 steps. Replay: `gifs/1m-dyna-q-death-3.gif`, `masters/case-1m-dyna-q-death-3.mkv`, `rollouts/case-1m-dyna-q-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. Tuning runs (`tune-coarse`, `tune-fine`) were not analysed (only the final phases are loaded). Mitigation: none tried after tuning, so no entry appends to [[Experiment decisions]].

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 9 / 10 | 16.4k to 999k | 4 drops, deepest drawdown 1.72 (floor 0.26) (run 9e31eadfa50d476c8270d287d992c223; 8 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Collapse | 1M | 5 / 10 | 16.4k to 999k | drawdown 1.74, final 0.26 of best 0.52 (floor 0.13) (run 57a2dac8eb094f2cb2cd97918b456204; 4 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | `training-analysis-1m-dyna-q/obstacles.csv` |
| Reward hacking | 1M | 7 / 10 | 111k to 999k | dyna/bonus_mean rising while the score is flat from 119k; action cycle (period 2, up to 110 steps) in 10 / 64 final episodes (run 91dadb59a31240b6b5eaff024f8dae84; 6 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 938k | >= 20% to < 5%: collect_drink (111k to 119k), collect_sapling (45.1k to 131k), collect_wood (28.7k to 36.9k), defeat_zombie (733k to 766k), eat_cow (700k to 729k), place_plant (45.1k to 53.2k), place_table (53.2k to 65.5k) (run 9e31eadfa50d476c8270d287d992c223; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv`, `training-analysis-1m-dyna-q/plots/achievement_status.png` |
| Skill interference | 1M | 10 / 10 | 4.1k to 971k | success from >= 20% to < 5%: collect_wood (4.1k to 81.9k; eat chosen +13%), eat (700k to 709k; collect_wood chosen +24%), fight (729k to 745k; descend chosen +13%), place_table (848k to 856k; descend chosen +7%), sleep (45.1k to 90.1k; collect_wood chosen +26%), unlock_any (4.1k to 49.2k; descend chosen +12%) (run 9e31eadfa50d476c8270d287d992c223; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Deep-skill starvation | 1M | 10 / 10 |  | deep skills chosen fewer than 100 times: collect_diamond about 0, make_iron_pickaxe about 0 (run 9e31eadfa50d476c8270d287d992c223; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Survival failure despite flags | 1M | 10 / 10 | 999k to 999k | 55% of 173 low-stat interruptions followed by death within 100 steps (run 9e31eadfa50d476c8270d287d992c223; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Planned-value gap | 1M | 10 / 10 | 307k to 999k | smdp/q_start > 2 x mean return after the epsilon decay from 307k (run 9e31eadfa50d476c8270d287d992c223; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Stale model | 1M | 10 / 10 | 4.1k to 12.3k | dyna/model_accuracy fell 0.76 within 10% of the budget (run 9e31eadfa50d476c8270d287d992c223; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Bonus-driven re-trying | 1M | 2 / 10 | 606k to 999k | share of decisions on saturated pairs +17% while the score is flat (run 8bf103fc8486422d8d75b0416ec8989b; 1 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |

- Absent in every run: Plateau, Entropy collapse, Numerical failure, Exploration failure, Survival failure, Stuck top level, Fallback collapse, Timeout dominance, Key aliasing, Blind skill test, No planning effect.
- Detector verdicts over the tuned runs (220 run-detector pairs): absent 127, detected 83, not logged 10; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

### 1B

1B: pending (final-1b not run, R75). The 1B obstacles table (3 seeds) is added by rerunning `analysis.training` once the 1B finals exist.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[Dyna-Q training analysis]] (this note): floors - only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; achievements - 50 never, 9 rare, 8 common; killer and stalls - zombie 52.5% of deaths, 0.127 stalls per episode (median 328 steps, 1.2% escaped); final score 1.077.
- [[PQN-RNN training analysis]] (base control): floors - only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.
- [[PPO training analysis]]: floors - only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PPO-RNN training analysis]]: floors - only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.

## Ablations

Ablation units of `final-1m` (R77): reported apart from the tuned config, never pooled into the sections above. The change is an override of the tuned config in `configs/phases/final-1m.yaml`.

| Unit | Change | n | Final score (sd) | Stochastic eval | Floors reached | Never / rare / common achievements | Top death causes | Stalls per episode (median, escape) |
|---|---|---|---|---|---|---|---|---|
| tuned | - | 10 | 1.077 (0.619) | 1.081 | floor 0 only | 50 / 9 / 8 | zombie 52.5%, starvation 31.6% | 0.127 (328, 1.2%) |
| kappa0 | `plan_kappa: 0.0` (no planning bonus) | 10 | 0.692 (0.591) | 0.689 | floor 0 only | 51 / 9 / 7 | zombie 47.5%, starvation 34.1% | 0.180 (331, 0.9%) |

Sources: `hard/*.csv` (rows of each `config_ref`) and the scores of `outputs/metrics/final-1m/runs.parquet`. The `tuned` row repeats the main sections for comparison.

### Obstacles: kappa0

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 10 / 10 | 16.4k to 999k | 2 drops, deepest drawdown 1.85 (floor 0.29) (run dc18150a2cd94cd2a947d3d1c14f5bd5; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Collapse | 1M | 8 / 10 | 16.4k to 999k | drawdown 1.85, final 0.99 of best 1.15 (floor 0.29) (run dc18150a2cd94cd2a947d3d1c14f5bd5; 7 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Plateau | 1M | 1 / 10 | 500k to 999k | score gain -0.28 from 500k to the end (< 0.05) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | `training-analysis-1m-dyna-q/obstacles.csv` |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 999k | >= 20% to < 5%: collect_drink (111k to 119k), collect_sapling (45.1k to 131k), collect_wood (28.7k to 36.9k), place_plant (45.1k to 53.2k), place_table (53.2k to 65.5k), wake_up (20.5k to 467k) (run dc18150a2cd94cd2a947d3d1c14f5bd5; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv`, `training-analysis-1m-dyna-q/plots/achievement_status.png` |
| Skill interference | 1M | 10 / 10 | 4.1k to 864k | success from >= 20% to < 5%: collect_wood (4.1k to 81.9k; eat chosen +13%), drink (598k to 606k; collect_wood chosen +14%), eat (602k to 610k; descend chosen +3%), place_table (623k to 631k; unlock_any chosen +9%), sleep (45.1k to 90.1k; collect_wood chosen +26%), unlock_any (4.1k to 49.2k; descend chosen +12%) (run dc18150a2cd94cd2a947d3d1c14f5bd5; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Deep-skill starvation | 1M | 10 / 10 |  | deep skills chosen fewer than 100 times: collect_diamond about 0, make_iron_pickaxe about 0 (run dc18150a2cd94cd2a947d3d1c14f5bd5; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Survival failure despite flags | 1M | 10 / 10 | 999k to 999k | 61% of 207 low-stat interruptions followed by death within 100 steps (run dc18150a2cd94cd2a947d3d1c14f5bd5; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Planned-value gap | 1M | 10 / 10 | 307k to 999k | smdp/q_start > 2 x mean return after the epsilon decay from 307k (run dc18150a2cd94cd2a947d3d1c14f5bd5; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |
| Stale model | 1M | 10 / 10 | 4.1k to 12.3k | dyna/model_accuracy fell 0.76 within 10% of the budget (run dc18150a2cd94cd2a947d3d1c14f5bd5; 9 more) | none after tuning | open at 1M | `training-analysis-1m-dyna-q/obstacles.csv` |

Cases of kappa0 (`cases.csv`, replay files under `training-analysis-1m-dyna-q`):

| Case | Kind | Detail | Run | Checkpoint | Eval seed | GIF |
|---|---|---|---|---|---|---|
| `1m-dyna-q-kappa0-stall-1` | stall | 324 steps | `dyna-q-1m-3fd02b70-s1` | 244 | 10044 | `gifs/1m-dyna-q-kappa0-stall-1.gif` |
| `1m-dyna-q-kappa0-stall-2` | stall | 317 steps | `dyna-q-1m-3fd02b70-s2` | 1 | 10048 | `gifs/1m-dyna-q-kappa0-stall-2.gif` |
| `1m-dyna-q-kappa0-stall-3` | stall | 325 steps | `dyna-q-1m-3fd02b70-s5` | 7 | 10012 | `gifs/1m-dyna-q-kappa0-stall-3.gif` |
| `1m-dyna-q-kappa0-stall-4` | stall | 374 steps | `dyna-q-1m-3fd02b70-s4` | 148 | 10062 | `gifs/1m-dyna-q-kappa0-stall-4.gif` |
| `1m-dyna-q-kappa0-deep_floor-1` | deep_floor | floor 0 | `dyna-q-1m-3fd02b70-s4` | 244 | 10002 | `gifs/1m-dyna-q-kappa0-deep_floor-1.gif` |
| `1m-dyna-q-kappa0-deep_floor-2` | deep_floor | floor 0 | `dyna-q-1m-3fd02b70-s4` | 244 | 10023 | `gifs/1m-dyna-q-kappa0-deep_floor-2.gif` |
| `1m-dyna-q-kappa0-deep_floor-3` | deep_floor | floor 0 | `dyna-q-1m-3fd02b70-s5` | 244 | 10032 | `gifs/1m-dyna-q-kappa0-deep_floor-3.gif` |
| `1m-dyna-q-kappa0-rare_unlock-1` | rare_unlock | eat_cow | `dyna-q-1m-3fd02b70-s6` | 244 | 10002 | `gifs/1m-dyna-q-kappa0-rare_unlock-1.gif` |
| `1m-dyna-q-kappa0-rare_unlock-2` | rare_unlock | make_wood_sword | `dyna-q-1m-3fd02b70-s4` | 244 | 10018 | `gifs/1m-dyna-q-kappa0-rare_unlock-2.gif` |
| `1m-dyna-q-kappa0-rare_unlock-3` | rare_unlock | defeat_zombie | `dyna-q-1m-3fd02b70-s1` | 244 | 10047 | `gifs/1m-dyna-q-kappa0-rare_unlock-3.gif` |
| `1m-dyna-q-kappa0-rare_unlock-4` | rare_unlock | collect_stone | `dyna-q-1m-3fd02b70-s5` | 244 | 10032 | `gifs/1m-dyna-q-kappa0-rare_unlock-4.gif` |
| `1m-dyna-q-kappa0-rare_unlock-5` | rare_unlock | defeat_skeleton | `dyna-q-1m-3fd02b70-s7` | 244 | 10058 | `gifs/1m-dyna-q-kappa0-rare_unlock-5.gif` |
| `1m-dyna-q-kappa0-rare_unlock-6` | rare_unlock | place_furnace | `dyna-q-1m-3fd02b70-s8` | 7 | 10014 | `gifs/1m-dyna-q-kappa0-rare_unlock-6.gif` |
| `1m-dyna-q-kappa0-rare_unlock-7` | rare_unlock | collect_coal | `dyna-q-1m-3fd02b70-s8` | 7 | 10014 | `gifs/1m-dyna-q-kappa0-rare_unlock-7.gif` |
| `1m-dyna-q-kappa0-death-1` | death | zombie | `dyna-q-1m-3fd02b70-s1` | 244 | 10038 | `gifs/1m-dyna-q-kappa0-death-1.gif` |
| `1m-dyna-q-kappa0-death-2` | death | starvation | `dyna-q-1m-3fd02b70-s1` | 244 | 10012 | `gifs/1m-dyna-q-kappa0-death-2.gif` |
| `1m-dyna-q-kappa0-death-3` | death | arrow | `dyna-q-1m-3fd02b70-s1` | 244 | 10005 | `gifs/1m-dyna-q-kappa0-death-3.gif` |
