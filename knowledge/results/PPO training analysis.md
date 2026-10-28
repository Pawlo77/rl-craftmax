---
title: "PPO training analysis"
type: concept
domain: analysis
parent: "[[2026-10-20-craftax-project]]"
tags: [concept, results, analysis, training, craftax-1m]
created: 2026-10-28
updated: 2026-10-28
related-code:
  - "[[src/wut_murl_craftax/analysis/hard.py]]"
  - "[[src/wut_murl_craftax/analysis/cases.py]]"
  - "[[src/wut_murl_craftax/analysis/obstacles.py]]"
  - "[[src/wut_murl_craftax/analysis/training.py]]"
---
# PPO training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 2.173 (sd 0.091), stochastic eval score 2.164; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%).
- Achievements: 48 of 67 never unlocked by any seed at any checkpoint (basic 7, intermediate 17, advanced 15, very advanced 9), 10 rare (final rate < 5%), 9 common.
- Deaths at the final checkpoint: zombie 61.1%, starvation 22.7%, arrow 14.5%, dehydration 1.6%, exhaustion 0.2% (the untrained policy: zombie 46.2%, starvation 34.8%, arrow 18.9%).
- Stalls: 61 stalls in 640 final eval episodes, 0.095 per episode (seed interval 0.078 to 0.114), median length 332 steps, 8.2% escaped by an unlock; they are started by collect drink (30%) and wake up (28%).
- Obstacles at 1M: plateau 7/10; not logged (R78): exploration failure, reward hacking. Caveat: skill forgetting 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m`, PPO (control), 10 seeds (0-9) of the tuned config; method: [[PPO design]], plan [[2026-10-20-craftax-method-ppo]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/ppo/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-ppo`, id `f2998769c76445cdb66cf95912b31f49`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 20 case episodes, all replayed from the stored eval files.
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- 1B: pending (final-1b not run, R75); sections marked 1B below are placeholders for the same tables at 1B.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 10 of this method's 10 tuned runs (10 of 10 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 63, detected 17, not logged 20; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Floor entries before or beside the final eval: `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%). Source: `training-analysis-1m-ppo`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (48 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (7): make stone pickaxe, collect iron, collect diamond, make iron pickaxe, make iron sword, make torch, place torch; intermediate (17): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-ppo`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status_tuned.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 10): defeat zombie 2.66% (10/10 seeds ever), collect stone 0.94% (10/10 seeds ever), defeat skeleton 0.47% (10/10 seeds ever), place furnace 0.47% (9/10 seeds ever), place stone 0.16% (7/10 seeds ever), eat plant 0.16% (10/10 seeds ever), make stone sword 0.00% (1/10 seeds ever), collect coal 0.00% (4/10 seeds ever), make arrow 0.00% (1/10 seeds ever), enter dungeon 0.00% (1/10 seeds ever). Same file.
- Common (9): collect sapling 97.8%, wake up 96.4%, place plant 93.9%, collect wood 93.8%, place table 81.6%, collect drink 55.2%, make wood pickaxe 22.5%, make wood sword 20.5%, eat cow 12.7%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 391 (61.1%); starvation 145 (22.7%); arrow 93 (14.5%); dehydration 10 (1.6%); exhaustion 1 (0.2%). At checkpoint 0 (untrained): zombie 46.2%, starvation 34.8%, arrow 18.9%. Source: `training-analysis-1m-ppo`, `hard/death_causes.csv`, plot `plots/death_causes_over_training_tuned.png`.
- Stalls (>= 300 steps without an unlock): 61 stalls in 640 final eval episodes, 0.095 per episode (seed interval 0.078 to 0.114), median length 332 steps, 8.2% escaped by an unlock. All started on floor 0; started by collect drink 18 (29.5%), wake up 17 (27.9%), place table 10 (16.4%), eat cow 8 (13.1%). Source: `training-analysis-1m-ppo`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 5 by an unlock, 56 by death. Longest stall 700 steps (run `b4dac6b32a184513b43c4e3f0dd76460`, seed 8, eval seed 10014, steps 280 to 980, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-ppo` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 61 stalls, 5 ended by an unlock, 56 by death (91.8%).
- Before the unlock (8 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 2 cases; most frequent actions do 25%, up 11%, down 8%; movement keys are 28% of the last 50 steps against 30% of the whole stall window, `do` 25% against 30%; the most frequent action of the last 50 differs from the window's in 3 of 8 cases.
- State at the unlock: health <= 4 in 7 of 8 cases (median health 2 at the unlock, 9 at the stall start), food 0 in 5 of 8 (median food 0, 8 at the start), drink 0 in 4 of 8.
- Achievements that ended the stalls: collect wood 2, eat plant 1, place table 1, eat cow 1, defeat skeleton 1, wake up 1, collect sapling 1.
- Deepest floor: the deep-floor cases are all overworld (floor 0) episodes, so the route question is moot at 1M.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-ppo-stall-1`** (run `ppo-1m-e06ac6db-s7`, `99a2896e883a4e99a123883904e36be0`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: a stall of 468 steps from step 834 on floor 0, begun after place table; at its start health 7, food 0, drink 9, energy 9, inventory sapling 2. Behaviour during the stall: most chosen actions do 168, down 55, left 21 over the 468 steps; 0.09% of overworld tiles first seen in the window; combat damage 8, close calls 2, lowest health 1. What broke it: eat plant at step 1302 (then health 8, food 9, drink 8, energy 9, inventory wood 1, sapling 4). Outcome: the episode lasted 1463 steps. Replay: `gifs/1m-ppo-stall-1.gif`, `overlays/1m-ppo-stall-1.mp4`, `masters/case-1m-ppo-stall-1.mkv`, `rollouts/case-1m-ppo-stall-1.npz`.

**`1m-ppo-stall-2`** (run `ppo-1m-e06ac6db-s7`, `99a2896e883a4e99a123883904e36be0`, checkpoint 244 (999,424 env steps), eval seed 10042, stochastic). Situation: a stall of 351 steps from step 47 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions do 140, down 30, up 27 over the 351 steps; 1.30% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: place table at step 398 (then health 3, food 0, drink 8, energy 8, inventory empty). Outcome: the episode lasted 444 steps. Replay: `gifs/1m-ppo-stall-2.gif`, `overlays/1m-ppo-stall-2.mp4`, `masters/case-1m-ppo-stall-2.mkv`, `rollouts/case-1m-ppo-stall-2.npz`.

**`1m-ppo-stall-3`** (run `ppo-1m-e06ac6db-s7`, `99a2896e883a4e99a123883904e36be0`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: a stall of 344 steps from step 43 on floor 0, begun after wake up; at its start health 9, food 8, drink 9, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions do 114, down 52, up 25 over the 344 steps; 7.25% of overworld tiles first seen in the window; combat damage 0, close calls 1, lowest health 2. What broke it: eat cow at step 387 (then health 2, food 6, drink 9, energy 9, inventory wood 1, sapling 1). Outcome: the episode lasted 1463 steps. Replay: `gifs/1m-ppo-stall-3.gif`, `overlays/1m-ppo-stall-3.mp4`, `masters/case-1m-ppo-stall-3.mkv`, `rollouts/case-1m-ppo-stall-3.npz`.

**`1m-ppo-stall-4`** (run `ppo-1m-e06ac6db-s7`, `99a2896e883a4e99a123883904e36be0`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: a stall of 343 steps from step 387 on floor 0, begun after eat cow; at its start health 2, food 6, drink 9, energy 9, inventory wood 1, sapling 1. Behaviour during the stall: most chosen actions do 128, down 36, up 19 over the 343 steps; 1.26% of overworld tiles first seen in the window; combat damage 4, close calls 1, lowest health 2. What broke it: defeat skeleton at step 730 (then health 2, food 4, drink 9, energy 7, inventory wood 1, sapling 2). Outcome: the episode lasted 1463 steps. Replay: `gifs/1m-ppo-stall-4.gif`, `overlays/1m-ppo-stall-4.mp4`, `masters/case-1m-ppo-stall-4.mkv`, `rollouts/case-1m-ppo-stall-4.npz`.

**`1m-ppo-stall-5`** (run `ppo-1m-e06ac6db-s3`, `5a49268d67d54599836795356e27437e`, checkpoint 244 (999,424 env steps), eval seed 10053, stochastic). Situation: a stall of 315 steps from step 59 on floor 0, begun after wake up; at its start health 9, food 7, drink 7, energy 9, inventory empty. Behaviour during the stall: most chosen actions up 78, place table 29, do 27 over the 315 steps; 0.00% of overworld tiles first seen in the window; combat damage 0, close calls 1, lowest health 1. What broke it: collect wood at step 374 (then health 1, food 0, drink 0, energy 9, inventory wood 1). Outcome: the episode lasted 375 steps. Replay: `gifs/1m-ppo-stall-5.gif`, `overlays/1m-ppo-stall-5.mp4`, `masters/case-1m-ppo-stall-5.mkv`, `rollouts/case-1m-ppo-stall-5.npz`.

**`1m-ppo-stall-6`** (run `ppo-1m-e06ac6db-s1`, `4ecf5fdb9bf24706ac2ce86553eed51b`, checkpoint 1 (4,096 env steps), eval seed 10041, stochastic). Situation: a stall of 355 steps from step 0 on floor 0; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions noop 12, ascend 11, level up intelligence 11 over the 355 steps; 3.86% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: wake up at step 355 (then health 2, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-ppo-stall-6.gif`, `overlays/1m-ppo-stall-6.mp4`, `masters/case-1m-ppo-stall-6.mkv`, `rollouts/case-1m-ppo-stall-6.npz`.

**`1m-ppo-stall-7`** (run `ppo-1m-e06ac6db-s3`, `5a49268d67d54599836795356e27437e`, checkpoint 12 (49,152 env steps), eval seed 10010, stochastic). Situation: a stall of 347 steps from step 48 on floor 0; at its start health 9, food 8, drink 7, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 50, sleep 24, place plant 24 over the 347 steps; 0.52% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect sapling at step 395 (then health 1, food 0, drink 0, energy 9, inventory sapling 1). Outcome: the episode lasted 398 steps. Replay: `gifs/1m-ppo-stall-7.gif`, `overlays/1m-ppo-stall-7.mp4`, `masters/case-1m-ppo-stall-7.mkv`, `rollouts/case-1m-ppo-stall-7.npz`.

**`1m-ppo-stall-8`** (run `ppo-1m-e06ac6db-s7`, `99a2896e883a4e99a123883904e36be0`, checkpoint 148 (606,208 env steps), eval seed 10031, stochastic). Situation: a stall of 321 steps from step 43 on floor 0; at its start health 9, food 8, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions up 125, do 40, sleep 24 over the 321 steps; 1.35% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: collect wood at step 364 (then health 2, food 0, drink 0, energy 9, inventory wood 1). Outcome: the episode lasted 389 steps. Replay: `gifs/1m-ppo-stall-8.gif`, `overlays/1m-ppo-stall-8.mp4`, `masters/case-1m-ppo-stall-8.mkv`, `rollouts/case-1m-ppo-stall-8.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-ppo-deep_floor-1`** (run `ppo-1m-e06ac6db-s0`, `22fa55cebeb140729639e8fb2a40c18b`, checkpoint 244 (999,424 env steps), eval seed 10036, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 422 to 572, at its start health 9, food 2, drink 7, energy 7, inventory wood 1, sapling 4. Behaviour: most chosen actions make wood sword 20, right 18, do 16; 6.29% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 5. At step 572 (the last unlock) health 5, food 0, drink 1, energy 9, inventory sapling 4. Outcome: the episode lasted 608 steps. Replay: `gifs/1m-ppo-deep_floor-1.gif`, `masters/case-1m-ppo-deep_floor-1.mkv`, `rollouts/case-1m-ppo-deep_floor-1.npz`.

**`1m-ppo-deep_floor-2`** (run `ppo-1m-e06ac6db-s5`, `6ede1610c0074b2c8e9a23c2c52a8d4d`, checkpoint 244 (999,424 env steps), eval seed 10002, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 38 to 188, at its start health 9, food 8, drink 8, energy 8, inventory wood 1. Behaviour: most chosen actions do 32, down 26, place table 25; 8.55% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 9. At step 188 (the last unlock) health 9, food 2, drink 9, energy 5, inventory empty. Outcome: the episode lasted 319 steps. Replay: `gifs/1m-ppo-deep_floor-2.gif`, `masters/case-1m-ppo-deep_floor-2.mkv`, `rollouts/case-1m-ppo-deep_floor-2.npz`.

**`1m-ppo-deep_floor-3`** (run `ppo-1m-e06ac6db-s5`, `6ede1610c0074b2c8e9a23c2c52a8d4d`, checkpoint 244 (999,424 env steps), eval seed 10017, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 85 to 235, at its start health 9, food 6, drink 6, energy 9, inventory wood 1, sapling 1. Behaviour: most chosen actions do 60, up 21, down 21; 3.56% of overworld tiles first seen in the window; combat damage 4, close calls 0, lowest health 5. At step 235 (the last unlock) health 5, food 8, drink 9, energy 4, inventory wood 1, sapling 5. Outcome: the episode lasted 568 steps. Replay: `gifs/1m-ppo-deep_floor-3.gif`, `masters/case-1m-ppo-deep_floor-3.mkv`, `rollouts/case-1m-ppo-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-ppo-rare_unlock-1`** (run `ppo-1m-e06ac6db-s6`, `05541d512ff6453ab2a6f528f7c365c2`, checkpoint 244 (999,424 env steps), eval seed 10047, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 32; window steps 0 to 32, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 18, sleep 5, place plant 4; combat damage 4, close calls 0, lowest health 6. At the unlock health 6, food 8, drink 8, energy 8, inventory wood 1. Outcome: the episode lasted 256 steps. Replay: `gifs/1m-ppo-rare_unlock-1.gif`, `masters/case-1m-ppo-rare_unlock-1.mkv`, `rollouts/case-1m-ppo-rare_unlock-1.npz`.

**`1m-ppo-rare_unlock-2`** (run `ppo-1m-e06ac6db-s5`, `6ede1610c0074b2c8e9a23c2c52a8d4d`, checkpoint 244 (999,424 env steps), eval seed 10002, stochastic). Situation: collect stone (rare at the final checkpoint) is unlocked at step 154; window steps 4 to 154, at its start health 9, food 9, drink 9, energy 9, inventory wood 1. Behaviour: most chosen actions do 36, place table 25, down 22; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 4, drink 8, energy 6, inventory stone 1. Outcome: the episode lasted 319 steps. Replay: `gifs/1m-ppo-rare_unlock-2.gif`, `masters/case-1m-ppo-rare_unlock-2.mkv`, `rollouts/case-1m-ppo-rare_unlock-2.npz`.

**`1m-ppo-rare_unlock-3`** (run `ppo-1m-e06ac6db-s5`, `6ede1610c0074b2c8e9a23c2c52a8d4d`, checkpoint 244 (999,424 env steps), eval seed 10002, stochastic). Situation: place stone (rare at the final checkpoint) is unlocked at step 188; window steps 38 to 188, at its start health 9, food 8, drink 8, energy 8, inventory wood 1. Behaviour: most chosen actions do 32, down 26, place table 25; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 2, drink 9, energy 5, inventory empty. Outcome: the episode lasted 319 steps. Replay: `gifs/1m-ppo-rare_unlock-3.gif`, `masters/case-1m-ppo-rare_unlock-3.mkv`, `rollouts/case-1m-ppo-rare_unlock-3.npz`.

**`1m-ppo-rare_unlock-4`** (run `ppo-1m-e06ac6db-s7`, `99a2896e883a4e99a123883904e36be0`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: eat plant (rare at the final checkpoint) is unlocked at step 1302; window steps 1152 to 1302, at its start health 7, food 6, drink 9, energy 8, inventory sapling 4. Behaviour: most chosen actions do 39, down 17, sleep 10; combat damage 6, close calls 1, lowest health 2. At the unlock health 8, food 9, drink 8, energy 9, inventory wood 1, sapling 4. Outcome: the episode lasted 1463 steps. Replay: `gifs/1m-ppo-rare_unlock-4.gif`, `masters/case-1m-ppo-rare_unlock-4.mkv`, `rollouts/case-1m-ppo-rare_unlock-4.npz`.

**`1m-ppo-rare_unlock-5`** (run `ppo-1m-e06ac6db-s4`, `bfa5201f7d6e4aa68b89050c47b4751f`, checkpoint 244 (999,424 env steps), eval seed 10056, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 79; window steps 0 to 79, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 44, place plant 8, down 6; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 7, drink 6, energy 9, inventory wood 1. Outcome: the episode lasted 220 steps. Replay: `gifs/1m-ppo-rare_unlock-5.gif`, `masters/case-1m-ppo-rare_unlock-5.mkv`, `rollouts/case-1m-ppo-rare_unlock-5.npz`.

**`1m-ppo-rare_unlock-6`** (run `ppo-1m-e06ac6db-s3`, `5a49268d67d54599836795356e27437e`, checkpoint 244 (999,424 env steps), eval seed 10038, stochastic). Situation: place furnace (rare at the final checkpoint) is unlocked at step 195; window steps 45 to 195, at its start health 9, food 8, drink 7, energy 8, inventory empty. Behaviour: most chosen actions place table 20, make wood pickaxe 20, right 17; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 2, drink 1, energy 6, inventory wood 1. Outcome: the episode lasted 356 steps. Replay: `gifs/1m-ppo-rare_unlock-6.gif`, `masters/case-1m-ppo-rare_unlock-6.mkv`, `rollouts/case-1m-ppo-rare_unlock-6.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-ppo-death-1`** (run `ppo-1m-e06ac6db-s6`, `05541d512ff6453ab2a6f528f7c365c2`, checkpoint 244 (999,424 env steps), eval seed 10048, stochastic). Situation: death by zombie at step 225; the window (steps 75 to 225) opens with health 9, food 7, drink 7, energy 9, inventory wood 1. Behaviour: most chosen actions do 55, up 24, down 23; combat damage 10, close calls 0, lowest health 0. At death health 0, food 1, drink 9, energy 5, inventory wood 1. Outcome: the episode ended after 225 steps. Replay: `gifs/1m-ppo-death-1.gif`, `masters/case-1m-ppo-death-1.mkv`, `rollouts/case-1m-ppo-death-1.npz`.

**`1m-ppo-death-2`** (run `ppo-1m-e06ac6db-s2`, `078ee719d8614c239d747e54586d44d1`, checkpoint 244 (999,424 env steps), eval seed 10010, stochastic). Situation: death by starvation at step 390; the window (steps 240 to 390) opens with health 8, food 1, drink 0, energy 8, inventory empty. Behaviour: most chosen actions make wood pickaxe 25, place table 14, make wood sword 9; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 9, inventory empty. Outcome: the episode ended after 390 steps. Replay: `gifs/1m-ppo-death-2.gif`, `masters/case-1m-ppo-death-2.mkv`, `rollouts/case-1m-ppo-death-2.npz`.

**`1m-ppo-death-3`** (run `ppo-1m-e06ac6db-s8`, `b4dac6b32a184513b43c4e3f0dd76460`, checkpoint 244 (999,424 env steps), eval seed 10061, stochastic). Situation: death by arrow at step 306; the window (steps 156 to 306) opens with health 9, food 4, drink 8, energy 9, inventory wood 1, sapling 3. Behaviour: most chosen actions do 77, place table 12, right 10; combat damage 8, close calls 0, lowest health 0. At death health 0, food 0, drink 9, energy 6, inventory wood 1, sapling 3. Outcome: the episode ended after 306 steps. Replay: `gifs/1m-ppo-death-3.gif`, `masters/case-1m-ppo-death-3.mkv`, `rollouts/case-1m-ppo-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-ppo`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Plateau | 1M | 7 / 10 | 500k to 999k | score gain -0.10 from 500k to the end (< 0.24) (run 22fa55cebeb140729639e8fb2a40c18b; 6 more) | not analysed (tuning runs, R107) | open at 1M | `outputs/results/figures/curves_score.png` (F1, learning curves) |
| Reward hacking | 1M | not logged: 10 |  | not logged: train/ach_reward_share | - | - | - |
| Exploration failure | 1M | not logged: 10 |  | not logged: first_unlock/ | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 24.6k to 963k | >= 20% to < 5%: collect_drink (81.9k to 135k), eat_cow (430k to 451k), make_wood_pickaxe (496k to 532k), make_wood_sword (471k to 684k) (run 22fa55cebeb140729639e8fb2a40c18b; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |

- Absent in every run: Instability, Collapse, Entropy collapse, Value failure, Numerical failure, Survival failure.
- Detector verdicts over the tuned runs (100 run-detector pairs): absent 63, detected 17, not logged 20; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

### 1B

1B: pending (final-1b not run, R75). The 1B obstacles table (3 seeds) is added by rerunning `analysis.training` once the 1B finals exist.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[PPO training analysis]] (this note): floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PPO-RNN training analysis]]: floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.
- [[PQN-RNN training analysis]]: floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.

## Ablations

No ablation of this method ran in `final-1m`.
