---
title: "Count-bonus PQN-RNN training analysis"
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
# Count-bonus PQN-RNN training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 1.481 (sd 0.375), stochastic eval score 1.469; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%).
- Achievements: 53 of 67 never unlocked by any seed at any checkpoint (basic 12, intermediate 17, advanced 15, very advanced 9), 7 rare (final rate < 5%), 7 common.
- Deaths at the final checkpoint: zombie 50.2%, starvation 30.3%, arrow 15.5%, dehydration 4.1% (the untrained policy: zombie 48.4%, starvation 37.5%, arrow 12.5%, dehydration 1.6%).
- Stalls: 74 stalls in 640 final eval episodes, 0.116 per episode (seed interval 0.095 to 0.138), median length 327 steps, 4.1% escaped by an unlock; they are started by wake up (61%) and place plant (14%).
- Obstacles at 1M: instability 10/10, collapse 3/10, bonus distraction 3/10, hoarding or key cycling 5/10, early-race slowdown 7/10; not logged (R78): value failure; uninformative at 1M: bonus starvation. Caveat: skill forgetting 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m`, Count-bonus PQN-RNN (count-bonus variant on the PQN-RNN control), 10 seeds (0-9) of the tuned config; method: [[Count-Based Progress Bonus design]], plan [[2026-10-20-craftax-method-count-bonus]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/count-bonus-pqn-rnn/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-count-bonus-pqn-rnn`, id `72145fdfab084517b4f5f84e6543f4a0`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 16 case episodes, all replayed from the stored eval files.
- Related analyses: [[Count-bonus PPO-RNN training analysis]] (the other variant).
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 3 of this method's 10 tuned runs (3 of 10 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 92, detected 38, not logged 10, uninformative at 1M 10; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Floor entries before or beside the final eval: `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%). Source: `training-analysis-1m-count-bonus-pqn-rnn`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (53 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (12): place stone, eat plant, make stone pickaxe, make stone sword, collect coal, collect iron, collect diamond, make iron pickaxe, make iron sword, make arrow, make torch, place torch; intermediate (17): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-count-bonus-pqn-rnn`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 7): defeat zombie 3.44% (10/10 seeds ever), make wood sword 0.62% (9/10 seeds ever), make wood pickaxe 0.31% (10/10 seeds ever), defeat skeleton 0.16% (10/10 seeds ever), collect stone 0.00% (2/10 seeds ever), place furnace 0.00% (1/10 seeds ever), enter dungeon 0.00% (1/10 seeds ever). Same file.
- Common (7): collect sapling 89.8%, wake up 89.2%, collect wood 77.5%, place plant 76.7%, place table 40.0%, collect drink 31.6%, eat cow 12.7%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 321 (50.2%); starvation 194 (30.3%); arrow 99 (15.5%); dehydration 26 (4.1%). At checkpoint 0 (untrained): zombie 48.4%, starvation 37.5%, arrow 12.5%, dehydration 1.6%. Source: `training-analysis-1m-count-bonus-pqn-rnn`, `hard/death_causes.csv`, plot `plots/death_causes_over_training.png`.
- Stalls (>= 300 steps without an unlock): 74 stalls in 640 final eval episodes, 0.116 per episode (seed interval 0.095 to 0.138), median length 327 steps, 4.1% escaped by an unlock. All started on floor 0; started by wake up 45 (60.8%), place plant 10 (13.5%), collect drink 8 (10.8%), eat cow 5 (6.8%). Source: `training-analysis-1m-count-bonus-pqn-rnn`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 3 by an unlock, 71 by death. Longest stall 397 steps (run `3131542c7fcf4dc99a1c6e7efe0fde8b`, seed 9, eval seed 10014, steps 95 to 492, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-count-bonus-pqn-rnn` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 74 stalls, 3 ended by an unlock, 71 by death (95.9%).
- Before the unlock (6 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 5 cases; most frequent actions right 22%, do 18%, down 17%; movement keys are 45% of the last 50 steps against 37% of the whole stall window, `do` 18% against 20%; the most frequent action of the last 50 differs from the window's in 3 of 6 cases.
- State at the unlock: health <= 4 in 6 of 6 cases (median health 1 at the unlock, 9 at the stall start), food 0 in 6 of 6 (median food 0, 9 at the start), drink 0 in 4 of 6.
- Achievements that ended the stalls: wake up 4, place table 1, collect sapling 1.
- Deepest floor: the deep-floor cases are all overworld (floor 0) episodes, so the route question is moot at 1M.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-count-bonus-pqn-rnn-stall-1`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s4`, `71cc64d4d10945d89dc87437b1524003`, checkpoint 244 (999,424 env steps), eval seed 10050, stochastic). Situation: a stall of 352 steps from step 2 on floor 0, begun after collect wood; at its start health 9, food 9, drink 9, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions up 143, right 112, do 4 over the 352 steps; 9.77% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: wake up at step 354 (then health 2, food 0, drink 0, energy 9, inventory wood 4). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-stall-1.gif`, `overlays/1m-count-bonus-pqn-rnn-stall-1.mp4`, `masters/case-1m-count-bonus-pqn-rnn-stall-1.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-stall-1.npz`.

**`1m-count-bonus-pqn-rnn-stall-2`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s4`, `71cc64d4d10945d89dc87437b1524003`, checkpoint 244 (999,424 env steps), eval seed 10030, stochastic). Situation: a stall of 335 steps from step 141 on floor 0, begun after eat cow; at its start health 9, food 9, drink 4, energy 5, inventory wood 1, sapling 1. Behaviour during the stall: most chosen actions do 95, right 67, ascend 29 over the 335 steps; 0.09% of overworld tiles first seen in the window; combat damage 0, close calls 1, lowest health 1. What broke it: wake up at step 476 (then health 1, food 0, drink 7, energy 9, inventory wood 1, sapling 1). Outcome: the episode lasted 478 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-stall-2.gif`, `overlays/1m-count-bonus-pqn-rnn-stall-2.mp4`, `masters/case-1m-count-bonus-pqn-rnn-stall-2.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-stall-2.npz`.

**`1m-count-bonus-pqn-rnn-stall-3`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s0`, `78a37cd1d11148b68617086a086a1c53`, checkpoint 244 (999,424 env steps), eval seed 10027, stochastic). Situation: a stall of 305 steps from step 88 on floor 0, begun after eat cow; at its start health 9, food 9, drink 8, energy 9, inventory wood 1, sapling 1. Behaviour during the stall: most chosen actions do 203, rest 47, make iron pickaxe 17 over the 305 steps; 0.00% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 393 (then health 1, food 0, drink 9, energy 0, inventory sapling 1). Outcome: the episode lasted 408 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-stall-3.gif`, `overlays/1m-count-bonus-pqn-rnn-stall-3.mp4`, `masters/case-1m-count-bonus-pqn-rnn-stall-3.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-stall-3.npz`.

**`1m-count-bonus-pqn-rnn-stall-4`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s3`, `066fa6b5b00d495589eb5c4fdd52edb3`, checkpoint 0 (0 env steps), eval seed 10044, stochastic). Situation: a stall of 303 steps from step 43 on floor 0; at its start health 9, food 8, drink 8, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions place stone 11, make arrow 11, make torch 11 over the 303 steps; 2.21% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: collect sapling at step 346 (then health 3, food 0, drink 0, energy 9, inventory wood 1, sapling 1). Outcome: the episode lasted 391 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-stall-4.gif`, `overlays/1m-count-bonus-pqn-rnn-stall-4.mp4`, `masters/case-1m-count-bonus-pqn-rnn-stall-4.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-stall-4.npz`.

**`1m-count-bonus-pqn-rnn-stall-5`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s1`, `8d2155b4b14b41dba7887066f5b7bd6c`, checkpoint 12 (49,152 env steps), eval seed 10031, stochastic). Situation: a stall of 368 steps from step 0 on floor 0; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions place table 156, left 7, right 6 over the 368 steps; 2.65% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: wake up at step 368 (then health 1, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-stall-5.gif`, `overlays/1m-count-bonus-pqn-rnn-stall-5.mp4`, `masters/case-1m-count-bonus-pqn-rnn-stall-5.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-stall-5.npz`.

**`1m-count-bonus-pqn-rnn-stall-6`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s3`, `066fa6b5b00d495589eb5c4fdd52edb3`, checkpoint 148 (606,208 env steps), eval seed 10030, stochastic). Situation: a stall of 318 steps from step 50 on floor 0; at its start health 9, food 8, drink 7, energy 8, inventory empty. Behaviour during the stall: most chosen actions down 196, descend 25, do 4 over the 318 steps; 2.30% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: wake up at step 368 (then health 1, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-stall-6.gif`, `overlays/1m-count-bonus-pqn-rnn-stall-6.mp4`, `masters/case-1m-count-bonus-pqn-rnn-stall-6.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-stall-6.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-count-bonus-pqn-rnn-deep_floor-1`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s7`, `7380b98f0fd042b7b63c527d117d8c31`, checkpoint 244 (999,424 env steps), eval seed 10015, stochastic). Situation: an episode with 8 achievements unlocked on floor 0; window steps 84 to 234, at its start health 9, food 6, drink 6, energy 9, inventory wood 1. Behaviour: most chosen actions do 66, place table 36, left 19; 4.73% of overworld tiles first seen in the window; combat damage 4, close calls 0, lowest health 6. At step 234 (the last unlock) health 6, food 8, drink 1, energy 6, inventory wood 1, sapling 6. Outcome: the episode lasted 241 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-deep_floor-1.gif`, `masters/case-1m-count-bonus-pqn-rnn-deep_floor-1.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-deep_floor-1.npz`.

**`1m-count-bonus-pqn-rnn-deep_floor-2`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s3`, `066fa6b5b00d495589eb5c4fdd52edb3`, checkpoint 244 (999,424 env steps), eval seed 10032, stochastic). Situation: an episode with 7 achievements unlocked on floor 0; window steps 0 to 68, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 15, sleep 14, left 11; 7.86% of overworld tiles first seen in the window; combat damage 2, close calls 0, lowest health 7. At step 68 (the last unlock) health 8, food 9, drink 8, energy 9, inventory empty. Outcome: the episode lasted 88 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-deep_floor-2.gif`, `masters/case-1m-count-bonus-pqn-rnn-deep_floor-2.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-deep_floor-2.npz`.

**`1m-count-bonus-pqn-rnn-deep_floor-3`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s7`, `7380b98f0fd042b7b63c527d117d8c31`, checkpoint 244 (999,424 env steps), eval seed 10017, stochastic). Situation: an episode with 7 achievements unlocked on floor 0; window steps 84 to 234, at its start health 9, food 8, drink 6, energy 8, inventory wood 1. Behaviour: most chosen actions do 40, left 36, right 28; 4.82% of overworld tiles first seen in the window; combat damage 4, close calls 0, lowest health 5. At step 234 (the last unlock) health 5, food 3, drink 1, energy 5, inventory empty. Outcome: the episode lasted 284 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-deep_floor-3.gif`, `masters/case-1m-count-bonus-pqn-rnn-deep_floor-3.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-count-bonus-pqn-rnn-rare_unlock-1`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s4`, `71cc64d4d10945d89dc87437b1524003`, checkpoint 244 (999,424 env steps), eval seed 10061, stochastic). Situation: make wood pickaxe (rare at the final checkpoint) is unlocked at step 52; window steps 0 to 52, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions up 21, right 18, do 9; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 7, drink 8, energy 8, inventory wood 2, sapling 1. Outcome: the episode lasted 183 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-rare_unlock-1.gif`, `masters/case-1m-count-bonus-pqn-rnn-rare_unlock-1.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-rare_unlock-1.npz`.

**`1m-count-bonus-pqn-rnn-rare_unlock-2`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s3`, `066fa6b5b00d495589eb5c4fdd52edb3`, checkpoint 244 (999,424 env steps), eval seed 10011, stochastic). Situation: make wood sword (rare at the final checkpoint) is unlocked at step 60; window steps 0 to 60, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions left 12, do 10, sleep 7; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 7, drink 7, energy 9, inventory empty. Outcome: the episode lasted 320 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-rare_unlock-2.gif`, `masters/case-1m-count-bonus-pqn-rnn-rare_unlock-2.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-rare_unlock-2.npz`.

**`1m-count-bonus-pqn-rnn-rare_unlock-3`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s8`, `e0f1196239f645f887f827cc0182db0a`, checkpoint 244 (999,424 env steps), eval seed 10024, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 25; window steps 0 to 25, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions up 18, do 7; combat damage 6, close calls 0, lowest health 3. At the unlock health 3, food 9, drink 8, energy 9, inventory wood 1, sapling 1. Outcome: the episode lasted 67 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-rare_unlock-3.gif`, `masters/case-1m-count-bonus-pqn-rnn-rare_unlock-3.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-rare_unlock-3.npz`.

**`1m-count-bonus-pqn-rnn-rare_unlock-4`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s7`, `7380b98f0fd042b7b63c527d117d8c31`, checkpoint 244 (999,424 env steps), eval seed 10015, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 197; window steps 47 to 197, at its start health 9, food 8, drink 7, energy 8, inventory wood 1. Behaviour: most chosen actions place table 54, do 42, right 13; combat damage 4, close calls 0, lowest health 7. At the unlock health 7, food 2, drink 1, energy 7, inventory sapling 3. Outcome: the episode lasted 241 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-rare_unlock-4.gif`, `masters/case-1m-count-bonus-pqn-rnn-rare_unlock-4.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-rare_unlock-4.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-count-bonus-pqn-rnn-death-1`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s0`, `78a37cd1d11148b68617086a086a1c53`, checkpoint 244 (999,424 env steps), eval seed 10011, stochastic). Situation: death by zombie at step 235; the window (steps 85 to 235) opens with health 9, food 7, drink 6, energy 8, inventory empty. Behaviour: most chosen actions up 31, down 23, sleep 22; combat damage 9, close calls 0, lowest health 0. At death health 0, food 1, drink 0, energy 8, inventory sapling 2. Outcome: the episode ended after 235 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-death-1.gif`, `masters/case-1m-count-bonus-pqn-rnn-death-1.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-death-1.npz`.

**`1m-count-bonus-pqn-rnn-death-2`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s3`, `066fa6b5b00d495589eb5c4fdd52edb3`, checkpoint 244 (999,424 env steps), eval seed 10014, stochastic). Situation: death by starvation at step 361; the window (steps 211 to 361) opens with health 9, food 2, drink 0, energy 5, inventory wood 2. Behaviour: most chosen actions down 43, up 42, place plant 32; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 1, energy 0, inventory wood 2. Outcome: the episode ended after 361 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-death-2.gif`, `masters/case-1m-count-bonus-pqn-rnn-death-2.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-death-2.npz`.

**`1m-count-bonus-pqn-rnn-death-3`** (run `count-bonus-pqn-rnn-1m-e6e1fafa-s4`, `71cc64d4d10945d89dc87437b1524003`, checkpoint 244 (999,424 env steps), eval seed 10019, stochastic). Situation: death by arrow at step 284; the window (steps 134 to 284) opens with health 9, food 5, drink 4, energy 9, inventory sapling 2. Behaviour: most chosen actions do 38, cast fireball 35, enchant armour 22; combat damage 6, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 7, inventory sapling 2. Outcome: the episode ended after 284 steps. Replay: `gifs/1m-count-bonus-pqn-rnn-death-3.gif`, `masters/case-1m-count-bonus-pqn-rnn-death-3.mkv`, `rollouts/case-1m-count-bonus-pqn-rnn-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-count-bonus-pqn-rnn`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 10 / 10 | 16.4k to 848k | 2 drops, deepest drawdown 1.26 (floor 0.49) (run 78a37cd1d11148b68617086a086a1c53; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-pqn-rnn-stall-3.gif` |
| Collapse | 1M | 3 / 10 | 16.4k to 999k | drawdown 1.26, final 1.72 of best 1.96 (floor 0.49) (run 78a37cd1d11148b68617086a086a1c53; 2 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-pqn-rnn-stall-3.gif` |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 8.19k to 999k | >= 20% to < 5%: collect_drink (45.1k to 53.2k), collect_sapling (12.3k to 20.5k), collect_wood (12.3k to 180k), eat_cow (848k to 930k), place_plant (12.3k to 20.5k), place_table (77.8k to 86k) (run 78a37cd1d11148b68617086a086a1c53; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status.png` |
| Bonus distraction | 1M | 3 / 10 | 209k to 717k | count/r_int_share > 0.5 with a flat score from 209k (run 71cc64d4d10945d89dc87437b1524003; 2 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-pqn-rnn-stall-1.gif` |
| Hoarding or key cycling | 1M | 5 / 10 | 999k to 999k | key cycling: action cycle (period 2, up to 115 steps) in 7 / 64 final episodes (run 066fa6b5b00d495589eb5c4fdd52edb3; 4 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-pqn-rnn-stall-4.gif` |
| Bonus starvation | 1M | uninformative at 1M: 10 |  | rule assumes 1B scale | - | - | - |
| Early-race slowdown | 1M | 7 / 10 | 1.28k to 980k | first unlocks later than the base control's: make_wood_sword x10.3, defeat_skeleton x2.1 (base median over 7 of 10 runs; 3 not logged) (run 78a37cd1d11148b68617086a086a1c53; 6 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status.png` |

- Absent in every run: Plateau, Entropy collapse, Numerical failure, Reward hacking, Exploration failure, Survival failure, Q overestimation.
- Detector verdicts over the tuned runs (150 run-detector pairs): absent 92, detected 38, not logged 10, uninformative at 1M 10; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[Count-bonus PQN-RNN training analysis]] (this note): floors - floor 0 only; achievements - 53 never, 7 rare, 7 common; killer and stalls - zombie 50.2% of deaths, 0.116 stalls per episode (median 327 steps, 4.1% escaped); final score 1.481.
- [[PQN-RNN training analysis]] (base control): floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.
- [[PPO training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PPO-RNN training analysis]]: floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.

## Ablations

No ablation of this method ran in `final-1m`.
