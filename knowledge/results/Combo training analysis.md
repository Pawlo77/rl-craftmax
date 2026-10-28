---
title: "Combo training analysis"
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
# Combo training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 2.132 (sd 0.093), stochastic eval score 2.144; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes.
- Achievements: 48 of 67 never unlocked by any seed at any checkpoint (basic 6, intermediate 18, advanced 15, very advanced 9), 10 rare (final rate < 5%), 9 common.
- Deaths at the final checkpoint: zombie 57.5%, starvation 27.7%, arrow 13.4%, dehydration 1.4% (the untrained policy: zombie 46.4%, starvation 35.8%, arrow 17.8%).
- Stalls: 54 stalls in 640 final eval episodes, 0.084 per episode (seed interval 0.067 to 0.106), median length 338 steps, 11.1% escaped by an unlock; they are started by wake up (44%) and collect drink (20%).
- Obstacles at 1M: plateau 3/10, overfitting to early lucky windows 10/10, forgetting 3/10; not logged (R78): early-race slowdown; uninformative at 1M: bonus starvation. Caveat: skill forgetting 10/10, insertion overflow 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m` and `combo-1m`, Combo (count-bonus-ppo-rnn plus sil on PPO-RNN (config fd6c4333, [[Experiment decisions]] 2026-10-24)), 10 seeds (0-9) of the tuned config; method: [[Count-Based Progress Bonus design]], [[Self-Imitation Learning design]], plan [[2026-10-20-craftax-method-ppo-rnn]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/combo/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-combo`, id `b75bca440ece446fa3fa7bcfeb2cbcfd`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 21 case episodes, all replayed from the stored eval files.
- Related analyses: [[Count-bonus PPO-RNN training analysis]] and [[SIL training analysis]] (its two components).
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- 1B: pending (final-1b not run, R75); sections marked 1B below are placeholders for the same tables at 1B.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 0 of this method's 10 tuned runs (0 of 10 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 144, detected 36, not logged 10, uninformative at 1M 10; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Source: `training-analysis-1m-combo`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (48 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (6): collect iron, collect diamond, make iron pickaxe, make iron sword, make torch, place torch; intermediate (18): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, enter dungeon, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-combo`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 10): collect stone 2.50% (10/10 seeds ever), defeat zombie 1.09% (10/10 seeds ever), place stone 0.94% (10/10 seeds ever), place furnace 0.94% (10/10 seeds ever), defeat skeleton 0.47% (10/10 seeds ever), collect coal 0.47% (8/10 seeds ever), eat plant 0.00% (5/10 seeds ever), make stone pickaxe 0.00% (2/10 seeds ever), make stone sword 0.00% (2/10 seeds ever), make arrow 0.00% (3/10 seeds ever). Same file.
- Common (9): collect sapling 99.1%, wake up 97.7%, collect wood 94.2%, place plant 92.8%, place table 80.8%, collect drink 55.6%, make wood pickaxe 20.8%, make wood sword 19.7%, eat cow 7.5%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 368 (57.5%); starvation 177 (27.7%); arrow 86 (13.4%); dehydration 9 (1.4%). At checkpoint 0 (untrained): zombie 46.4%, starvation 35.8%, arrow 17.8%. Source: `training-analysis-1m-combo`, `hard/death_causes.csv`, plot `plots/death_causes_over_training.png`.
- Stalls (>= 300 steps without an unlock): 54 stalls in 640 final eval episodes, 0.084 per episode (seed interval 0.067 to 0.106), median length 338 steps, 11.1% escaped by an unlock. All started on floor 0; started by wake up 24 (44.4%), collect drink 11 (20.4%), place table 11 (20.4%), eat cow 5 (9.3%). Source: `training-analysis-1m-combo`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 6 by an unlock, 48 by death. Longest stall 464 steps (run `34d96b84949343d18b5a4fef4772a53e`, seed 8, eval seed 10008, steps 329 to 793, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-combo` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 54 stalls, 6 ended by an unlock, 48 by death (88.9%).
- Before the unlock (8 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 1 cases; most frequent actions do 14%, place table 13%, up 8%; movement keys are 26% of the last 50 steps against 24% of the whole stall window, `do` 14% against 18%; the most frequent action of the last 50 differs from the window's in 4 of 8 cases.
- State at the unlock: health <= 4 in 7 of 8 cases (median health 2 at the unlock, 9 at the stall start), food 0 in 6 of 8 (median food 0, 8 at the start), drink 0 in 3 of 8.
- Achievements that ended the stalls: place table 2, collect wood 2, collect sapling 2, eat cow 1, eat plant 1.
- Deepest floor: the deep-floor cases are all overworld (floor 0) episodes, so the route question is moot at 1M.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-combo-stall-1`** (run `combo-1m-fd6c4333-s8`, `34d96b84949343d18b5a4fef4772a53e`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: a stall of 354 steps from step 50 on floor 0, begun after wake up; at its start health 9, food 8, drink 9, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions do 56, place table 37, make wood pickaxe 34 over the 354 steps; 0.56% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 404 (then health 1, food 0, drink 9, energy 6, inventory sapling 2). Outcome: the episode lasted 419 steps. Replay: `gifs/1m-combo-stall-1.gif`, `overlays/1m-combo-stall-1.mp4`, `masters/case-1m-combo-stall-1.mkv`, `rollouts/case-1m-combo-stall-1.npz`.

**`1m-combo-stall-2`** (run `combo-1m-fd6c4333-s4`, `2eeca4ede8df4fc3a4d5ae7ebcf18ce8`, checkpoint 244 (999,424 env steps), eval seed 10062, stochastic). Situation: a stall of 347 steps from step 50 on floor 0, begun after place table; at its start health 9, food 8, drink 9, energy 9, inventory sapling 3. Behaviour during the stall: most chosen actions do 73, place table 28, up 26 over the 347 steps; 5.08% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: eat cow at step 397 (then health 3, food 6, drink 9, energy 9, inventory wood 1, sapling 3). Outcome: the episode lasted 658 steps. Replay: `gifs/1m-combo-stall-2.gif`, `overlays/1m-combo-stall-2.mp4`, `masters/case-1m-combo-stall-2.mkv`, `rollouts/case-1m-combo-stall-2.npz`.

**`1m-combo-stall-3`** (run `combo-1m-fd6c4333-s4`, `2eeca4ede8df4fc3a4d5ae7ebcf18ce8`, checkpoint 244 (999,424 env steps), eval seed 10025, stochastic). Situation: a stall of 345 steps from step 43 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory sapling 1. Behaviour during the stall: most chosen actions sleep 96, do 44, up 34 over the 345 steps; 0.39% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 388 (then health 1, food 0, drink 0, energy 8, inventory wood 1, sapling 1). Outcome: the episode lasted 405 steps. Replay: `gifs/1m-combo-stall-3.gif`, `overlays/1m-combo-stall-3.mp4`, `masters/case-1m-combo-stall-3.mkv`, `rollouts/case-1m-combo-stall-3.npz`.

**`1m-combo-stall-4`** (run `combo-1m-fd6c4333-s9`, `2aa286643b2e447193caace70c2638f8`, checkpoint 244 (999,424 env steps), eval seed 10042, stochastic). Situation: a stall of 338 steps from step 43 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions sleep 76, do 39, place table 24 over the 338 steps; 1.04% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 381 (then health 1, food 0, drink 0, energy 9, inventory sapling 1). Outcome: the episode lasted 392 steps. Replay: `gifs/1m-combo-stall-4.gif`, `overlays/1m-combo-stall-4.mp4`, `masters/case-1m-combo-stall-4.mkv`, `rollouts/case-1m-combo-stall-4.npz`.

**`1m-combo-stall-5`** (run `combo-1m-fd6c4333-s6`, `affb3f1022874715a2e323a0cccdb6a6`, checkpoint 244 (999,424 env steps), eval seed 10008, stochastic). Situation: a stall of 321 steps from step 43 on floor 0, begun after wake up; at its start health 9, food 8, drink 9, energy 9, inventory sapling 2. Behaviour during the stall: most chosen actions do 52, up 40, down 32 over the 321 steps; 7.29% of overworld tiles first seen in the window; combat damage 2, close calls 0, lowest health 4. What broke it: collect wood at step 364 (then health 4, food 0, drink 7, energy 9, inventory wood 1, sapling 8). Outcome: the episode lasted 426 steps. Replay: `gifs/1m-combo-stall-5.gif`, `overlays/1m-combo-stall-5.mp4`, `masters/case-1m-combo-stall-5.mkv`, `rollouts/case-1m-combo-stall-5.npz`.

**`1m-combo-stall-6`** (run `combo-1m-fd6c4333-s0`, `5857ac97f2014a92a241dddced93d1a8`, checkpoint 2 (8,192 env steps), eval seed 10025, stochastic). Situation: a stall of 314 steps from step 97 on floor 0; at its start health 9, food 6, drink 7, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 13, make diamond armour 12, noop 11 over the 314 steps; 0.87% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect sapling at step 411 (then health 1, food 0, drink 0, energy 9, inventory sapling 1). Outcome: the episode lasted 416 steps. Replay: `gifs/1m-combo-stall-6.gif`, `overlays/1m-combo-stall-6.mp4`, `masters/case-1m-combo-stall-6.mkv`, `rollouts/case-1m-combo-stall-6.npz`.

**`1m-combo-stall-7`** (run `combo-1m-fd6c4333-s9`, `2aa286643b2e447193caace70c2638f8`, checkpoint 12 (49,152 env steps), eval seed 10027, stochastic). Situation: a stall of 314 steps from step 57 on floor 0; at its start health 9, food 8, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 33, place plant 20, up 18 over the 314 steps; 1.17% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 4. What broke it: collect sapling at step 371 (then health 4, food 0, drink 6, energy 8, inventory sapling 1). Outcome: the episode lasted 448 steps. Replay: `gifs/1m-combo-stall-7.gif`, `overlays/1m-combo-stall-7.mp4`, `masters/case-1m-combo-stall-7.mkv`, `rollouts/case-1m-combo-stall-7.npz`.

**`1m-combo-stall-8`** (run `combo-1m-fd6c4333-s3`, `eb6bf403f40241e988ac834d59a83832`, checkpoint 148 (606,208 env steps), eval seed 10042, stochastic). Situation: a stall of 517 steps from step 423 on floor 0; at its start health 1, food 6, drink 9, energy 7, inventory empty. Behaviour during the stall: most chosen actions place table 105, do 91, sleep 31 over the 517 steps; 0.00% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: eat plant at step 940 (then health 6, food 4, drink 9, energy 9, inventory sapling 1). Outcome: the episode lasted 1391 steps. Replay: `gifs/1m-combo-stall-8.gif`, `overlays/1m-combo-stall-8.mp4`, `masters/case-1m-combo-stall-8.mkv`, `rollouts/case-1m-combo-stall-8.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-combo-deep_floor-1`** (run `combo-1m-fd6c4333-s8`, `34d96b84949343d18b5a4fef4772a53e`, checkpoint 244 (999,424 env steps), eval seed 10061, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 274 to 424, at its start health 9, food 0, drink 8, energy 9, inventory wood 1, sapling 1. Behaviour: most chosen actions left 28, right 21, do 21; 1.30% of overworld tiles first seen in the window; combat damage 4, close calls 0, lowest health 5. At step 424 (the last unlock) health 5, food 8, drink 8, energy 8, inventory wood 1. Outcome: the episode lasted 567 steps. Replay: `gifs/1m-combo-deep_floor-1.gif`, `masters/case-1m-combo-deep_floor-1.mkv`, `rollouts/case-1m-combo-deep_floor-1.npz`.

**`1m-combo-deep_floor-2`** (run `combo-1m-fd6c4333-s0`, `5857ac97f2014a92a241dddced93d1a8`, checkpoint 244 (999,424 env steps), eval seed 10016, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 441 to 591, at its start health 6, food 5, drink 5, energy 9, inventory stone 1, sapling 2. Behaviour: most chosen actions do 11, make wood pickaxe 9, make wood sword 6; 0.22% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 6. At step 591 (the last unlock) health 7, food 0, drink 0, energy 9, inventory sapling 2. Outcome: the episode lasted 620 steps. Replay: `gifs/1m-combo-deep_floor-2.gif`, `masters/case-1m-combo-deep_floor-2.mkv`, `rollouts/case-1m-combo-deep_floor-2.npz`.

**`1m-combo-deep_floor-3`** (run `combo-1m-fd6c4333-s6`, `affb3f1022874715a2e323a0cccdb6a6`, checkpoint 244 (999,424 env steps), eval seed 10005, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 197 to 347, at its start health 9, food 3, drink 1, energy 8, inventory wood 1, sapling 1. Behaviour: most chosen actions right 13, make wood pickaxe 12, do 10; 4.69% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. At step 347 (the last unlock) health 2, food 0, drink 0, energy 9, inventory coal 1, sapling 1. Outcome: the episode lasted 378 steps. Replay: `gifs/1m-combo-deep_floor-3.gif`, `masters/case-1m-combo-deep_floor-3.mkv`, `rollouts/case-1m-combo-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-combo-rare_unlock-1`** (run `combo-1m-fd6c4333-s2`, `35613fb3bca74b4c89b0396c8f6a1a4a`, checkpoint 244 (999,424 env steps), eval seed 10043, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 208; window steps 58 to 208, at its start health 9, food 7, drink 7, energy 9, inventory wood 1, sapling 2. Behaviour: most chosen actions make wood pickaxe 35, place table 21, do 17; combat damage 8, close calls 0, lowest health 1. At the unlock health 1, food 2, drink 0, energy 6, inventory sapling 2. Outcome: the episode lasted 214 steps. Replay: `gifs/1m-combo-rare_unlock-1.gif`, `masters/case-1m-combo-rare_unlock-1.mkv`, `rollouts/case-1m-combo-rare_unlock-1.npz`.

**`1m-combo-rare_unlock-2`** (run `combo-1m-fd6c4333-s5`, `3c5adf35a6be4ae8b1e9d6fc6bc16976`, checkpoint 244 (999,424 env steps), eval seed 10033, stochastic). Situation: collect stone (rare at the final checkpoint) is unlocked at step 78; window steps 0 to 78, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 33, place table 8, left 6; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 7, drink 6, energy 9, inventory stone 1, sapling 2. Outcome: the episode lasted 314 steps. Replay: `gifs/1m-combo-rare_unlock-2.gif`, `masters/case-1m-combo-rare_unlock-2.mkv`, `rollouts/case-1m-combo-rare_unlock-2.npz`.

**`1m-combo-rare_unlock-3`** (run `combo-1m-fd6c4333-s6`, `affb3f1022874715a2e323a0cccdb6a6`, checkpoint 244 (999,424 env steps), eval seed 10011, stochastic). Situation: place stone (rare at the final checkpoint) is unlocked at step 109; window steps 0 to 109, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 25, left 12, place table 12; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 5, drink 5, energy 8, inventory empty. Outcome: the episode lasted 352 steps. Replay: `gifs/1m-combo-rare_unlock-3.gif`, `masters/case-1m-combo-rare_unlock-3.mkv`, `rollouts/case-1m-combo-rare_unlock-3.npz`.

**`1m-combo-rare_unlock-4`** (run `combo-1m-fd6c4333-s3`, `eb6bf403f40241e988ac834d59a83832`, checkpoint 148 (606,208 env steps), eval seed 10042, stochastic). Situation: eat plant (rare at the final checkpoint) is unlocked at step 940; window steps 790 to 940, at its start health 9, food 3, drink 9, energy 8, inventory sapling 1. Behaviour: most chosen actions place table 33, do 20, sleep 14; combat damage 0, close calls 0, lowest health 6. At the unlock health 6, food 4, drink 9, energy 9, inventory sapling 1. Outcome: the episode lasted 1391 steps. Replay: `gifs/1m-combo-rare_unlock-4.gif`, `masters/case-1m-combo-rare_unlock-4.mkv`, `rollouts/case-1m-combo-rare_unlock-4.npz`.

**`1m-combo-rare_unlock-5`** (run `combo-1m-fd6c4333-s7`, `c3cee73407a5483cba125e6116e22b0c`, checkpoint 244 (999,424 env steps), eval seed 10060, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 149; window steps 0 to 149, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 49, left 20, place table 17; combat damage 4, close calls 0, lowest health 6. At the unlock health 6, food 4, drink 7, energy 8, inventory sapling 4. Outcome: the episode lasted 330 steps. Replay: `gifs/1m-combo-rare_unlock-5.gif`, `masters/case-1m-combo-rare_unlock-5.mkv`, `rollouts/case-1m-combo-rare_unlock-5.npz`.

**`1m-combo-rare_unlock-6`** (run `combo-1m-fd6c4333-s5`, `3c5adf35a6be4ae8b1e9d6fc6bc16976`, checkpoint 244 (999,424 env steps), eval seed 10033, stochastic). Situation: place furnace (rare at the final checkpoint) is unlocked at step 120; window steps 0 to 120, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 40, place table 9, make wood pickaxe 9; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 5, drink 4, energy 9, inventory sapling 2. Outcome: the episode lasted 314 steps. Replay: `gifs/1m-combo-rare_unlock-6.gif`, `masters/case-1m-combo-rare_unlock-6.mkv`, `rollouts/case-1m-combo-rare_unlock-6.npz`.

**`1m-combo-rare_unlock-7`** (run `combo-1m-fd6c4333-s6`, `affb3f1022874715a2e323a0cccdb6a6`, checkpoint 244 (999,424 env steps), eval seed 10005, stochastic). Situation: collect coal (rare at the final checkpoint) is unlocked at step 234; window steps 84 to 234, at its start health 9, food 6, drink 6, energy 8, inventory empty. Behaviour: most chosen actions do 27, right 12, place table 12; combat damage 0, close calls 0, lowest health 8. At the unlock health 8, food 1, drink 0, energy 7, inventory wood 1, stone 1, coal 1, sapling 1. Outcome: the episode lasted 378 steps. Replay: `gifs/1m-combo-rare_unlock-7.gif`, `masters/case-1m-combo-rare_unlock-7.mkv`, `rollouts/case-1m-combo-rare_unlock-7.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-combo-death-1`** (run `combo-1m-fd6c4333-s8`, `34d96b84949343d18b5a4fef4772a53e`, checkpoint 244 (999,424 env steps), eval seed 10029, stochastic). Situation: death by zombie at step 231; the window (steps 81 to 231) opens with health 9, food 6, drink 7, energy 7, inventory wood 2. Behaviour: most chosen actions left 45, do 21, up 11; combat damage 10, close calls 0, lowest health 0. At death health 0, food 1, drink 9, energy 2, inventory wood 1. Outcome: the episode ended after 231 steps. Replay: `gifs/1m-combo-death-1.gif`, `masters/case-1m-combo-death-1.mkv`, `rollouts/case-1m-combo-death-1.npz`.

**`1m-combo-death-2`** (run `combo-1m-fd6c4333-s9`, `2aa286643b2e447193caace70c2638f8`, checkpoint 244 (999,424 env steps), eval seed 10010, stochastic). Situation: death by starvation at step 393; the window (steps 243 to 393) opens with health 9, food 1, drink 0, energy 8, inventory wood 1, sapling 1. Behaviour: most chosen actions place table 17, right 10, do 10; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 9, inventory wood 1, sapling 1. Outcome: the episode ended after 393 steps. Replay: `gifs/1m-combo-death-2.gif`, `masters/case-1m-combo-death-2.mkv`, `rollouts/case-1m-combo-death-2.npz`.

**`1m-combo-death-3`** (run `combo-1m-fd6c4333-s8`, `34d96b84949343d18b5a4fef4772a53e`, checkpoint 244 (999,424 env steps), eval seed 10054, stochastic). Situation: death by arrow at step 302; the window (steps 152 to 302) opens with health 9, food 4, drink 9, energy 8, inventory wood 1, sapling 3. Behaviour: most chosen actions do 33, right 26, left 19; combat damage 6, close calls 0, lowest health 0. At death health 0, food 0, drink 9, energy 3, inventory wood 1, sapling 3. Outcome: the episode ended after 302 steps. Replay: `gifs/1m-combo-death-3.gif`, `masters/case-1m-combo-death-3.mkv`, `rollouts/case-1m-combo-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, `combo-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-combo`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting, insertion overflow fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Plateau | 1M | 3 / 10 | 500k to 999k | score gain 0.05 from 500k to the end (< 0.23) (run 7f319d3651064571a83dab0bc4a42808; 2 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-combo-death-1.gif` |
| Skill forgetting | 1M | 10 / 10 | 16.4k to 975k | >= 20% to < 5%: collect_drink (41k to 53.2k), eat_cow (610k to 618k), make_wood_pickaxe (557k to 676k), make_wood_sword (721k to 975k) (run 5857ac97f2014a92a241dddced93d1a8; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status.png` |
| Bonus starvation | 1M | uninformative at 1M: 10 |  | rule assumes 1B scale | - | - | - |
| Early-race slowdown | 1M | not logged: 10 |  | base control ppo-rnn lacks first_unlock/ | - | - | - |
| Overfitting to early lucky windows | 1M | 10 / 10 | 20.5k to 999k | sil/sample_age_max > 50% of the updates, sil/active > 0.5, entropy below the base control's from 324k (8 stretches) (run 5857ac97f2014a92a241dddced93d1a8; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-combo-stall-6.gif` |
| Forgetting | 1M | 3 / 10 | 900k to 1M | basic success rates below the base control's: make_wood_sword -11 points (run 7f319d3651064571a83dab0bc4a42808; 2 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status.png` |
| Insertion overflow | 1M | 10 / 10 | 12.3k to 999k | sil/overflow > 0 in 83% of logged updates (run 5857ac97f2014a92a241dddced93d1a8; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-combo-stall-6.gif` |

- Absent in every run: Instability, Collapse, Entropy collapse, Value failure, Numerical failure, Reward hacking, Exploration failure, Survival failure, Bonus distraction, Hoarding or key cycling, Imitated unsafe play, Value overestimation, Dead SIL.
- Detector verdicts over the tuned runs (200 run-detector pairs): absent 144, detected 36, not logged 10, uninformative at 1M 10; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

### 1B

1B: pending (final-1b not run, R75). The 1B obstacles table (3 seeds) and `combo-1b` is added by rerunning `analysis.training` once the 1B finals exist.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[Combo training analysis]] (this note): floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 57.5% of deaths, 0.084 stalls per episode (median 338 steps, 11.1% escaped); final score 2.132.
- [[PPO-RNN training analysis]] (base control): floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.
- [[PPO training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PQN-RNN training analysis]]: floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.
- Component [[Count-bonus PPO-RNN training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 58.9% of deaths, 0.094 stalls per episode (median 327.5 steps, 16.7% escaped); final score 2.274.
- Component [[SIL training analysis]]: floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 55.2% of deaths, 0.113 stalls per episode (median 333.5 steps, 18.1% escaped); final score 2.137.

## Ablations

No ablation of this method ran in `final-1m`.
