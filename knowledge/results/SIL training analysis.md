---
title: "SIL training analysis"
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
# SIL training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 2.137 (sd 0.046), stochastic eval score 2.131; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%).
- Achievements: 47 of 67 never unlocked by any seed at any checkpoint (basic 6, intermediate 17, advanced 15, very advanced 9), 11 rare (final rate < 5%), 9 common.
- Deaths at the final checkpoint: zombie 55.2%, starvation 29.7%, arrow 14.1%, dehydration 0.9%, exhaustion 0.2% (the untrained policy: zombie 46.4%, starvation 35.8%, arrow 17.8%).
- Stalls: 72 stalls in 640 final eval episodes, 0.113 per episode (seed interval 0.084 to 0.145), median length 333.5 steps, 18.1% escaped by an unlock; they are started by wake up (47%) and collect drink (22%).
- Obstacles at 1M: instability 2/10, collapse 1/10, plateau 2/10, overfitting to early lucky windows 9/10, forgetting 4/10. Caveat: skill forgetting 10/10, insertion overflow 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m`, SIL (self-imitation on the PPO-RNN control), 10 seeds (0-9) of the tuned config; method: [[Self-Imitation Learning design]], plan [[2026-10-20-craftax-method-sil]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/sil/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-sil`, id `0fc0d981963f4a1b9227298c88add9c1`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 44 case episodes, all replayed from the stored eval files.
- Related analyses: [[Combo training analysis]] (uses SIL as a component).
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 3 of this method's 10 tuned runs (3 of 20 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 246, detected 74; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Floor entries before or beside the final eval: `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%). Source: `training-analysis-1m-sil`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (47 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (6): collect iron, collect diamond, make iron pickaxe, make iron sword, make torch, place torch; intermediate (17): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-sil`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status_tuned.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 11): collect stone 3.59% (10/10 seeds ever), place furnace 1.41% (9/10 seeds ever), place stone 1.09% (10/10 seeds ever), defeat zombie 0.31% (10/10 seeds ever), eat plant 0.31% (8/10 seeds ever), collect coal 0.31% (9/10 seeds ever), defeat skeleton 0.16% (10/10 seeds ever), make arrow 0.16% (3/10 seeds ever), make stone pickaxe 0.00% (2/10 seeds ever), make stone sword 0.00% (3/10 seeds ever), enter dungeon 0.00% (1/10 seeds ever). Same file.
- Common (9): wake up 99.2%, collect sapling 98.0%, collect wood 94.2%, place plant 92.2%, place table 79.5%, collect drink 59.4%, make wood pickaxe 16.4%, make wood sword 15.3%, eat cow 10.2%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 353 (55.2%); starvation 190 (29.7%); arrow 90 (14.1%); dehydration 6 (0.9%); exhaustion 1 (0.2%). At checkpoint 0 (untrained): zombie 46.4%, starvation 35.8%, arrow 17.8%. Source: `training-analysis-1m-sil`, `hard/death_causes.csv`, plot `plots/death_causes_over_training_tuned.png`.
- Stalls (>= 300 steps without an unlock): 72 stalls in 640 final eval episodes, 0.113 per episode (seed interval 0.084 to 0.145), median length 333.5 steps, 18.1% escaped by an unlock. All started on floor 0; started by wake up 34 (47.2%), collect drink 16 (22.2%), eat cow 6 (8.3%), place table 6 (8.3%). Source: `training-analysis-1m-sil`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 13 by an unlock, 59 by death. Longest stall 623 steps (run `f14833a47a6d47978742f496a4ce3821`, seed 2, eval seed 10008, steps 96 to 719, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-sil` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 72 stalls, 13 ended by an unlock, 59 by death (81.9%).
- Before the unlock (8 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 0 cases; most frequent actions do 13%, place table 9%, sleep 6%; movement keys are 22% of the last 50 steps against 27% of the whole stall window, `do` 13% against 14%; the most frequent action of the last 50 differs from the window's in 2 of 8 cases.
- State at the unlock: health <= 4 in 8 of 8 cases (median health 1.5 at the unlock, 9 at the stall start), food 0 in 7 of 8 (median food 0, 7.5 at the start), drink 0 in 3 of 8.
- Achievements that ended the stalls: collect wood 3, place table 1, eat cow 1, make wood sword 1, wake up 1, collect sapling 1.
- Deepest floor: the deep-floor cases are all overworld (floor 0) episodes, so the route question is moot at 1M.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-sil-stall-1`** (run `sil-1m-6f8690ad-s8`, `a75c74baa77543f697ba819708b69882`, checkpoint 244 (999,424 env steps), eval seed 10022, stochastic). Situation: a stall of 377 steps from step 46 on floor 0, begun after wake up; at its start health 9, food 8, drink 9, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions do 41, right 37, left 34 over the 377 steps; 2.08% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 423 (then health 1, food 0, drink 8, energy 8, inventory empty). Outcome: the episode lasted 427 steps. Replay: `gifs/1m-sil-stall-1.gif`, `overlays/1m-sil-stall-1.mp4`, `masters/case-1m-sil-stall-1.mkv`, `rollouts/case-1m-sil-stall-1.npz`.

**`1m-sil-stall-2`** (run `sil-1m-6f8690ad-s5`, `a4f49419e863402e946a7a869e1fa922`, checkpoint 244 (999,424 env steps), eval seed 10042, stochastic). Situation: a stall of 342 steps from step 70 on floor 0, begun after place table; at its start health 9, food 7, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 57, place table 34, down 28 over the 342 steps; 1.95% of overworld tiles first seen in the window; combat damage 0, close calls 1, lowest health 2. What broke it: eat cow at step 412 (then health 2, food 6, drink 8, energy 8, inventory empty). Outcome: the episode lasted 1127 steps. Replay: `gifs/1m-sil-stall-2.gif`, `overlays/1m-sil-stall-2.mp4`, `masters/case-1m-sil-stall-2.mkv`, `rollouts/case-1m-sil-stall-2.npz`.

**`1m-sil-stall-3`** (run `sil-1m-6f8690ad-s7`, `ad672b356a2748028a8b3f7364bf0a82`, checkpoint 244 (999,424 env steps), eval seed 10009, stochastic). Situation: a stall of 327 steps from step 64 on floor 0, begun after place table; at its start health 9, food 7, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions up 106, left 29, do 25 over the 327 steps; 4.38% of overworld tiles first seen in the window; combat damage 6, close calls 0, lowest health 1. What broke it: make wood sword at step 391 (then health 1, food 0, drink 3, energy 9, inventory empty). Outcome: the episode lasted 404 steps. Replay: `gifs/1m-sil-stall-3.gif`, `overlays/1m-sil-stall-3.mp4`, `masters/case-1m-sil-stall-3.mkv`, `rollouts/case-1m-sil-stall-3.npz`.

**`1m-sil-stall-4`** (run `sil-1m-6f8690ad-s0`, `98aadc7012c44416ae6ab215a3ed197a`, checkpoint 244 (999,424 env steps), eval seed 10046, stochastic). Situation: a stall of 326 steps from step 71 on floor 0, begun after wake up; at its start health 9, food 7, drink 9, energy 9, inventory sapling 1. Behaviour during the stall: most chosen actions do 64, place table 51, right 29 over the 326 steps; 3.52% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: collect wood at step 397 (then health 3, food 0, drink 9, energy 8, inventory wood 1, sapling 4). Outcome: the episode lasted 438 steps. Replay: `gifs/1m-sil-stall-4.gif`, `overlays/1m-sil-stall-4.mp4`, `masters/case-1m-sil-stall-4.mkv`, `rollouts/case-1m-sil-stall-4.npz`.

**`1m-sil-stall-5`** (run `sil-1m-6f8690ad-s6`, `0a7c86571c9f4ad681588857143d788a`, checkpoint 244 (999,424 env steps), eval seed 10010, stochastic). Situation: a stall of 323 steps from step 53 on floor 0, begun after wake up; at its start health 9, food 8, drink 7, energy 9, inventory sapling 1. Behaviour during the stall: most chosen actions do 49, sleep 37, up 22 over the 323 steps; 0.52% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 376 (then health 1, food 0, drink 0, energy 8, inventory wood 1, sapling 1). Outcome: the episode lasted 388 steps. Replay: `gifs/1m-sil-stall-5.gif`, `overlays/1m-sil-stall-5.mp4`, `masters/case-1m-sil-stall-5.mkv`, `rollouts/case-1m-sil-stall-5.npz`.

**`1m-sil-stall-6`** (run `sil-1m-6f8690ad-s1`, `85bc7e9b8ba7401c9edd0ca48d41a620`, checkpoint 2 (8,192 env steps), eval seed 10041, stochastic). Situation: a stall of 355 steps from step 0 on floor 0; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions make wood pickaxe 12, ascend 12, make iron pickaxe 11 over the 355 steps; 3.82% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: wake up at step 355 (then health 2, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-sil-stall-6.gif`, `overlays/1m-sil-stall-6.mp4`, `masters/case-1m-sil-stall-6.mkv`, `rollouts/case-1m-sil-stall-6.npz`.

**`1m-sil-stall-7`** (run `sil-1m-6f8690ad-s7`, `ad672b356a2748028a8b3f7364bf0a82`, checkpoint 12 (49,152 env steps), eval seed 10025, stochastic). Situation: a stall of 314 steps from step 97 on floor 0; at its start health 9, food 6, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 34, up 15, place plant 13 over the 314 steps; 4.51% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: collect sapling at step 411 (then health 2, food 0, drink 6, energy 9, inventory sapling 1). Outcome: the episode lasted 435 steps. Replay: `gifs/1m-sil-stall-7.gif`, `overlays/1m-sil-stall-7.mp4`, `masters/case-1m-sil-stall-7.mkv`, `rollouts/case-1m-sil-stall-7.npz`.

**`1m-sil-stall-8`** (run `sil-1m-6f8690ad-s4`, `7441395a440642209b5b38a827a20db6`, checkpoint 148 (606,208 env steps), eval seed 10042, stochastic). Situation: a stall of 347 steps from step 47 on floor 0; at its start health 9, food 8, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions place table 57, sleep 50, do 24 over the 347 steps; 1.04% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 394 (then health 1, food 0, drink 0, energy 9, inventory wood 1, sapling 2). Outcome: the episode lasted 398 steps. Replay: `gifs/1m-sil-stall-8.gif`, `overlays/1m-sil-stall-8.mp4`, `masters/case-1m-sil-stall-8.mkv`, `rollouts/case-1m-sil-stall-8.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-sil-deep_floor-1`** (run `sil-1m-6f8690ad-s6`, `0a7c86571c9f4ad681588857143d788a`, checkpoint 244 (999,424 env steps), eval seed 10016, stochastic). Situation: an episode with 12 achievements unlocked on floor 0; window steps 233 to 383, at its start health 9, food 2, drink 2, energy 9, inventory sapling 2. Behaviour: most chosen actions do 20, up 16, make wood sword 15; 2.56% of overworld tiles first seen in the window; combat damage 2, close calls 0, lowest health 1. At step 383 (the last unlock) health 1, food 6, drink 0, energy 5, inventory stone 2, sapling 2. Outcome: the episode lasted 385 steps. Replay: `gifs/1m-sil-deep_floor-1.gif`, `masters/case-1m-sil-deep_floor-1.mkv`, `rollouts/case-1m-sil-deep_floor-1.npz`.

**`1m-sil-deep_floor-2`** (run `sil-1m-6f8690ad-s9`, `7d96e1129d1f450d91a91dfe582adc2c`, checkpoint 244 (999,424 env steps), eval seed 10025, stochastic). Situation: an episode with 11 achievements unlocked on floor 0; window steps 81 to 231, at its start health 9, food 7, drink 7, energy 8, inventory sapling 1. Behaviour: most chosen actions do 31, place table 17, make wood pickaxe 6; 1.26% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 9. At step 231 (the last unlock) health 9, food 5, drink 0, energy 6, inventory coal 1, sapling 2. Outcome: the episode lasted 267 steps. Replay: `gifs/1m-sil-deep_floor-2.gif`, `masters/case-1m-sil-deep_floor-2.mkv`, `rollouts/case-1m-sil-deep_floor-2.npz`.

**`1m-sil-deep_floor-3`** (run `sil-1m-6f8690ad-s1`, `85bc7e9b8ba7401c9edd0ca48d41a620`, checkpoint 244 (999,424 env steps), eval seed 10058, stochastic). Situation: an episode with 11 achievements unlocked on floor 0; window steps 214 to 364, at its start health 9, food 2, drink 0, energy 9, inventory sapling 1. Behaviour: most chosen actions left 19, do 17, place table 14; 2.47% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 6. At step 364 (the last unlock) health 6, food 3, drink 0, energy 6, inventory wood 1, sapling 1. Outcome: the episode lasted 410 steps. Replay: `gifs/1m-sil-deep_floor-3.gif`, `masters/case-1m-sil-deep_floor-3.mkv`, `rollouts/case-1m-sil-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-sil-rare_unlock-1`** (run `sil-1m-6f8690ad-s1`, `85bc7e9b8ba7401c9edd0ca48d41a620`, checkpoint 244 (999,424 env steps), eval seed 10006, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 125; window steps 0 to 125, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 36, place table 13, left 12; combat damage 8, close calls 0, lowest health 3. At the unlock health 3, food 5, drink 7, energy 7, inventory empty. Outcome: the episode lasted 146 steps. Replay: `gifs/1m-sil-rare_unlock-1.gif`, `masters/case-1m-sil-rare_unlock-1.mkv`, `rollouts/case-1m-sil-rare_unlock-1.npz`.

**`1m-sil-rare_unlock-2`** (run `sil-1m-6f8690ad-s2`, `f14833a47a6d47978742f496a4ce3821`, checkpoint 244 (999,424 env steps), eval seed 10011, stochastic). Situation: collect stone (rare at the final checkpoint) is unlocked at step 88; window steps 0 to 88, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 21, right 11, down 10; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 6, drink 6, energy 8, inventory stone 1, sapling 2. Outcome: the episode lasted 281 steps. Replay: `gifs/1m-sil-rare_unlock-2.gif`, `masters/case-1m-sil-rare_unlock-2.mkv`, `rollouts/case-1m-sil-rare_unlock-2.npz`.

**`1m-sil-rare_unlock-3`** (run `sil-1m-6f8690ad-s9`, `7d96e1129d1f450d91a91dfe582adc2c`, checkpoint 244 (999,424 env steps), eval seed 10017, stochastic). Situation: place stone (rare at the final checkpoint) is unlocked at step 194; window steps 44 to 194, at its start health 9, food 8, drink 8, energy 8, inventory wood 1, sapling 2. Behaviour: most chosen actions do 34, make wood pickaxe 19, place table 16; combat damage 8, close calls 1, lowest health 2. At the unlock health 4, food 2, drink 1, energy 7, inventory wood 1, stone 3, sapling 4. Outcome: the episode lasted 291 steps. Replay: `gifs/1m-sil-rare_unlock-3.gif`, `masters/case-1m-sil-rare_unlock-3.mkv`, `rollouts/case-1m-sil-rare_unlock-3.npz`.

**`1m-sil-rare_unlock-4`** (run `sil-1m-6f8690ad-s0`, `98aadc7012c44416ae6ab215a3ed197a`, checkpoint 244 (999,424 env steps), eval seed 10014, stochastic). Situation: eat plant (rare at the final checkpoint) is unlocked at step 692; window steps 542 to 692, at its start health 9, food 1, drink 9, energy 8, inventory wood 1, sapling 6. Behaviour: most chosen actions place table 23, do 15, left 13; combat damage 0, close calls 0, lowest health 3. At the unlock health 3, food 4, drink 3, energy 9, inventory wood 1, sapling 6. Outcome: the episode lasted 898 steps. Replay: `gifs/1m-sil-rare_unlock-4.gif`, `masters/case-1m-sil-rare_unlock-4.mkv`, `rollouts/case-1m-sil-rare_unlock-4.npz`.

**`1m-sil-rare_unlock-5`** (run `sil-1m-6f8690ad-s5`, `a4f49419e863402e946a7a869e1fa922`, checkpoint 244 (999,424 env steps), eval seed 10057, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 259; window steps 109 to 259, at its start health 9, food 6, drink 5, energy 9, inventory sapling 2. Behaviour: most chosen actions do 31, place table 21, make wood pickaxe 13; combat damage 6, close calls 0, lowest health 4. At the unlock health 4, food 0, drink 1, energy 7, inventory wood 1, sapling 3. Outcome: the episode lasted 332 steps. Replay: `gifs/1m-sil-rare_unlock-5.gif`, `masters/case-1m-sil-rare_unlock-5.mkv`, `rollouts/case-1m-sil-rare_unlock-5.npz`.

**`1m-sil-rare_unlock-6`** (run `sil-1m-6f8690ad-s2`, `f14833a47a6d47978742f496a4ce3821`, checkpoint 244 (999,424 env steps), eval seed 10011, stochastic). Situation: place furnace (rare at the final checkpoint) is unlocked at step 104; window steps 0 to 104, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 22, right 12, down 11; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 6, drink 5, energy 8, inventory sapling 2. Outcome: the episode lasted 281 steps. Replay: `gifs/1m-sil-rare_unlock-6.gif`, `masters/case-1m-sil-rare_unlock-6.mkv`, `rollouts/case-1m-sil-rare_unlock-6.npz`.

**`1m-sil-rare_unlock-7`** (run `sil-1m-6f8690ad-s5`, `a4f49419e863402e946a7a869e1fa922`, checkpoint 244 (999,424 env steps), eval seed 10025, stochastic). Situation: collect coal (rare at the final checkpoint) is unlocked at step 145; window steps 0 to 145, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 31, place table 16, down 11; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 5, drink 3, energy 9, inventory stone 1, coal 1, sapling 1. Outcome: the episode lasted 274 steps. Replay: `gifs/1m-sil-rare_unlock-7.gif`, `masters/case-1m-sil-rare_unlock-7.mkv`, `rollouts/case-1m-sil-rare_unlock-7.npz`.

**`1m-sil-rare_unlock-8`** (run `sil-1m-6f8690ad-s6`, `0a7c86571c9f4ad681588857143d788a`, checkpoint 244 (999,424 env steps), eval seed 10039, stochastic). Situation: make arrow (rare at the final checkpoint) is unlocked at step 296; window steps 146 to 296, at its start health 9, food 5, drink 3, energy 9, inventory wood 1, stone 1, sapling 2. Behaviour: most chosen actions up 25, do 22, make wood sword 13; combat damage 0, close calls 0, lowest health 5. At the unlock health 5, food 0, drink 0, energy 9, inventory sapling 2. Outcome: the episode lasted 370 steps. Replay: `gifs/1m-sil-rare_unlock-8.gif`, `masters/case-1m-sil-rare_unlock-8.mkv`, `rollouts/case-1m-sil-rare_unlock-8.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-sil-death-1`** (run `sil-1m-6f8690ad-s6`, `0a7c86571c9f4ad681588857143d788a`, checkpoint 244 (999,424 env steps), eval seed 10002, stochastic). Situation: death by zombie at step 231; the window (steps 81 to 231) opens with health 9, food 7, drink 6, energy 8, inventory empty. Behaviour: most chosen actions up 35, do 23, make wood sword 19; combat damage 8, close calls 0, lowest health 0. At death health 0, food 1, drink 0, energy 7, inventory stone 2, sapling 2. Outcome: the episode ended after 231 steps. Replay: `gifs/1m-sil-death-1.gif`, `masters/case-1m-sil-death-1.mkv`, `rollouts/case-1m-sil-death-1.npz`.

**`1m-sil-death-2`** (run `sil-1m-6f8690ad-s0`, `98aadc7012c44416ae6ab215a3ed197a`, checkpoint 244 (999,424 env steps), eval seed 10042, stochastic). Situation: death by starvation at step 392; the window (steps 242 to 392) opens with health 9, food 1, drink 0, energy 8, inventory wood 1. Behaviour: most chosen actions place table 20, do 19, sleep 12; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 8, inventory wood 1. Outcome: the episode ended after 392 steps. Replay: `gifs/1m-sil-death-2.gif`, `masters/case-1m-sil-death-2.mkv`, `rollouts/case-1m-sil-death-2.npz`.

**`1m-sil-death-3`** (run `sil-1m-6f8690ad-s1`, `85bc7e9b8ba7401c9edd0ca48d41a620`, checkpoint 244 (999,424 env steps), eval seed 10015, stochastic). Situation: death by arrow at step 301; the window (steps 151 to 301) opens with health 9, food 4, drink 9, energy 8, inventory wood 2. Behaviour: most chosen actions do 44, left 28, right 25; combat damage 8, close calls 0, lowest health 0. At death health 0, food 0, drink 9, energy 3, inventory empty. Outcome: the episode ended after 301 steps. Replay: `gifs/1m-sil-death-3.gif`, `masters/case-1m-sil-death-3.mkv`, `rollouts/case-1m-sil-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-sil`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting, insertion overflow fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 2 / 10 | 45.1k to 135k | 2 drops, deepest drawdown 0.54 (floor 0.63) (run f14833a47a6d47978742f496a4ce3821; 1 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-rare_unlock-2.gif` |
| Collapse | 1M | 1 / 10 | 45.1k to 999k | drawdown 0.54, final 2.18 of best 2.51 (floor 0.63) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-death-1.gif` |
| Plateau | 1M | 2 / 10 | 500k to 999k | score gain 0.21 from 500k to the end (< 0.23) (run a4f49419e863402e946a7a869e1fa922; 1 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-stall-2.gif` |
| Skill forgetting | 1M | 10 / 10 | 16.4k to 967k | >= 20% to < 5%: collect_drink (73.7k to 225k), make_wood_pickaxe (598k to 606k), place_plant (24.6k to 32.8k) (run 98aadc7012c44416ae6ab215a3ed197a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Overfitting to early lucky windows | 1M | 9 / 10 | 188k to 999k | sil/sample_age_max > 50% of the updates, sil/active > 0.5, entropy below the base control's from 410k (8 stretches) (run 98aadc7012c44416ae6ab215a3ed197a; 8 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-stall-4.gif` |
| Forgetting | 1M | 4 / 10 | 900k to 1M | basic success rates below the base control's: make_wood_sword -15 points, collect_drink -12 points (run 98aadc7012c44416ae6ab215a3ed197a; 3 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Insertion overflow | 1M | 10 / 10 | 12.3k to 999k | sil/overflow > 0 in 78% of logged updates (run 98aadc7012c44416ae6ab215a3ed197a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-stall-4.gif` |

- Absent in every run: Entropy collapse, Value failure, Numerical failure, Reward hacking, Exploration failure, Survival failure, Imitated unsafe play, Value overestimation, Dead SIL.
- Detector verdicts over the tuned runs (160 run-detector pairs): absent 122, detected 38; needs replay 0; needs eval actions 0. No metric family is missing.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[SIL training analysis]] (this note): floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 55.2% of deaths, 0.113 stalls per episode (median 333.5 steps, 18.1% escaped); final score 2.137.
- [[PPO-RNN training analysis]] (base control): floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.
- [[PPO training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PQN-RNN training analysis]]: floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.

## Ablations

Ablation units of `final-1m` (R77): reported apart from the tuned config, never pooled into the sections above. The change is an override of the tuned config in `configs/phases/final-1m.yaml`.

| Unit | Change | n | Final score (sd) | Stochastic eval | Floors reached | Never / rare / common achievements | Top death causes | Stalls per episode (median, escape) |
|---|---|---|---|---|---|---|---|---|
| tuned | - | 10 | 2.137 (0.046) | 2.131 | floor 0 only | 47 / 11 / 9 | zombie 55.2%, starvation 29.7% | 0.113 (333.5, 18.1%) |
| single-buffer | `layout: single` on the tuned config | 10 | 2.138 (0.094) | 2.181 | floor 0 only | 49 / 9 / 9 | zombie 59.1%, starvation 24.8% | 0.109 (336, 14.3%) |

Sources: `hard/*.csv` (rows of each `config_ref`) and the scores of `outputs/metrics/final-1m/runs.parquet`. The `tuned` row repeats the main sections for comparison.

### Obstacles: single-buffer

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Plateau | 1M | 3 / 10 | 500k to 999k | score gain 0.18 from 500k to the end (< 0.23) (run 5a84ae16934b4a758ec74a3b6414dc49; 2 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-single-buffer-stall-7.gif` |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 999k | >= 20% to < 5%: collect_drink (77.8k to 131k), eat_cow (311k to 328k), make_wood_pickaxe (455k to 479k), make_wood_sword (520k to 537k) (run c53b79209d7e40fd985db0a407386588; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Overfitting to early lucky windows | 1M | 5 / 10 | 12.3k to 12.3k | sil/sample_age_max > 50% of the updates, sil/active > 0.5, entropy below the base control's from 12.3k (run c53b79209d7e40fd985db0a407386588; 4 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-single-buffer-death-1.gif` |
| Forgetting | 1M | 2 / 10 | 900k to 1M | basic success rates below the base control's: make_wood_sword -10 points (run 8a1fa373f38f4f4b995335d01054edb4; 1 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Value overestimation | 1M | 6 / 10 | 500k to 1M | second half: sil/value_bias 0.206 > 0, explained variance 0.94 < base control's 0.95 (run c53b79209d7e40fd985db0a407386588; 5 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-single-buffer-death-1.gif` |
| Insertion overflow | 1M | 10 / 10 | 12.3k to 999k | sil/overflow > 0 in 82% of logged updates (run c53b79209d7e40fd985db0a407386588; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-sil-single-buffer-death-1.gif` |

Cases of single-buffer (`cases.csv`, replay files under `training-analysis-1m-sil`):

| Case | Kind | Detail | Run | Checkpoint | Eval seed | GIF |
|---|---|---|---|---|---|---|
| `1m-sil-single-buffer-stall-1` | stall | 469 steps | `sil-1m-57a864f9-s4` | 244 | 10027 | `gifs/1m-sil-single-buffer-stall-1.gif` |
| `1m-sil-single-buffer-stall-2` | stall | 433 steps | `sil-1m-57a864f9-s4` | 244 | 10009 | `gifs/1m-sil-single-buffer-stall-2.gif` |
| `1m-sil-single-buffer-stall-3` | stall | 414 steps | `sil-1m-57a864f9-s4` | 244 | 10014 | `gifs/1m-sil-single-buffer-stall-3.gif` |
| `1m-sil-single-buffer-stall-4` | stall | 361 steps | `sil-1m-57a864f9-s8` | 244 | 10012 | `gifs/1m-sil-single-buffer-stall-4.gif` |
| `1m-sil-single-buffer-stall-5` | stall | 354 steps | `sil-1m-57a864f9-s1` | 244 | 10053 | `gifs/1m-sil-single-buffer-stall-5.gif` |
| `1m-sil-single-buffer-stall-6` | stall | 355 steps | `sil-1m-57a864f9-s1` | 2 | 10041 | `gifs/1m-sil-single-buffer-stall-6.gif` |
| `1m-sil-single-buffer-stall-7` | stall | 308 steps | `sil-1m-57a864f9-s5` | 12 | 10042 | `gifs/1m-sil-single-buffer-stall-7.gif` |
| `1m-sil-single-buffer-stall-8` | stall | 380 steps | `sil-1m-57a864f9-s6` | 148 | 10062 | `gifs/1m-sil-single-buffer-stall-8.gif` |
| `1m-sil-single-buffer-deep_floor-1` | deep_floor | floor 0 | `sil-1m-57a864f9-s2` | 244 | 10009 | `gifs/1m-sil-single-buffer-deep_floor-1.gif` |
| `1m-sil-single-buffer-deep_floor-2` | deep_floor | floor 0 | `sil-1m-57a864f9-s4` | 244 | 10014 | `gifs/1m-sil-single-buffer-deep_floor-2.gif` |
| `1m-sil-single-buffer-deep_floor-3` | deep_floor | floor 0 | `sil-1m-57a864f9-s4` | 244 | 10022 | `gifs/1m-sil-single-buffer-deep_floor-3.gif` |
| `1m-sil-single-buffer-rare_unlock-1` | rare_unlock | defeat_zombie | `sil-1m-57a864f9-s9` | 244 | 10047 | `gifs/1m-sil-single-buffer-rare_unlock-1.gif` |
| `1m-sil-single-buffer-rare_unlock-2` | rare_unlock | collect_stone | `sil-1m-57a864f9-s7` | 244 | 10052 | `gifs/1m-sil-single-buffer-rare_unlock-2.gif` |
| `1m-sil-single-buffer-rare_unlock-3` | rare_unlock | place_stone | `sil-1m-57a864f9-s2` | 244 | 10043 | `gifs/1m-sil-single-buffer-rare_unlock-3.gif` |
| `1m-sil-single-buffer-rare_unlock-4` | rare_unlock | eat_plant | `sil-1m-57a864f9-s4` | 244 | 10014 | `gifs/1m-sil-single-buffer-rare_unlock-4.gif` |
| `1m-sil-single-buffer-rare_unlock-5` | rare_unlock | defeat_skeleton | `sil-1m-57a864f9-s3` | 244 | 10005 | `gifs/1m-sil-single-buffer-rare_unlock-5.gif` |
| `1m-sil-single-buffer-rare_unlock-6` | rare_unlock | make_stone_sword | `sil-1m-57a864f9-s2` | 244 | 10009 | `gifs/1m-sil-single-buffer-rare_unlock-6.gif` |
| `1m-sil-single-buffer-rare_unlock-7` | rare_unlock | place_furnace | `sil-1m-57a864f9-s2` | 244 | 10031 | `gifs/1m-sil-single-buffer-rare_unlock-7.gif` |
| `1m-sil-single-buffer-rare_unlock-8` | rare_unlock | collect_coal | `sil-1m-57a864f9-s7` | 244 | 10041 | `gifs/1m-sil-single-buffer-rare_unlock-8.gif` |
| `1m-sil-single-buffer-death-1` | death | zombie | `sil-1m-57a864f9-s1` | 244 | 10034 | `gifs/1m-sil-single-buffer-death-1.gif` |
| `1m-sil-single-buffer-death-2` | death | starvation | `sil-1m-57a864f9-s2` | 244 | 10057 | `gifs/1m-sil-single-buffer-death-2.gif` |
| `1m-sil-single-buffer-death-3` | death | arrow | `sil-1m-57a864f9-s2` | 244 | 10004 | `gifs/1m-sil-single-buffer-death-3.gif` |
