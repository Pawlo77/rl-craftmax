---
title: "PPO-RNN training analysis"
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
# PPO-RNN training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 2.188 (sd 0.107), stochastic eval score 2.160; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%).
- Achievements: 47 of 67 never unlocked by any seed at any checkpoint (basic 6, intermediate 17, advanced 15, very advanced 9), 11 rare (final rate < 5%), 9 common.
- Deaths at the final checkpoint: zombie 56.6%, starvation 28.3%, arrow 13.4%, dehydration 1.7% (the untrained policy: zombie 46.4%, starvation 35.8%, arrow 17.8%).
- Stalls: 72 stalls in 640 final eval episodes, 0.113 per episode (seed interval 0.094 to 0.133), median length 334.5 steps, 11.1% escaped by an unlock; they are started by wake up (47%) and collect drink (18%).
- Obstacles at 1M: instability 1/10, plateau 3/10; not logged (R78): exploration failure, reward hacking. Caveat: skill forgetting 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m`, PPO-RNN (control, base of count-bonus-ppo-rnn, sil and the combination), 10 seeds (0-9) of the tuned config; method: [[PPO-RNN design]], plan [[2026-10-20-craftax-method-ppo-rnn]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/ppo-rnn/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-ppo-rnn`, id `e2ef82fde83041f9a8501e9abd546f02`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 20 case episodes, all replayed from the stored eval files.
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- 1B: pending (final-1b not run, R75); sections marked 1B below are placeholders for the same tables at 1B.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 10 of this method's 10 tuned runs (10 of 10 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 66, detected 14, not logged 20; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Floor entries before or beside the final eval: `enter dungeon` unlocked in 1 of 10 seeds at some point (seeds with a training-episode unlock: 1, with an eval unlock: 0; final stochastic rate 0.00%). Source: `training-analysis-1m-ppo-rnn`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (47 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (6): make stone sword, collect iron, collect diamond, make iron pickaxe, make iron sword, place torch; intermediate (17): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-ppo-rnn`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 11): collect stone 2.97% (10/10 seeds ever), place stone 1.41% (9/10 seeds ever), place furnace 1.25% (10/10 seeds ever), defeat zombie 0.47% (10/10 seeds ever), eat plant 0.16% (7/10 seeds ever), defeat skeleton 0.16% (10/10 seeds ever), make stone pickaxe 0.00% (2/10 seeds ever), collect coal 0.00% (7/10 seeds ever), make arrow 0.00% (2/10 seeds ever), make torch 0.00% (1/10 seeds ever), enter dungeon 0.00% (1/10 seeds ever). Same file.
- Common (9): collect sapling 99.5%, wake up 99.2%, collect wood 94.8%, place plant 94.7%, place table 81.4%, collect drink 56.9%, make wood sword 18.9%, make wood pickaxe 16.1%, eat cow 10.2%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 362 (56.6%); starvation 181 (28.3%); arrow 86 (13.4%); dehydration 11 (1.7%). At checkpoint 0 (untrained): zombie 46.4%, starvation 35.8%, arrow 17.8%. Source: `training-analysis-1m-ppo-rnn`, `hard/death_causes.csv`, plot `plots/death_causes_over_training.png`.
- Stalls (>= 300 steps without an unlock): 72 stalls in 640 final eval episodes, 0.113 per episode (seed interval 0.094 to 0.133), median length 334.5 steps, 11.1% escaped by an unlock. All started on floor 0; started by wake up 34 (47.2%), collect drink 13 (18.1%), eat cow 10 (13.9%), place table 8 (11.1%). Source: `training-analysis-1m-ppo-rnn`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 8 by an unlock, 64 by death. Longest stall 758 steps (run `5576c36feaa348b48ef3bef498d87d38`, seed 0, eval seed 10042, steps 298 to 1056, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-ppo-rnn` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 72 stalls, 8 ended by an unlock, 64 by death (88.9%).
- Before the unlock (8 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 0 cases; most frequent actions do 12%, up 10%, place table 10%; movement keys are 27% of the last 50 steps against 20% of the whole stall window, `do` 12% against 20%; the most frequent action of the last 50 differs from the window's in 5 of 8 cases.
- State at the unlock: health <= 4 in 7 of 8 cases (median health 1 at the unlock, 9 at the stall start), food 0 in 7 of 8 (median food 0, 8 at the start), drink 0 in 4 of 8.
- Achievements that ended the stalls: place table 3, collect wood 2, collect sapling 2, eat plant 1.
- Deepest floor: the deep-floor cases are all overworld (floor 0) episodes, so the route question is moot at 1M.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-ppo-rnn-stall-1`** (run `ppo-rnn-1m-b3ce95c4-s2`, `dfa9710e58ad4b1abb4a10b0e6ee54e3`, checkpoint 244 (999,424 env steps), eval seed 10057, stochastic). Situation: a stall of 449 steps from step 306 on floor 0, begun after make wood sword; at its start health 8, food 6, drink 4, energy 8, inventory empty. Behaviour during the stall: most chosen actions do 89, make wood sword 21, right 19 over the 449 steps; 2.78% of overworld tiles first seen in the window; combat damage 2, close calls 0, lowest health 6. What broke it: eat plant at step 755 (then health 9, food 9, drink 5, energy 9, inventory wood 1, sapling 1). Outcome: the episode lasted 1076 steps. Replay: `gifs/1m-ppo-rnn-stall-1.gif`, `overlays/1m-ppo-rnn-stall-1.mp4`, `masters/case-1m-ppo-rnn-stall-1.mkv`, `rollouts/case-1m-ppo-rnn-stall-1.npz`.

**`1m-ppo-rnn-stall-2`** (run `ppo-rnn-1m-b3ce95c4-s9`, `fec562d6ca1e45458d67215851224790`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: a stall of 372 steps from step 43 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions sleep 81, do 68, up 41 over the 372 steps; 4.08% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 415 (then health 1, food 0, drink 3, energy 6, inventory wood 1, sapling 3). Outcome: the episode lasted 423 steps. Replay: `gifs/1m-ppo-rnn-stall-2.gif`, `overlays/1m-ppo-rnn-stall-2.mp4`, `masters/case-1m-ppo-rnn-stall-2.mkv`, `rollouts/case-1m-ppo-rnn-stall-2.npz`.

**`1m-ppo-rnn-stall-3`** (run `ppo-rnn-1m-b3ce95c4-s4`, `d82155238efc4586811753cf4f79b5d5`, checkpoint 244 (999,424 env steps), eval seed 10006, stochastic). Situation: a stall of 360 steps from step 54 on floor 0, begun after wake up; at its start health 9, food 8, drink 9, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions do 80, sleep 55, place table 48 over the 360 steps; 0.56% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 414 (then health 1, food 0, drink 5, energy 9, inventory empty). Outcome: the episode lasted 430 steps. Replay: `gifs/1m-ppo-rnn-stall-3.gif`, `overlays/1m-ppo-rnn-stall-3.mp4`, `masters/case-1m-ppo-rnn-stall-3.mkv`, `rollouts/case-1m-ppo-rnn-stall-3.npz`.

**`1m-ppo-rnn-stall-4`** (run `ppo-rnn-1m-b3ce95c4-s5`, `b8faf8481ef54c0dad1d5c1d119a0202`, checkpoint 244 (999,424 env steps), eval seed 10026, stochastic). Situation: a stall of 345 steps from step 45 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory sapling 4. Behaviour during the stall: most chosen actions do 90, sleep 37, place table 32 over the 345 steps; 1.78% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 390 (then health 1, food 0, drink 0, energy 8, inventory wood 1, sapling 8). Outcome: the episode lasted 401 steps. Replay: `gifs/1m-ppo-rnn-stall-4.gif`, `overlays/1m-ppo-rnn-stall-4.mp4`, `masters/case-1m-ppo-rnn-stall-4.mkv`, `rollouts/case-1m-ppo-rnn-stall-4.npz`.

**`1m-ppo-rnn-stall-5`** (run `ppo-rnn-1m-b3ce95c4-s7`, `afe400a7d25b4f1487b1105094d05000`, checkpoint 244 (999,424 env steps), eval seed 10010, stochastic). Situation: a stall of 344 steps from step 46 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions place table 50, do 29, sleep 24 over the 344 steps; 2.78% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 390 (then health 1, food 0, drink 0, energy 8, inventory wood 1, sapling 1). Outcome: the episode lasted 391 steps. Replay: `gifs/1m-ppo-rnn-stall-5.gif`, `overlays/1m-ppo-rnn-stall-5.mp4`, `masters/case-1m-ppo-rnn-stall-5.mkv`, `rollouts/case-1m-ppo-rnn-stall-5.npz`.

**`1m-ppo-rnn-stall-6`** (run `ppo-rnn-1m-b3ce95c4-s8`, `3e56a870134d4163956a8ff28864e13b`, checkpoint 1 (4,096 env steps), eval seed 10025, stochastic). Situation: a stall of 314 steps from step 97 on floor 0; at its start health 9, food 6, drink 7, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 12, drink potion green 12, make diamond armour 10 over the 314 steps; 0.95% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect sapling at step 411 (then health 1, food 0, drink 0, energy 9, inventory sapling 1). Outcome: the episode lasted 416 steps. Replay: `gifs/1m-ppo-rnn-stall-6.gif`, `overlays/1m-ppo-rnn-stall-6.mp4`, `masters/case-1m-ppo-rnn-stall-6.mkv`, `rollouts/case-1m-ppo-rnn-stall-6.npz`.

**`1m-ppo-rnn-stall-7`** (run `ppo-rnn-1m-b3ce95c4-s0`, `5576c36feaa348b48ef3bef498d87d38`, checkpoint 20 (81,920 env steps), eval seed 10042, stochastic). Situation: a stall of 308 steps from step 134 on floor 0; at its start health 9, food 5, drink 7, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions do 45, place plant 33, right 12 over the 308 steps; 0.04% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect sapling at step 442 (then health 1, food 0, drink 9, energy 8, inventory wood 1, sapling 1). Outcome: the episode lasted 447 steps. Replay: `gifs/1m-ppo-rnn-stall-7.gif`, `overlays/1m-ppo-rnn-stall-7.mp4`, `masters/case-1m-ppo-rnn-stall-7.mkv`, `rollouts/case-1m-ppo-rnn-stall-7.npz`.

**`1m-ppo-rnn-stall-8`** (run `ppo-rnn-1m-b3ce95c4-s2`, `dfa9710e58ad4b1abb4a10b0e6ee54e3`, checkpoint 148 (606,208 env steps), eval seed 10057, stochastic). Situation: a stall of 361 steps from step 54 on floor 0; at its start health 9, food 8, drink 9, energy 9, inventory wood 1, sapling 2. Behaviour during the stall: most chosen actions left 53, up 46, do 39 over the 361 steps; 2.08% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 415 (then health 1, food 0, drink 0, energy 8, inventory sapling 2). Outcome: the episode lasted 425 steps. Replay: `gifs/1m-ppo-rnn-stall-8.gif`, `overlays/1m-ppo-rnn-stall-8.mp4`, `masters/case-1m-ppo-rnn-stall-8.mkv`, `rollouts/case-1m-ppo-rnn-stall-8.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-ppo-rnn-deep_floor-1`** (run `ppo-rnn-1m-b3ce95c4-s3`, `09b48ed2324c45c1a5de200c337f4d04`, checkpoint 244 (999,424 env steps), eval seed 10030, stochastic). Situation: an episode with 12 achievements unlocked on floor 0; window steps 102 to 252, at its start health 9, food 6, drink 5, energy 9, inventory sapling 1. Behaviour: most chosen actions do 26, make wood sword 10, place table 7; 2.17% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 9. At step 252 (the last unlock) health 9, food 7, drink 1, energy 8, inventory sapling 1. Outcome: the episode lasted 592 steps. Replay: `gifs/1m-ppo-rnn-deep_floor-1.gif`, `masters/case-1m-ppo-rnn-deep_floor-1.mkv`, `rollouts/case-1m-ppo-rnn-deep_floor-1.npz`.

**`1m-ppo-rnn-deep_floor-2`** (run `ppo-rnn-1m-b3ce95c4-s3`, `09b48ed2324c45c1a5de200c337f4d04`, checkpoint 244 (999,424 env steps), eval seed 10045, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 233 to 383, at its start health 9, food 1, drink 0, energy 8, inventory wood 1, stone 2, sapling 4. Behaviour: most chosen actions do 20, left 9, down 9; 4.82% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 0. At step 383 (the last unlock) health 0, food 0, drink 0, energy 7, inventory sapling 4. Outcome: the episode lasted 383 steps. Replay: `gifs/1m-ppo-rnn-deep_floor-2.gif`, `masters/case-1m-ppo-rnn-deep_floor-2.mkv`, `rollouts/case-1m-ppo-rnn-deep_floor-2.npz`.

**`1m-ppo-rnn-deep_floor-3`** (run `ppo-rnn-1m-b3ce95c4-s2`, `dfa9710e58ad4b1abb4a10b0e6ee54e3`, checkpoint 244 (999,424 env steps), eval seed 10028, stochastic). Situation: an episode with 10 achievements unlocked on floor 0; window steps 146 to 296, at its start health 9, food 4, drink 3, energy 6, inventory wood 3, sapling 1. Behaviour: most chosen actions do 17, make wood sword 12, right 11; 4.47% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 6. At step 296 (the last unlock) health 6, food 0, drink 0, energy 7, inventory sapling 1. Outcome: the episode lasted 394 steps. Replay: `gifs/1m-ppo-rnn-deep_floor-3.gif`, `masters/case-1m-ppo-rnn-deep_floor-3.mkv`, `rollouts/case-1m-ppo-rnn-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-ppo-rnn-rare_unlock-1`** (run `ppo-rnn-1m-b3ce95c4-s7`, `afe400a7d25b4f1487b1105094d05000`, checkpoint 244 (999,424 env steps), eval seed 10013, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 201; window steps 51 to 201, at its start health 9, food 8, drink 7, energy 9, inventory empty. Behaviour: most chosen actions do 25, place table 23, make wood sword 20; combat damage 8, close calls 0, lowest health 1. At the unlock health 2, food 2, drink 1, energy 8, inventory empty. Outcome: the episode lasted 231 steps. Replay: `gifs/1m-ppo-rnn-rare_unlock-1.gif`, `masters/case-1m-ppo-rnn-rare_unlock-1.mkv`, `rollouts/case-1m-ppo-rnn-rare_unlock-1.npz`.

**`1m-ppo-rnn-rare_unlock-2`** (run `ppo-rnn-1m-b3ce95c4-s8`, `3e56a870134d4163956a8ff28864e13b`, checkpoint 244 (999,424 env steps), eval seed 10006, stochastic). Situation: collect stone (rare at the final checkpoint) is unlocked at step 114; window steps 0 to 114, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 30, make wood pickaxe 16, right 15; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 5, drink 4, energy 8, inventory wood 1, stone 1. Outcome: the episode lasted 388 steps. Replay: `gifs/1m-ppo-rnn-rare_unlock-2.gif`, `masters/case-1m-ppo-rnn-rare_unlock-2.mkv`, `rollouts/case-1m-ppo-rnn-rare_unlock-2.npz`.

**`1m-ppo-rnn-rare_unlock-3`** (run `ppo-rnn-1m-b3ce95c4-s9`, `fec562d6ca1e45458d67215851224790`, checkpoint 244 (999,424 env steps), eval seed 10001, stochastic). Situation: place stone (rare at the final checkpoint) is unlocked at step 197; window steps 47 to 197, at its start health 9, food 8, drink 7, energy 8, inventory sapling 1. Behaviour: most chosen actions do 26, place table 22, down 18; combat damage 4, close calls 0, lowest health 5. At the unlock health 5, food 2, drink 1, energy 6, inventory wood 1, stone 1, sapling 2. Outcome: the episode lasted 215 steps. Replay: `gifs/1m-ppo-rnn-rare_unlock-3.gif`, `masters/case-1m-ppo-rnn-rare_unlock-3.mkv`, `rollouts/case-1m-ppo-rnn-rare_unlock-3.npz`.

**`1m-ppo-rnn-rare_unlock-4`** (run `ppo-rnn-1m-b3ce95c4-s2`, `dfa9710e58ad4b1abb4a10b0e6ee54e3`, checkpoint 244 (999,424 env steps), eval seed 10057, stochastic). Situation: eat plant (rare at the final checkpoint) is unlocked at step 755; window steps 605 to 755, at its start health 8, food 5, drink 6, energy 9, inventory wood 1. Behaviour: most chosen actions do 22, make wood sword 12, left 7; combat damage 0, close calls 0, lowest health 8. At the unlock health 9, food 9, drink 5, energy 9, inventory wood 1, sapling 1. Outcome: the episode lasted 1076 steps. Replay: `gifs/1m-ppo-rnn-rare_unlock-4.gif`, `masters/case-1m-ppo-rnn-rare_unlock-4.mkv`, `rollouts/case-1m-ppo-rnn-rare_unlock-4.npz`.

**`1m-ppo-rnn-rare_unlock-5`** (run `ppo-rnn-1m-b3ce95c4-s0`, `5576c36feaa348b48ef3bef498d87d38`, checkpoint 244 (999,424 env steps), eval seed 10022, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 241; window steps 91 to 241, at its start health 9, food 6, drink 5, energy 8, inventory sapling 2. Behaviour: most chosen actions do 25, right 21, left 18; combat damage 10, close calls 0, lowest health 4. At the unlock health 5, food 1, drink 8, energy 9, inventory wood 2, sapling 2. Outcome: the episode lasted 263 steps. Replay: `gifs/1m-ppo-rnn-rare_unlock-5.gif`, `masters/case-1m-ppo-rnn-rare_unlock-5.mkv`, `rollouts/case-1m-ppo-rnn-rare_unlock-5.npz`.

**`1m-ppo-rnn-rare_unlock-6`** (run `ppo-rnn-1m-b3ce95c4-s6`, `1fc4f2d7b8034e11b557ded6dbdd53ca`, checkpoint 244 (999,424 env steps), eval seed 10019, stochastic). Situation: place furnace (rare at the final checkpoint) is unlocked at step 167; window steps 17 to 167, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions place table 33, sleep 25, do 12; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 4, drink 2, energy 9, inventory empty. Outcome: the episode lasted 213 steps. Replay: `gifs/1m-ppo-rnn-rare_unlock-6.gif`, `masters/case-1m-ppo-rnn-rare_unlock-6.mkv`, `rollouts/case-1m-ppo-rnn-rare_unlock-6.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-ppo-rnn-death-1`** (run `ppo-rnn-1m-b3ce95c4-s3`, `09b48ed2324c45c1a5de200c337f4d04`, checkpoint 244 (999,424 env steps), eval seed 10054, stochastic). Situation: death by zombie at step 225; the window (steps 75 to 225) opens with health 9, food 7, drink 9, energy 9, inventory sapling 1. Behaviour: most chosen actions down 32, do 27, place table 24; combat damage 9, close calls 0, lowest health 0. At death health 0, food 2, drink 8, energy 7, inventory wood 2, sapling 1. Outcome: the episode ended after 225 steps. Replay: `gifs/1m-ppo-rnn-death-1.gif`, `masters/case-1m-ppo-rnn-death-1.mkv`, `rollouts/case-1m-ppo-rnn-death-1.npz`.

**`1m-ppo-rnn-death-2`** (run `ppo-rnn-1m-b3ce95c4-s3`, `09b48ed2324c45c1a5de200c337f4d04`, checkpoint 244 (999,424 env steps), eval seed 10050, stochastic). Situation: death by starvation at step 390; the window (steps 240 to 390) opens with health 8, food 1, drink 0, energy 5, inventory empty. Behaviour: most chosen actions do 16, down 9, place plant 5; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 8, inventory empty. Outcome: the episode ended after 390 steps. Replay: `gifs/1m-ppo-rnn-death-2.gif`, `masters/case-1m-ppo-rnn-death-2.mkv`, `rollouts/case-1m-ppo-rnn-death-2.npz`.

**`1m-ppo-rnn-death-3`** (run `ppo-rnn-1m-b3ce95c4-s0`, `5576c36feaa348b48ef3bef498d87d38`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: death by arrow at step 313; the window (steps 163 to 313) opens with health 9, food 4, drink 2, energy 7, inventory wood 1. Behaviour: most chosen actions do 18, up 13, left 11; combat damage 10, close calls 0, lowest health 0. At death health 0, food 0, drink 9, energy 7, inventory wood 1. Outcome: the episode ended after 313 steps. Replay: `gifs/1m-ppo-rnn-death-3.gif`, `masters/case-1m-ppo-rnn-death-3.mkv`, `rollouts/case-1m-ppo-rnn-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-ppo-rnn`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 1 / 10 | 492k to 524k | 1 drops, deepest drawdown 0.35 (floor 0.61) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-ppo-rnn-death-1.gif` |
| Plateau | 1M | 3 / 10 | 500k to 999k | score gain -0.00 from 500k to the end (< 0.23) (run 09b48ed2324c45c1a5de200c337f4d04; 2 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-ppo-rnn-deep_floor-1.gif` |
| Reward hacking | 1M | not logged: 10 |  | not logged: train/ach_reward_share | - | - | - |
| Exploration failure | 1M | not logged: 10 |  | not logged: first_unlock/ | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 16.4k to 848k | >= 20% to < 5%: collect_drink (86k to 258k), eat_cow (758k to 778k), make_wood_pickaxe (369k to 451k) (run 5576c36feaa348b48ef3bef498d87d38; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status.png` |

- Absent in every run: Collapse, Entropy collapse, Value failure, Numerical failure, Survival failure.
- Detector verdicts over the tuned runs (100 run-detector pairs): absent 66, detected 14, not logged 20; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

### 1B

1B: pending (final-1b not run, R75). The 1B obstacles table (3 seeds) is added by rerunning `analysis.training` once the 1B finals exist.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[PPO-RNN training analysis]] (this note): floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.
- [[PPO training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PQN-RNN training analysis]]: floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.

## Ablations

No ablation of this method ran in `final-1m`.
