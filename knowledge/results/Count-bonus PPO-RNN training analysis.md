---
title: "Count-bonus PPO-RNN training analysis"
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
# Count-bonus PPO-RNN training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 2.274 (sd 0.117), stochastic eval score 2.244; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; earlier checkpoints: `1m-count-bonus-ppo-rnn-deep_floor-1` reaches floor 1 at checkpoint 90 (368,640 env steps); `enter dungeon` unlocked in 4 of 10 seeds at some point (seeds with a training-episode unlock: 3, with an eval unlock: 1; final stochastic rate 0.00%).
- Achievements: 48 of 67 never unlocked by any seed at any checkpoint (basic 7, intermediate 17, advanced 15, very advanced 9), 10 rare (final rate < 5%), 9 common.
- Deaths at the final checkpoint: zombie 58.9%, starvation 27.5%, arrow 11.1%, dehydration 2.5% (the untrained policy: zombie 46.4%, starvation 35.8%, arrow 17.8%).
- Stalls: 60 stalls in 640 final eval episodes, 0.094 per episode (seed interval 0.075 to 0.116), median length 327.5 steps, 16.7% escaped by an unlock; they are started by wake up (40%) and place table (22%).
- Obstacles at 1M: instability 1/10, plateau 1/10; not logged (R78): early-race slowdown, exploration failure, reward hacking; uninformative at 1M: bonus starvation. Caveat: skill forgetting 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m`, Count-bonus PPO-RNN (count-bonus variant on the PPO-RNN control), 10 seeds (0-9) of the tuned config; method: [[Count-Based Progress Bonus design]], plan [[2026-10-20-craftax-method-count-bonus]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/count-bonus-ppo-rnn/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-count-bonus-ppo-rnn`, id `89043ca3e10a49e08add422530c5bc2a`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 45 case episodes, all replayed from the stored eval files.
- Related analyses: [[Count-bonus PQN-RNN training analysis]] (the other variant), [[Combo training analysis]] (uses this variant as a component).
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- 1B: pending (final-1b not run, R75); sections marked 1B below are placeholders for the same tables at 1B.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 3 of this method's 10 tuned runs (3 of 20 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 189, detected 45, not logged 26, uninformative at 1M 20; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Floor entries before or beside the final eval: `enter dungeon` unlocked in 4 of 10 seeds at some point (seeds with a training-episode unlock: 3, with an eval unlock: 1; final stochastic rate 0.00%). Earlier checkpoints: `1m-count-bonus-ppo-rnn-deep_floor-1` reaches floor 1 at checkpoint 90 (368,640 env steps). Source: `training-analysis-1m-count-bonus-ppo-rnn`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (48 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (7): collect iron, collect diamond, make iron pickaxe, make iron sword, make arrow, make torch, place torch; intermediate (17): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-count-bonus-ppo-rnn`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status_tuned.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 10): collect stone 3.28% (10/10 seeds ever), place stone 1.25% (10/10 seeds ever), place furnace 1.09% (10/10 seeds ever), defeat zombie 0.78% (10/10 seeds ever), defeat skeleton 0.62% (10/10 seeds ever), collect coal 0.47% (10/10 seeds ever), eat plant 0.16% (6/10 seeds ever), make stone pickaxe 0.16% (5/10 seeds ever), make stone sword 0.00% (1/10 seeds ever), enter dungeon 0.00% (4/10 seeds ever). Same file.
- Common (9): collect sapling 97.5%, wake up 96.1%, collect wood 95.0%, place plant 93.9%, place table 85.0%, collect drink 60.2%, make wood pickaxe 27.5%, make wood sword 22.7%, eat cow 11.4%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 377 (58.9%); starvation 176 (27.5%); arrow 71 (11.1%); dehydration 16 (2.5%). At checkpoint 0 (untrained): zombie 46.4%, starvation 35.8%, arrow 17.8%. Source: `training-analysis-1m-count-bonus-ppo-rnn`, `hard/death_causes.csv`, plot `plots/death_causes_over_training_tuned.png`.
- Stalls (>= 300 steps without an unlock): 60 stalls in 640 final eval episodes, 0.094 per episode (seed interval 0.075 to 0.116), median length 327.5 steps, 16.7% escaped by an unlock. All started on floor 0; started by wake up 24 (40.0%), place table 13 (21.7%), collect drink 10 (16.7%), eat cow 9 (15.0%). Source: `training-analysis-1m-count-bonus-ppo-rnn`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 10 by an unlock, 50 by death. Longest stall 446 steps (run `123a4892a0404b1c83cde82f13f79f76`, seed 2, eval seed 10020, steps 167 to 613, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-count-bonus-ppo-rnn` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 60 stalls, 10 ended by an unlock, 50 by death (83.3%).
- Before the unlock (8 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 0 cases; most frequent actions do 20%, up 10%, make wood pickaxe 9%; movement keys are 21% of the last 50 steps against 22% of the whole stall window, `do` 20% against 22%; the most frequent action of the last 50 differs from the window's in 4 of 8 cases.
- State at the unlock: health <= 4 in 8 of 8 cases (median health 1.5 at the unlock, 9 at the stall start), food 0 in 8 of 8 (median food 0, 7 at the start), drink 0 in 6 of 8.
- Achievements that ended the stalls: place table 2, collect wood 2, make wood sword 1, make wood pickaxe 1, collect sapling 1, wake up 1.
- Deepest floor: the deep-floor cases are floor 0, floor 1; the others are overworld episodes (floor 0).

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-count-bonus-ppo-rnn-stall-1`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s9`, `815617b169e44c46827ffd6cd49b4d4a`, checkpoint 244 (999,424 env steps), eval seed 10033, stochastic). Situation: a stall of 334 steps from step 45 on floor 0, begun after wake up; at its start health 9, food 8, drink 8, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions do 55, place furnace 49, place table 31 over the 334 steps; 3.34% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: place table at step 379 (then health 1, food 0, drink 0, energy 7, inventory sapling 1). Outcome: the episode lasted 383 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-1.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-1.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-1.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-1.npz`.

**`1m-count-bonus-ppo-rnn-stall-2`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s0`, `5005fb7d3c904a36943dfe64fb5d7266`, checkpoint 244 (999,424 env steps), eval seed 10044, stochastic). Situation: a stall of 331 steps from step 65 on floor 0, begun after place plant; at its start health 9, food 7, drink 8, energy 9, inventory sapling 1. Behaviour during the stall: most chosen actions up 61, do 38, left 28 over the 331 steps; 17.01% of overworld tiles first seen in the window; combat damage 6, close calls 0, lowest health 2. What broke it: make wood sword at step 396 (then health 2, food 0, drink 0, energy 8, inventory sapling 3). Outcome: the episode lasted 421 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-2.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-2.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-2.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-2.npz`.

**`1m-count-bonus-ppo-rnn-stall-3`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s2`, `123a4892a0404b1c83cde82f13f79f76`, checkpoint 244 (999,424 env steps), eval seed 10057, stochastic). Situation: a stall of 325 steps from step 52 on floor 0, begun after collect drink; at its start health 9, food 8, drink 8, energy 9, inventory wood 1, sapling 1. Behaviour during the stall: most chosen actions do 48, right 36, left 34 over the 325 steps; 4.30% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: place table at step 377 (then health 3, food 0, drink 9, energy 9, inventory sapling 3). Outcome: the episode lasted 420 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-3.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-3.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-3.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-3.npz`.

**`1m-count-bonus-ppo-rnn-stall-4`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s5`, `32966a558b0e42de9d9590941faef956`, checkpoint 244 (999,424 env steps), eval seed 10026, stochastic). Situation: a stall of 314 steps from step 62 on floor 0, begun after wake up; at its start health 9, food 7, drink 7, energy 9, inventory sapling 4. Behaviour during the stall: most chosen actions do 47, make wood pickaxe 30, place table 28 over the 314 steps; 0.52% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 376 (then health 1, food 0, drink 0, energy 9, inventory wood 1, sapling 5). Outcome: the episode lasted 382 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-4.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-4.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-4.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-4.npz`.

**`1m-count-bonus-ppo-rnn-stall-5`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s2`, `123a4892a0404b1c83cde82f13f79f76`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: a stall of 311 steps from step 74 on floor 0, begun after place table; at its start health 9, food 7, drink 6, energy 8, inventory sapling 2. Behaviour during the stall: most chosen actions do 95, make wood pickaxe 55, up 23 over the 311 steps; 10.50% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: make wood pickaxe at step 385 (then health 1, food 0, drink 0, energy 8, inventory sapling 8). Outcome: the episode lasted 389 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-5.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-5.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-5.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-5.npz`.

**`1m-count-bonus-ppo-rnn-stall-6`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s0`, `5005fb7d3c904a36943dfe64fb5d7266`, checkpoint 2 (8,192 env steps), eval seed 10025, stochastic). Situation: a stall of 314 steps from step 97 on floor 0; at its start health 9, food 6, drink 7, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 13, make diamond armour 12, noop 11 over the 314 steps; 0.91% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect sapling at step 411 (then health 1, food 0, drink 0, energy 9, inventory sapling 1). Outcome: the episode lasted 416 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-6.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-6.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-6.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-6.npz`.

**`1m-count-bonus-ppo-rnn-stall-7`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s2`, `123a4892a0404b1c83cde82f13f79f76`, checkpoint 20 (81,920 env steps), eval seed 10025, stochastic). Situation: a stall of 364 steps from step 58 on floor 0; at its start health 9, food 7, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 70, place plant 29, make diamond armour 18 over the 364 steps; 3.43% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: collect wood at step 422 (then health 2, food 0, drink 0, energy 9, inventory wood 1, sapling 1). Outcome: the episode lasted 435 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-7.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-7.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-7.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-7.npz`.

**`1m-count-bonus-ppo-rnn-stall-8`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s1`, `a1a9de9545fa42f0af6e7c59dcd72771`, checkpoint 90 (368,640 env steps), eval seed 10020, stochastic). Situation: a stall of 344 steps from step 55 on floor 0; at its start health 9, food 7, drink 9, energy 8, inventory sapling 2. Behaviour during the stall: most chosen actions do 80, place table 40, up 28 over the 344 steps; 6.94% of overworld tiles first seen in the window; combat damage 2, close calls 0, lowest health 4. What broke it: wake up at step 399 (then health 4, food 0, drink 7, energy 9, inventory sapling 2). Outcome: the episode lasted 450 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-stall-8.gif`, `overlays/1m-count-bonus-ppo-rnn-stall-8.mp4`, `masters/case-1m-count-bonus-ppo-rnn-stall-8.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-stall-8.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-count-bonus-ppo-rnn-deep_floor-1`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s6`, `7dda81e447614c34a535c597ac80d073`, checkpoint 90 (368,640 env steps), eval seed 10031, stochastic). Situation: the episode reaches floor 1 (dungeon) at step 214, its first step there (window steps 64 to 214), at its start health 9, food 7, drink 7, energy 9, inventory empty. Behaviour: most chosen actions right 15, down 13, left 12; 3.91% of overworld tiles first seen in the window, 2.86% of dungeon tiles first seen in the window; combat damage 0, close calls 0, lowest health 9. At step 214 health 9, food 2, drink 0, energy 7, inventory empty. Outcome: the episode lasted 271 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-deep_floor-1.gif`, `masters/case-1m-count-bonus-ppo-rnn-deep_floor-1.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-deep_floor-1.npz`.

**`1m-count-bonus-ppo-rnn-deep_floor-2`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s8`, `b030d84341ba4448aab68dbe9fc17b55`, checkpoint 244 (999,424 env steps), eval seed 10024, stochastic). Situation: an episode with 11 achievements unlocked on floor 0; window steps 107 to 257, at its start health 8, food 6, drink 5, energy 8, inventory empty. Behaviour: most chosen actions do 41, right 29, place stone 12; 8.72% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 7. At step 257 (the last unlock) health 7, food 0, drink 0, energy 6, inventory stone 5, coal 1. Outcome: the episode lasted 382 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-deep_floor-2.gif`, `masters/case-1m-count-bonus-ppo-rnn-deep_floor-2.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-deep_floor-2.npz`.

**`1m-count-bonus-ppo-rnn-deep_floor-3`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s8`, `b030d84341ba4448aab68dbe9fc17b55`, checkpoint 244 (999,424 env steps), eval seed 10043, stochastic). Situation: an episode with 11 achievements unlocked on floor 0; window steps 207 to 357, at its start health 9, food 8, drink 5, energy 7, inventory stone 1, sapling 1. Behaviour: most chosen actions do 38, right 13, place stone 11; 2.34% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 9. At step 357 (the last unlock) health 9, food 3, drink 0, energy 8, inventory wood 1, stone 8, sapling 2. Outcome: the episode lasted 378 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-deep_floor-3.gif`, `masters/case-1m-count-bonus-ppo-rnn-deep_floor-3.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-count-bonus-ppo-rnn-rare_unlock-1`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s8`, `b030d84341ba4448aab68dbe9fc17b55`, checkpoint 244 (999,424 env steps), eval seed 10063, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 157; window steps 7 to 157, at its start health 9, food 9, drink 9, energy 9, inventory sapling 1. Behaviour: most chosen actions do 36, up 25, right 18; combat damage 4, close calls 0, lowest health 5. At the unlock health 5, food 4, drink 3, energy 8, inventory wood 1, sapling 3. Outcome: the episode lasted 270 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-1.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-1.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-1.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-2`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s8`, `b030d84341ba4448aab68dbe9fc17b55`, checkpoint 244 (999,424 env steps), eval seed 10028, stochastic). Situation: collect stone (rare at the final checkpoint) is unlocked at step 128; window steps 0 to 128, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 34, right 28, place table 17; combat damage 10, close calls 1, lowest health 2. At the unlock health 4, food 5, drink 4, energy 7, inventory wood 1, stone 1. Outcome: the episode lasted 155 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-2.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-2.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-2.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-3`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s8`, `b030d84341ba4448aab68dbe9fc17b55`, checkpoint 244 (999,424 env steps), eval seed 10028, stochastic). Situation: place stone (rare at the final checkpoint) is unlocked at step 143; window steps 0 to 143, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 41, right 31, place table 17; combat damage 12, close calls 1, lowest health 2. At the unlock health 3, food 4, drink 3, energy 6, inventory wood 1, stone 4. Outcome: the episode lasted 155 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-3.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-3.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-3.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-4`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s9`, `815617b169e44c46827ffd6cd49b4d4a`, checkpoint 244 (999,424 env steps), eval seed 10042, stochastic). Situation: eat plant (rare at the final checkpoint) is unlocked at step 667; window steps 517 to 667, at its start health 5, food 2, drink 4, energy 8, inventory wood 1. Behaviour: most chosen actions place table 13, do 11, right 10; combat damage 0, close calls 0, lowest health 3. At the unlock health 3, food 4, drink 6, energy 9, inventory wood 1. Outcome: the episode lasted 908 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-4.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-4.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-4.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-5`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s3`, `af2890fa1afc4bfd85272bce2cf7fa87`, checkpoint 244 (999,424 env steps), eval seed 10043, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 128; window steps 0 to 128, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 37, down 22, place table 9; combat damage 6, close calls 0, lowest health 5. At the unlock health 5, food 5, drink 8, energy 9, inventory wood 2, sapling 1. Outcome: the episode lasted 228 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-5.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-5.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-5.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-6`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s7`, `a0698a919a734897b02fed0ac33b14ba`, checkpoint 244 (999,424 env steps), eval seed 10048, stochastic). Situation: make stone pickaxe (rare at the final checkpoint) is unlocked at step 283; window steps 133 to 283, at its start health 9, food 5, drink 4, energy 9, inventory wood 1, sapling 1. Behaviour: most chosen actions do 22, left 21, right 21; combat damage 0, close calls 0, lowest health 8. At the unlock health 8, food 0, drink 5, energy 6, inventory sapling 2. Outcome: the episode lasted 434 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-6.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-6.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-6.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-7`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s9`, `815617b169e44c46827ffd6cd49b4d4a`, checkpoint 244 (999,424 env steps), eval seed 10022, stochastic). Situation: place furnace (rare at the final checkpoint) is unlocked at step 152; window steps 2 to 152, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 41, place table 19, up 16; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 4, drink 3, energy 6, inventory stone 1, sapling 3. Outcome: the episode lasted 367 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-7.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-7.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-7.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-8`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s6`, `7dda81e447614c34a535c597ac80d073`, checkpoint 244 (999,424 env steps), eval seed 10006, stochastic). Situation: collect coal (rare at the final checkpoint) is unlocked at step 191; window steps 41 to 191, at its start health 9, food 8, drink 8, energy 8, inventory wood 1. Behaviour: most chosen actions do 36, make wood sword 19, place table 18; combat damage 6, close calls 0, lowest health 4. At the unlock health 7, food 2, drink 1, energy 5, inventory wood 1, coal 1, sapling 3. Outcome: the episode lasted 263 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-8.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-8.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-8.npz`.

**`1m-count-bonus-ppo-rnn-rare_unlock-9`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s6`, `7dda81e447614c34a535c597ac80d073`, checkpoint 90 (368,640 env steps), eval seed 10031, stochastic). Situation: enter dungeon (rare at the final checkpoint) is unlocked at step 214; window steps 64 to 214, at its start health 9, food 7, drink 7, energy 9, inventory empty. Behaviour: most chosen actions right 15, down 13, left 12; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 2, drink 0, energy 7, inventory empty. Outcome: the episode lasted 271 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-rare_unlock-9.gif`, `masters/case-1m-count-bonus-ppo-rnn-rare_unlock-9.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-rare_unlock-9.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-count-bonus-ppo-rnn-death-1`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s2`, `123a4892a0404b1c83cde82f13f79f76`, checkpoint 244 (999,424 env steps), eval seed 10004, stochastic). Situation: death by zombie at step 226; the window (steps 76 to 226) opens with health 9, food 7, drink 9, energy 9, inventory sapling 1. Behaviour: most chosen actions do 46, up 16, down 13; combat damage 10, close calls 0, lowest health 0. At death health 0, food 2, drink 9, energy 8, inventory wood 1, sapling 2. Outcome: the episode ended after 226 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-death-1.gif`, `masters/case-1m-count-bonus-ppo-rnn-death-1.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-death-1.npz`.

**`1m-count-bonus-ppo-rnn-death-2`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s2`, `123a4892a0404b1c83cde82f13f79f76`, checkpoint 244 (999,424 env steps), eval seed 10012, stochastic). Situation: death by starvation at step 389; the window (steps 239 to 389) opens with health 9, food 1, drink 0, energy 8, inventory wood 2, sapling 6. Behaviour: most chosen actions make wood pickaxe 44, do 34, up 13; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 7, inventory sapling 8. Outcome: the episode ended after 389 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-death-2.gif`, `masters/case-1m-count-bonus-ppo-rnn-death-2.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-death-2.npz`.

**`1m-count-bonus-ppo-rnn-death-3`** (run `count-bonus-ppo-rnn-1m-b0b1c535-s9`, `815617b169e44c46827ffd6cd49b4d4a`, checkpoint 244 (999,424 env steps), eval seed 10014, stochastic). Situation: death by arrow at step 315; the window (steps 165 to 315) opens with health 9, food 4, drink 4, energy 9, inventory sapling 1. Behaviour: most chosen actions do 30, right 13, place furnace 11; combat damage 7, close calls 0, lowest health 0. At death health 0, food 0, drink 9, energy 5, inventory empty. Outcome: the episode ended after 315 steps. Replay: `gifs/1m-count-bonus-ppo-rnn-death-3.gif`, `masters/case-1m-count-bonus-ppo-rnn-death-3.mkv`, `rollouts/case-1m-count-bonus-ppo-rnn-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-count-bonus-ppo-rnn`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 1 / 10 | 397k to 414k | 1 drops, deepest drawdown 0.25 (floor 0.58) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-ppo-rnn-death-1.gif` |
| Plateau | 1M | 1 / 10 | 500k to 999k | score gain 0.22 from 500k to the end (< 0.23) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-ppo-rnn-death-1.gif` |
| Reward hacking | 1M | 0 / 7; not logged: 3 |  | not logged: train/ach_reward_share | - | - | - |
| Exploration failure | 1M | 0 / 7; not logged: 3 |  | not logged: first_unlock/ | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 983k | >= 20% to < 5%: collect_drink (86k to 123k), make_wood_pickaxe (492k to 528k), make_wood_sword (639k to 672k) (run 5005fb7d3c904a36943dfe64fb5d7266; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Bonus starvation | 1M | uninformative at 1M: 10 |  | rule assumes 1B scale | - | - | - |
| Early-race slowdown | 1M | not logged: 10 |  | not logged: first_unlock/ | - | - | - |

- Absent in every run: Collapse, Entropy collapse, Value failure, Numerical failure, Survival failure, Bonus distraction, Hoarding or key cycling.
- Detector verdicts over the tuned runs (140 run-detector pairs): absent 102, detected 12, not logged 16, uninformative at 1M 10; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

### 1B

1B: pending (final-1b not run, R75). The 1B obstacles table (3 seeds) is added by rerunning `analysis.training` once the 1B finals exist.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[Count-bonus PPO-RNN training analysis]] (this note): floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 58.9% of deaths, 0.094 stalls per episode (median 327.5 steps, 16.7% escaped); final score 2.274.
- [[PPO-RNN training analysis]] (base control): floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.
- [[PPO training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PQN-RNN training analysis]]: floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.

## Ablations

Ablation units of `final-1m` (R77): reported apart from the tuned config, never pooled into the sections above. The change is an override of the tuned config in `configs/phases/final-1m.yaml`.

| Unit | Change | n | Final score (sd) | Stochastic eval | Floors reached | Never / rare / common achievements | Top death causes | Stalls per episode (median, escape) |
|---|---|---|---|---|---|---|---|---|
| tuned | - | 10 | 2.274 (0.117) | 2.244 | floor 0 only | 48 / 10 / 9 | zombie 58.9%, starvation 27.5% | 0.094 (327.5, 16.7%) |
| simhash | `key: simhash`, `log2_k: 24`, `max_occupancy: 0.1` on the tuned PPO-RNN variant | 10 | 1.783 (0.125) | 1.799 | floor 0 only | 42 / 17 / 8 | zombie 50.6%, starvation 32.7% | 0.070 (323, 11.1%) |

Sources: `hard/*.csv` (rows of each `config_ref`) and the scores of `outputs/metrics/final-1m/runs.parquet`. The `tuned` row repeats the main sections for comparison.

### Obstacles: simhash

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 7 / 10 | 119k to 823k | 2 drops, deepest drawdown 0.35 (floor 0.49) (run 99514fbe44de42628a615be78fee364a; 6 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-4.gif` |
| Plateau | 1M | 6 / 10 | 500k to 999k | score gain 0.09 from 500k to the end (< 0.19) (run 8d31f8d7e271440db33c298f01c04fde; 5 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-ppo-rnn-simhash-death-1.gif` |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 954k | >= 20% to < 5%: collect_drink (32.8k to 61.4k), eat_cow (750k to 819k), make_wood_pickaxe (582k to 590k), make_wood_sword (696k to 717k), wake_up (12.3k to 127k) (run 99514fbe44de42628a615be78fee364a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Bonus distraction | 1M | 10 / 10 | 201k to 987k | count/r_int_share > 0.5 with a flat score from 291k (2 stretches) (run 99514fbe44de42628a615be78fee364a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-4.gif` |
| Bonus starvation | 1M | uninformative at 1M: 10 |  | rule assumes 1B scale | - | - | - |
| Early-race slowdown | 1M | not logged: 10 |  | base control ppo-rnn lacks first_unlock/ | - | - | - |

Cases of simhash (`cases.csv`, replay files under `training-analysis-1m-count-bonus-ppo-rnn`):

| Case | Kind | Detail | Run | Checkpoint | Eval seed | GIF |
|---|---|---|---|---|---|---|
| `1m-count-bonus-ppo-rnn-simhash-stall-1` | stall | 347 steps | `count-bonus-ppo-rnn-1m-9403d21d-s9` | 244 | 10004 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-1.gif` |
| `1m-count-bonus-ppo-rnn-simhash-stall-2` | stall | 346 steps | `count-bonus-ppo-rnn-1m-9403d21d-s5` | 244 | 10059 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-2.gif` |
| `1m-count-bonus-ppo-rnn-simhash-stall-3` | stall | 336 steps | `count-bonus-ppo-rnn-1m-9403d21d-s3` | 244 | 10013 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-3.gif` |
| `1m-count-bonus-ppo-rnn-simhash-stall-4` | stall | 335 steps | `count-bonus-ppo-rnn-1m-9403d21d-s0` | 244 | 10027 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-4.gif` |
| `1m-count-bonus-ppo-rnn-simhash-stall-5` | stall | 304 steps | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10042 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-5.gif` |
| `1m-count-bonus-ppo-rnn-simhash-stall-6` | stall | 355 steps | `count-bonus-ppo-rnn-1m-9403d21d-s9` | 1 | 10041 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-6.gif` |
| `1m-count-bonus-ppo-rnn-simhash-stall-7` | stall | 319 steps | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 12 | 10007 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-7.gif` |
| `1m-count-bonus-ppo-rnn-simhash-stall-8` | stall | 371 steps | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 148 | 10042 | `gifs/1m-count-bonus-ppo-rnn-simhash-stall-8.gif` |
| `1m-count-bonus-ppo-rnn-simhash-deep_floor-1` | deep_floor | floor 0 | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10041 | `gifs/1m-count-bonus-ppo-rnn-simhash-deep_floor-1.gif` |
| `1m-count-bonus-ppo-rnn-simhash-deep_floor-2` | deep_floor | floor 0 | `count-bonus-ppo-rnn-1m-9403d21d-s5` | 148 | 10005 | `gifs/1m-count-bonus-ppo-rnn-simhash-deep_floor-2.gif` |
| `1m-count-bonus-ppo-rnn-simhash-deep_floor-3` | deep_floor | floor 0 | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10018 | `gifs/1m-count-bonus-ppo-rnn-simhash-deep_floor-3.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-1` | rare_unlock | defeat_zombie | `count-bonus-ppo-rnn-1m-9403d21d-s7` | 54 | 10004 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-1.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-2` | rare_unlock | collect_stone | `count-bonus-ppo-rnn-1m-9403d21d-s5` | 244 | 10021 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-2.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-3` | rare_unlock | place_stone | `count-bonus-ppo-rnn-1m-9403d21d-s5` | 148 | 10026 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-3.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-4` | rare_unlock | eat_plant | `count-bonus-ppo-rnn-1m-9403d21d-s9` | 54 | 10027 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-4.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-5` | rare_unlock | defeat_skeleton | `count-bonus-ppo-rnn-1m-9403d21d-s5` | 148 | 10041 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-5.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-6` | rare_unlock | wake_up | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10047 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-6.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-7` | rare_unlock | place_furnace | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10041 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-7.gif` |
| `1m-count-bonus-ppo-rnn-simhash-rare_unlock-8` | rare_unlock | collect_coal | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10018 | `gifs/1m-count-bonus-ppo-rnn-simhash-rare_unlock-8.gif` |
| `1m-count-bonus-ppo-rnn-simhash-death-1` | death | zombie | `count-bonus-ppo-rnn-1m-9403d21d-s4` | 244 | 10003 | `gifs/1m-count-bonus-ppo-rnn-simhash-death-1.gif` |
| `1m-count-bonus-ppo-rnn-simhash-death-2` | death | starvation | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10011 | `gifs/1m-count-bonus-ppo-rnn-simhash-death-2.gif` |
| `1m-count-bonus-ppo-rnn-simhash-death-3` | death | arrow | `count-bonus-ppo-rnn-1m-9403d21d-s8` | 244 | 10021 | `gifs/1m-count-bonus-ppo-rnn-simhash-death-3.gif` |
