---
title: "SMDP skills training analysis"
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
# SMDP skills training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 0.831 (sd 0.651), stochastic eval score 0.819; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes; `enter dungeon` unlocked in 4 of 10 seeds at some point (seeds with a training-episode unlock: 4, with an eval unlock: 0; final stochastic rate 0.00%).
- Achievements: 52 of 67 never unlocked by any seed at any checkpoint (basic 11, intermediate 17, advanced 15, very advanced 9), 8 rare (final rate < 5%), 7 common.
- Deaths at the final checkpoint: zombie 49.5%, starvation 28.4%, arrow 19.5%, dehydration 2.3%, exhaustion 0.2% (the untrained policy: zombie 51.6%, starvation 28.1%, arrow 20.3%).
- Stalls: 88 stalls in 640 final eval episodes, 0.138 per episode (seed interval 0.089 to 0.186), median length 329 steps, 2.3% escaped by an unlock; they are started by collect sapling (30%) and episode start (25%).
- Obstacles at 1M: instability 10/10, collapse 8/10, plateau 1/10, stuck top level 1/10; not logged (R78): value failure. Caveat: skill forgetting 10/10, skill interference 10/10, deep-skill starvation 10/10, survival failure despite flags 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m`, SMDP skills (Q-learning over hand-defined skills, on the PQN-RNN control), 10 seeds (0-9) of the tuned config; method: [[SMDP Q-Learning over Hand-Defined Skills design]], plan [[2026-10-20-craftax-method-smdp-skills]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/smdp-skills/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-smdp-skills`, id `57f6b5a4b8bd467c9309200d88d968d3`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 73 case episodes, all replayed from the stored eval files.
- Related analyses: [[Dyna-Q training analysis]] (same skill library; Dyna-Q runs on the tuned smdp-skills knobs).
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- 1B: pending (final-1b not run, R75); sections marked 1B below are placeholders for the same tables at 1B.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 3 of this method's 10 tuned runs (3 of 40 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 443, detected 237, not logged 40; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Floor entries before or beside the final eval: `enter dungeon` unlocked in 4 of 10 seeds at some point (seeds with a training-episode unlock: 4, with an eval unlock: 0; final stochastic rate 0.00%). Source: `training-analysis-1m-smdp-skills`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (52 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (11): place stone, eat plant, make stone pickaxe, make stone sword, collect iron, collect diamond, make iron pickaxe, make iron sword, make arrow, make torch, place torch; intermediate (17): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-smdp-skills`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status_tuned.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 8): make wood sword 2.81% (10/10 seeds ever), make wood pickaxe 2.03% (9/10 seeds ever), defeat zombie 1.56% (9/10 seeds ever), defeat skeleton 0.47% (10/10 seeds ever), collect stone 0.00% (4/10 seeds ever), place furnace 0.00% (1/10 seeds ever), collect coal 0.00% (1/10 seeds ever), enter dungeon 0.00% (4/10 seeds ever). Same file.
- Common (7): collect sapling 82.0%, collect wood 60.5%, place plant 43.1%, wake up 31.1%, place table 29.2%, collect drink 17.2%, eat cow 5.2%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 317 (49.5%); starvation 182 (28.4%); arrow 125 (19.5%); dehydration 15 (2.3%); exhaustion 1 (0.2%). At checkpoint 0 (untrained): zombie 51.6%, starvation 28.1%, arrow 20.3%. Source: `training-analysis-1m-smdp-skills`, `hard/death_causes.csv`, plot `plots/death_causes_over_training_tuned.png`.
- Stalls (>= 300 steps without an unlock): 88 stalls in 640 final eval episodes, 0.138 per episode (seed interval 0.089 to 0.186), median length 329 steps, 2.3% escaped by an unlock. All started on floor 0; started by collect sapling 26 (29.5%), episode start 22 (25.0%), place plant 15 (17.0%), collect drink 10 (11.4%). Source: `training-analysis-1m-smdp-skills`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 2 by an unlock, 86 by death. Longest stall 393 steps (run `d9a4a243ace54998b26d15fbdb1a675b`, seed 6, eval seed 10006, steps 9 to 402, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-smdp-skills` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 88 stalls, 2 ended by an unlock, 86 by death (97.7%).
- Before the unlock (5 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 2 cases; most frequent actions right 20%, do 15%, left 12%; movement keys are 43% of the last 50 steps against 19% of the whole stall window, `do` 15% against 33%; the most frequent action of the last 50 differs from the window's in 3 of 5 cases.
- State at the unlock: health <= 4 in 5 of 5 cases (median health 2 at the unlock, 9 at the stall start), food 0 in 4 of 5 (median food 0, 9 at the start), drink 0 in 4 of 5.
- Achievements that ended the stalls: wake up 2, collect wood 2, eat cow 1.
- Deepest floor: the deep-floor cases are all overworld (floor 0) episodes, so the route question is moot at 1M.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-smdp-skills-stall-1`** (run `smdp-skills-1m-0b84c247-s6`, `d9a4a243ace54998b26d15fbdb1a675b`, checkpoint 244 (999,424 env steps), eval seed 10049, stochastic). Situation: a stall of 373 steps from step 15 on floor 0, begun after collect drink; at its start health 9, food 9, drink 9, energy 9, inventory sapling 1. Behaviour during the stall: most chosen actions do 70, left 43, make torch 41 over the 373 steps; 0.56% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 4. What broke it: wake up at step 388 (then health 4, food 0, drink 1, energy 9, inventory sapling 1). Outcome: the episode lasted 450 steps. Replay: `gifs/1m-smdp-skills-stall-1.gif`, `overlays/1m-smdp-skills-stall-1.mp4`, `masters/case-1m-smdp-skills-stall-1.mkv`, `rollouts/case-1m-smdp-skills-stall-1.npz`.

**`1m-smdp-skills-stall-2`** (run `smdp-skills-1m-0b84c247-s2`, `e49e12f658554df6a1fc4b55a7ababe7`, checkpoint 244 (999,424 env steps), eval seed 10018, stochastic). Situation: a stall of 331 steps from step 14 on floor 0, begun after collect sapling; at its start health 9, food 9, drink 9, energy 9, inventory wood 1, sapling 1. Behaviour during the stall: most chosen actions do 177, make stone pickaxe 56, left 2 over the 331 steps; 0.91% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: wake up at step 345 (then health 3, food 0, drink 0, energy 9, inventory wood 3, sapling 18). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-smdp-skills-stall-2.gif`, `overlays/1m-smdp-skills-stall-2.mp4`, `masters/case-1m-smdp-skills-stall-2.mkv`, `rollouts/case-1m-smdp-skills-stall-2.npz`.

**`1m-smdp-skills-stall-3`** (run `smdp-skills-1m-0b84c247-s9`, `0f7fbc363bd248698fbc57c386419bda`, checkpoint 1 (4,096 env steps), eval seed 10048, stochastic). Situation: a stall of 317 steps from step 66 on floor 0; at its start health 9, food 7, drink 7, energy 9, inventory empty. Behaviour during the stall: most chosen actions make iron armour 11, make torch 11, drink potion pink 9 over the 317 steps; 1.52% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: collect wood at step 383 (then health 2, food 0, drink 0, energy 9, inventory wood 1). Outcome: the episode lasted 400 steps. Replay: `gifs/1m-smdp-skills-stall-3.gif`, `overlays/1m-smdp-skills-stall-3.mp4`, `masters/case-1m-smdp-skills-stall-3.mkv`, `rollouts/case-1m-smdp-skills-stall-3.npz`.

**`1m-smdp-skills-stall-4`** (run `smdp-skills-1m-0b84c247-s5`, `3d8c68bf2142405b93bc12c46d258aae`, checkpoint 7 (28,672 env steps), eval seed 10012, stochastic). Situation: a stall of 325 steps from step 45 on floor 0; at its start health 9, food 8, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions right 26, sleep 25, make stone sword 24 over the 325 steps; 1.48% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: collect wood at step 370 (then health 1, food 0, drink 0, energy 6, inventory wood 1). Outcome: the episode lasted 388 steps. Replay: `gifs/1m-smdp-skills-stall-4.gif`, `overlays/1m-smdp-skills-stall-4.mp4`, `masters/case-1m-smdp-skills-stall-4.mkv`, `rollouts/case-1m-smdp-skills-stall-4.npz`.

**`1m-smdp-skills-stall-5`** (run `smdp-skills-1m-0b84c247-s0`, `98e117c14cee449c996ab818c439561a`, checkpoint 148 (606,208 env steps), eval seed 10035, stochastic). Situation: a stall of 310 steps from step 11 on floor 0; at its start health 9, food 9, drink 9, energy 9, inventory wood 1, sapling 1. Behaviour during the stall: most chosen actions do 196, down 54, up 51 over the 310 steps; 0.52% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: eat cow at step 321 (then health 2, food 6, drink 0, energy 0, inventory wood 1, sapling 22). Outcome: the episode lasted 338 steps. Replay: `gifs/1m-smdp-skills-stall-5.gif`, `overlays/1m-smdp-skills-stall-5.mp4`, `masters/case-1m-smdp-skills-stall-5.mkv`, `rollouts/case-1m-smdp-skills-stall-5.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-smdp-skills-deep_floor-1`** (run `smdp-skills-1m-0b84c247-s1`, `f6ce810bea344edaac5fd906598f8b0b`, checkpoint 244 (999,424 env steps), eval seed 10056, stochastic). Situation: an episode with 8 achievements unlocked on floor 0; window steps 133 to 283, at its start health 9, food 7, drink 5, energy 8, inventory sapling 1. Behaviour: most chosen actions down 41, do 29, up 28; 1.00% of overworld tiles first seen in the window; combat damage 6, close calls 0, lowest health 2. At step 283 (the last unlock) health 2, food 2, drink 0, energy 5, inventory sapling 1. Outcome: the episode lasted 307 steps. Replay: `gifs/1m-smdp-skills-deep_floor-1.gif`, `masters/case-1m-smdp-skills-deep_floor-1.mkv`, `rollouts/case-1m-smdp-skills-deep_floor-1.npz`.

**`1m-smdp-skills-deep_floor-2`** (run `smdp-skills-1m-0b84c247-s4`, `6eb4b6a30d1d450fb5ca4516f1ae74c2`, checkpoint 244 (999,424 env steps), eval seed 10008, stochastic). Situation: an episode with 7 achievements unlocked on floor 0; window steps 136 to 286, at its start health 9, food 5, drink 4, energy 9, inventory wood 2. Behaviour: most chosen actions do 40, up 27, down 20; 2.04% of overworld tiles first seen in the window; combat damage 2, close calls 0, lowest health 5. At step 286 (the last unlock) health 5, food 6, drink 0, energy 5, inventory wood 1, sapling 2. Outcome: the episode lasted 357 steps. Replay: `gifs/1m-smdp-skills-deep_floor-2.gif`, `masters/case-1m-smdp-skills-deep_floor-2.mkv`, `rollouts/case-1m-smdp-skills-deep_floor-2.npz`.

**`1m-smdp-skills-deep_floor-3`** (run `smdp-skills-1m-0b84c247-s4`, `6eb4b6a30d1d450fb5ca4516f1ae74c2`, checkpoint 244 (999,424 env steps), eval seed 10029, stochastic). Situation: an episode with 7 achievements unlocked on floor 0; window steps 179 to 329, at its start health 9, food 3, drink 9, energy 4, inventory wood 1. Behaviour: most chosen actions down 22, do 19, level up strength 10; 3.30% of overworld tiles first seen in the window; combat damage 4, close calls 0, lowest health 2. At step 329 (the last unlock) health 2, food 0, drink 4, energy 8, inventory empty. Outcome: the episode lasted 348 steps. Replay: `gifs/1m-smdp-skills-deep_floor-3.gif`, `masters/case-1m-smdp-skills-deep_floor-3.mkv`, `rollouts/case-1m-smdp-skills-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-smdp-skills-rare_unlock-1`** (run `smdp-skills-1m-0b84c247-s6`, `d9a4a243ace54998b26d15fbdb1a675b`, checkpoint 244 (999,424 env steps), eval seed 10008, stochastic). Situation: make wood pickaxe (rare at the final checkpoint) is unlocked at step 34; window steps 0 to 34, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions up 17, do 7, left 6; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 8, drink 8, energy 8, inventory empty. Outcome: the episode lasted 74 steps. Replay: `gifs/1m-smdp-skills-rare_unlock-1.gif`, `masters/case-1m-smdp-skills-rare_unlock-1.mkv`, `rollouts/case-1m-smdp-skills-rare_unlock-1.npz`.

**`1m-smdp-skills-rare_unlock-2`** (run `smdp-skills-1m-0b84c247-s2`, `e49e12f658554df6a1fc4b55a7ababe7`, checkpoint 244 (999,424 env steps), eval seed 10054, stochastic). Situation: make wood sword (rare at the final checkpoint) is unlocked at step 33; window steps 0 to 33, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions up 20, do 4, right 3; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 8, drink 8, energy 8, inventory wood 1. Outcome: the episode lasted 256 steps. Replay: `gifs/1m-smdp-skills-rare_unlock-2.gif`, `masters/case-1m-smdp-skills-rare_unlock-2.mkv`, `rollouts/case-1m-smdp-skills-rare_unlock-2.npz`.

**`1m-smdp-skills-rare_unlock-3`** (run `smdp-skills-1m-0b84c247-s4`, `6eb4b6a30d1d450fb5ca4516f1ae74c2`, checkpoint 244 (999,424 env steps), eval seed 10047, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 32; window steps 0 to 32, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 14, up 13, place plant 2; combat damage 6, close calls 0, lowest health 4. At the unlock health 4, food 8, drink 8, energy 8, inventory sapling 1. Outcome: the episode lasted 261 steps. Replay: `gifs/1m-smdp-skills-rare_unlock-3.gif`, `masters/case-1m-smdp-skills-rare_unlock-3.mkv`, `rollouts/case-1m-smdp-skills-rare_unlock-3.npz`.

**`1m-smdp-skills-rare_unlock-4`** (run `smdp-skills-1m-0b84c247-s8`, `648455a711594441acd8c3be28c61ac5`, checkpoint 7 (28,672 env steps), eval seed 10014, stochastic). Situation: collect stone (rare at the final checkpoint) is unlocked at step 121; window steps 0 to 121, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions place furnace 9, make stone pickaxe 7, up 6; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 5, drink 4, energy 9, inventory stone 1. Outcome: the episode lasted 388 steps. Replay: `gifs/1m-smdp-skills-rare_unlock-4.gif`, `masters/case-1m-smdp-skills-rare_unlock-4.mkv`, `rollouts/case-1m-smdp-skills-rare_unlock-4.npz`.

**`1m-smdp-skills-rare_unlock-5`** (run `smdp-skills-1m-0b84c247-s4`, `6eb4b6a30d1d450fb5ca4516f1ae74c2`, checkpoint 244 (999,424 env steps), eval seed 10028, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 127; window steps 0 to 127, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 64, place plant 40, right 5; combat damage 2, close calls 0, lowest health 7. At the unlock health 8, food 5, drink 4, energy 8, inventory wood 3, sapling 11. Outcome: the episode lasted 228 steps. Replay: `gifs/1m-smdp-skills-rare_unlock-5.gif`, `masters/case-1m-smdp-skills-rare_unlock-5.mkv`, `rollouts/case-1m-smdp-skills-rare_unlock-5.npz`.

**`1m-smdp-skills-rare_unlock-6`** (run `smdp-skills-1m-0b84c247-s8`, `648455a711594441acd8c3be28c61ac5`, checkpoint 7 (28,672 env steps), eval seed 10014, stochastic). Situation: place furnace (rare at the final checkpoint) is unlocked at step 124; window steps 0 to 124, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions place furnace 10, make stone pickaxe 7, up 6; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 5, drink 4, energy 9, inventory empty. Outcome: the episode lasted 388 steps. Replay: `gifs/1m-smdp-skills-rare_unlock-6.gif`, `masters/case-1m-smdp-skills-rare_unlock-6.mkv`, `rollouts/case-1m-smdp-skills-rare_unlock-6.npz`.

**`1m-smdp-skills-rare_unlock-7`** (run `smdp-skills-1m-0b84c247-s8`, `648455a711594441acd8c3be28c61ac5`, checkpoint 7 (28,672 env steps), eval seed 10014, stochastic). Situation: collect coal (rare at the final checkpoint) is unlocked at step 227; window steps 77 to 227, at its start health 9, food 7, drink 6, energy 7, inventory wood 3. Behaviour: most chosen actions place furnace 20, make stone pickaxe 16, left 9; combat damage 0, close calls 0, lowest health 9. At the unlock health 9, food 1, drink 0, energy 6, inventory coal 1. Outcome: the episode lasted 388 steps. Replay: `gifs/1m-smdp-skills-rare_unlock-7.gif`, `masters/case-1m-smdp-skills-rare_unlock-7.mkv`, `rollouts/case-1m-smdp-skills-rare_unlock-7.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-smdp-skills-death-1`** (run `smdp-skills-1m-0b84c247-s9`, `0f7fbc363bd248698fbc57c386419bda`, checkpoint 244 (999,424 env steps), eval seed 10009, stochastic). Situation: death by zombie at step 226; the window (steps 76 to 226) opens with health 9, food 7, drink 6, energy 7, inventory wood 1, sapling 2. Behaviour: most chosen actions place stone 134, enchant armour 14, descend 1; combat damage 8, close calls 0, lowest health 0. At death health 0, food 1, drink 0, energy 2, inventory wood 1, sapling 2. Outcome: the episode ended after 226 steps. Replay: `gifs/1m-smdp-skills-death-1.gif`, `masters/case-1m-smdp-skills-death-1.mkv`, `rollouts/case-1m-smdp-skills-death-1.npz`.

**`1m-smdp-skills-death-2`** (run `smdp-skills-1m-0b84c247-s9`, `0f7fbc363bd248698fbc57c386419bda`, checkpoint 244 (999,424 env steps), eval seed 10004, stochastic). Situation: death by starvation at step 338; the window (steps 188 to 338) opens with health 9, food 2, drink 1, energy 3, inventory sapling 5. Behaviour: most chosen actions place stone 78, cast fireball 72; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 0, inventory sapling 5. Outcome: the episode ended after 338 steps. Replay: `gifs/1m-smdp-skills-death-2.gif`, `masters/case-1m-smdp-skills-death-2.mkv`, `rollouts/case-1m-smdp-skills-death-2.npz`.

**`1m-smdp-skills-death-3`** (run `smdp-skills-1m-0b84c247-s6`, `d9a4a243ace54998b26d15fbdb1a675b`, checkpoint 244 (999,424 env steps), eval seed 10020, stochastic). Situation: death by arrow at step 252; the window (steps 102 to 252) opens with health 9, food 6, drink 5, energy 6, inventory empty. Behaviour: most chosen actions do 16, drink potion pink 15, up 13; combat damage 11, close calls 0, lowest health 0. At death health 0, food 1, drink 0, energy 9, inventory wood 2. Outcome: the episode ended after 252 steps. Replay: `gifs/1m-smdp-skills-death-3.gif`, `masters/case-1m-smdp-skills-death-3.mkv`, `rollouts/case-1m-smdp-skills-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-smdp-skills`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting, skill interference, deep-skill starvation, survival failure despite flags fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 10 / 10 | 16.4k to 999k | 3 drops, deepest drawdown 1.80 (floor 0.34) (run 98e117c14cee449c996ab818c439561a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-stall-5.gif` |
| Collapse | 1M | 8 / 10 | 16.4k to 999k | drawdown 1.80, final 1.17 of best 1.37 (floor 0.34) (run 98e117c14cee449c996ab818c439561a; 7 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-stall-5.gif` |
| Plateau | 1M | 1 / 10 | 500k to 999k | score gain -0.39 from 500k to the end (< 0.04) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-death-1.gif` |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 991k | >= 20% to < 5%: collect_drink (111k to 123k), collect_sapling (45.1k to 131k), collect_wood (28.7k to 36.9k), defeat_zombie (545k to 553k), eat_cow (508k to 516k), place_plant (45.1k to 53.2k), place_table (53.2k to 65.5k), wake_up (20.5k to 319k) (run 98e117c14cee449c996ab818c439561a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Stuck top level | 1M | 1 / 10 | 725k to 868k | success < 5% with > 20% of decisions: descend | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Skill interference | 1M | 10 / 10 | 4.1k to 999k | success from >= 20% to < 5%: collect_wood (4.1k to 81.9k; descend chosen +10%), drink (414k to 422k; unlock_any chosen +18%), eat (549k to 557k; fight chosen +15%), place_table (897k to 926k; eat chosen +7%), sleep (45.1k to 90.1k; collect_wood chosen +19%), unlock_any (4.1k to 49.2k; descend chosen +12%) (run 98e117c14cee449c996ab818c439561a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Deep-skill starvation | 1M | 10 / 10 |  | deep skills chosen fewer than 100 times: collect_diamond about 0, make_iron_pickaxe about 0 (run 98e117c14cee449c996ab818c439561a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Survival failure despite flags | 1M | 10 / 10 | 999k to 999k | 54% of 175 low-stat interruptions followed by death within 100 steps (run 98e117c14cee449c996ab818c439561a; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/death_causes_over_training_tuned.png` |

- Absent in every run: Entropy collapse, Numerical failure, Reward hacking, Exploration failure, Survival failure, Fallback collapse, Timeout dominance, Key aliasing, Blind skill test.
- Detector verdicts over the tuned runs (180 run-detector pairs): absent 110, detected 60, not logged 10; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

### 1B

1B: pending (final-1b not run, R75). The 1B obstacles table (3 seeds) is added by rerunning `analysis.training` once the 1B finals exist.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[SMDP skills training analysis]] (this note): floors - floor 0 only; achievements - 52 never, 8 rare, 7 common; killer and stalls - zombie 49.5% of deaths, 0.138 stalls per episode (median 329 steps, 2.3% escaped); final score 0.831.
- [[PQN-RNN training analysis]] (base control): floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.
- [[PPO training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PPO-RNN training analysis]]: floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.

## Ablations

Ablation units of `final-1m` (R77): reported apart from the tuned config, never pooled into the sections above. The change is an override of the tuned config in `configs/phases/final-1m.yaml`.

| Unit | Change | n | Final score (sd) | Stochastic eval | Floors reached | Never / rare / common achievements | Top death causes | Stalls per episode (median, escape) |
|---|---|---|---|---|---|---|---|---|
| tuned | - | 10 | 0.831 (0.651) | 0.819 | floor 0 only | 52 / 8 / 7 | zombie 49.5%, starvation 28.4% | 0.138 (329, 2.3%) |
| monte-carlo | `monte_carlo: true` (Monte Carlo top level) | 10 | 0.478 (0.482) | 0.468 | floor 0 only | 52 / 10 / 5 | zombie 47.0%, starvation 35.9% | 0.170 (331, 4.6%) |
| no-flags | `low_flags: false` (no low-stat flags) | 10 | 0.446 (0.531) | 0.479 | floor 0 only | 55 / 5 / 7 | zombie 47.0%, starvation 30.8% | 0.172 (338, 0.9%) |
| random-top | `eps_top_end: 1.0` (random top level) | 10 | 0.274 (0.399) | 0.286 | floor 0 only | 52 / 9 / 6 | zombie 50.3%, starvation 34.7% | 0.194 (333, 0.8%) |

Sources: `hard/*.csv` (rows of each `config_ref`) and the scores of `outputs/metrics/final-1m/runs.parquet`. The `tuned` row repeats the main sections for comparison.

### Obstacles: monte-carlo

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 10 / 10 | 16.4k to 999k | 6 drops, deepest drawdown 1.83 (floor 0.16) (run 1ce311f8e41b493a9574c1fe7b5f1108; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-monte-carlo-stall-3.gif` |
| Collapse | 1M | 9 / 10 | 16.4k to 999k | drawdown 1.83, final 0.30 of best 0.65 (floor 0.16) (run 1ce311f8e41b493a9574c1fe7b5f1108; 8 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-monte-carlo-stall-3.gif` |
| Plateau | 1M | 2 / 10 | 500k to 999k | score gain 0.04 from 500k to the end (< 0.06) (run 1ce311f8e41b493a9574c1fe7b5f1108; 1 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-monte-carlo-stall-3.gif` |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 868k | >= 20% to < 5%: collect_drink (111k to 123k), collect_sapling (45.1k to 135k), collect_wood (28.7k to 143k), place_plant (45.1k to 77.8k), wake_up (20.5k to 193k) (run 1ce311f8e41b493a9574c1fe7b5f1108; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Stuck top level | 1M | 1 / 10 | 324k to 496k | success < 5% with > 20% of decisions: descend | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Skill interference | 1M | 10 / 10 | 4.1k to 967k | success from >= 20% to < 5%: collect_wood (4.1k to 57.3k; descend chosen +30%), drink (184k to 193k; sleep chosen +5%), sleep (45.1k to 86k; unlock_any chosen +15%), unlock_any (4.1k to 115k; sleep chosen +18%) (run 1ce311f8e41b493a9574c1fe7b5f1108; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Deep-skill starvation | 1M | 10 / 10 |  | deep skills chosen fewer than 100 times: collect_diamond about 0, make_iron_pickaxe about 0 (run 1ce311f8e41b493a9574c1fe7b5f1108; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Survival failure despite flags | 1M | 10 / 10 | 999k to 999k | 36% of 90 low-stat interruptions followed by death within 100 steps (run 1ce311f8e41b493a9574c1fe7b5f1108; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/death_causes_over_training_tuned.png` |

Cases of monte-carlo (`cases.csv`, replay files under `training-analysis-1m-smdp-skills`):

| Case | Kind | Detail | Run | Checkpoint | Eval seed | GIF |
|---|---|---|---|---|---|---|
| `1m-smdp-skills-monte-carlo-stall-1` | stall | 333 steps | `smdp-skills-1m-20ec0807-s3` | 244 | 10003 | `gifs/1m-smdp-skills-monte-carlo-stall-1.gif` |
| `1m-smdp-skills-monte-carlo-stall-2` | stall | 333 steps | `smdp-skills-1m-20ec0807-s6` | 244 | 10003 | `gifs/1m-smdp-skills-monte-carlo-stall-2.gif` |
| `1m-smdp-skills-monte-carlo-stall-3` | stall | 318 steps | `smdp-skills-1m-20ec0807-s0` | 244 | 10035 | `gifs/1m-smdp-skills-monte-carlo-stall-3.gif` |
| `1m-smdp-skills-monte-carlo-stall-4` | stall | 318 steps | `smdp-skills-1m-20ec0807-s5` | 244 | 10006 | `gifs/1m-smdp-skills-monte-carlo-stall-4.gif` |
| `1m-smdp-skills-monte-carlo-stall-5` | stall | 308 steps | `smdp-skills-1m-20ec0807-s7` | 244 | 10014 | `gifs/1m-smdp-skills-monte-carlo-stall-5.gif` |
| `1m-smdp-skills-monte-carlo-stall-6` | stall | 317 steps | `smdp-skills-1m-20ec0807-s3` | 1 | 10048 | `gifs/1m-smdp-skills-monte-carlo-stall-6.gif` |
| `1m-smdp-skills-monte-carlo-stall-7` | stall | 384 steps | `smdp-skills-1m-20ec0807-s3` | 12 | 10009 | `gifs/1m-smdp-skills-monte-carlo-stall-7.gif` |
| `1m-smdp-skills-monte-carlo-stall-8` | stall | 382 steps | `smdp-skills-1m-20ec0807-s7` | 148 | 10022 | `gifs/1m-smdp-skills-monte-carlo-stall-8.gif` |
| `1m-smdp-skills-monte-carlo-deep_floor-1` | deep_floor | floor 0 | `smdp-skills-1m-20ec0807-s9` | 12 | 10011 | `gifs/1m-smdp-skills-monte-carlo-deep_floor-1.gif` |
| `1m-smdp-skills-monte-carlo-deep_floor-2` | deep_floor | floor 0 | `smdp-skills-1m-20ec0807-s9` | 12 | 10047 | `gifs/1m-smdp-skills-monte-carlo-deep_floor-2.gif` |
| `1m-smdp-skills-monte-carlo-deep_floor-3` | deep_floor | floor 0 | `smdp-skills-1m-20ec0807-s8` | 7 | 10014 | `gifs/1m-smdp-skills-monte-carlo-deep_floor-3.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-1` | rare_unlock | place_table | `smdp-skills-1m-20ec0807-s5` | 244 | 10054 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-1.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-2` | rare_unlock | eat_cow | `smdp-skills-1m-20ec0807-s8` | 244 | 10060 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-2.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-3` | rare_unlock | make_wood_pickaxe | `smdp-skills-1m-20ec0807-s8` | 7 | 10014 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-3.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-4` | rare_unlock | make_wood_sword | `smdp-skills-1m-20ec0807-s9` | 7 | 10032 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-4.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-5` | rare_unlock | defeat_zombie | `smdp-skills-1m-20ec0807-s0` | 244 | 10047 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-5.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-6` | rare_unlock | collect_stone | `smdp-skills-1m-20ec0807-s8` | 7 | 10014 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-6.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-7` | rare_unlock | defeat_skeleton | `smdp-skills-1m-20ec0807-s0` | 244 | 10000 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-7.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-8` | rare_unlock | place_furnace | `smdp-skills-1m-20ec0807-s8` | 7 | 10014 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-8.gif` |
| `1m-smdp-skills-monte-carlo-rare_unlock-9` | rare_unlock | collect_coal | `smdp-skills-1m-20ec0807-s8` | 7 | 10014 | `gifs/1m-smdp-skills-monte-carlo-rare_unlock-9.gif` |
| `1m-smdp-skills-monte-carlo-death-1` | death | zombie | `smdp-skills-1m-20ec0807-s3` | 244 | 10051 | `gifs/1m-smdp-skills-monte-carlo-death-1.gif` |
| `1m-smdp-skills-monte-carlo-death-2` | death | starvation | `smdp-skills-1m-20ec0807-s3` | 244 | 10001 | `gifs/1m-smdp-skills-monte-carlo-death-2.gif` |
| `1m-smdp-skills-monte-carlo-death-3` | death | arrow | `smdp-skills-1m-20ec0807-s3` | 244 | 10017 | `gifs/1m-smdp-skills-monte-carlo-death-3.gif` |

### Obstacles: no-flags

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 10 / 10 | 16.4k to 999k | 1 drops, deepest drawdown 1.60 (floor 0.16) (run bdb382f4cac8417085bd8c0c9b52d895; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-no-flags-death-3.gif` |
| Collapse | 1M | 5 / 10 | 16.4k to 999k | drawdown 1.60, final -0.21 of best 0.66 (floor 0.16) (run bdb382f4cac8417085bd8c0c9b52d895; 4 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-no-flags-death-3.gif` |
| Plateau | 1M | 1 / 10 | 500k to 999k | score gain -0.34 from 500k to the end (< 0.09) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-no-flags-death-1.gif` |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 991k | >= 20% to < 5%: collect_drink (106k to 123k), collect_sapling (36.9k to 106k), collect_wood (28.7k to 36.9k), place_plant (49.2k to 102k), wake_up (20.5k to 172k) (run bdb382f4cac8417085bd8c0c9b52d895; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Fallback collapse | 1M | 6 / 10 | 307k to 1M | fight + unlock_any take 79% of decisions after eps decay (run 5c9b6299e486411f8b2c19a1dce1ef97; 5 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-no-flags-rare_unlock-3.gif` |
| Skill interference | 1M | 10 / 10 | 4.1k to 938k | success from >= 20% to < 5%: collect_wood (4.1k to 24.6k; fight chosen +46%), sleep (217k to 225k; drink chosen +34%), unlock_any (4.1k to 111k; descend chosen +18%) (run bdb382f4cac8417085bd8c0c9b52d895; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Deep-skill starvation | 1M | 10 / 10 |  | deep skills chosen fewer than 100 times: collect_diamond about 0, make_iron_pickaxe about 0 (run bdb382f4cac8417085bd8c0c9b52d895; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |

Cases of no-flags (`cases.csv`, replay files under `training-analysis-1m-smdp-skills`):

| Case | Kind | Detail | Run | Checkpoint | Eval seed | GIF |
|---|---|---|---|---|---|---|
| `1m-smdp-skills-no-flags-stall-1` | stall | 308 steps | `smdp-skills-1m-89f3c98f-s3` | 244 | 10010 | `gifs/1m-smdp-skills-no-flags-stall-1.gif` |
| `1m-smdp-skills-no-flags-stall-2` | stall | 317 steps | `smdp-skills-1m-89f3c98f-s8` | 1 | 10048 | `gifs/1m-smdp-skills-no-flags-stall-2.gif` |
| `1m-smdp-skills-no-flags-stall-3` | stall | 325 steps | `smdp-skills-1m-89f3c98f-s5` | 7 | 10012 | `gifs/1m-smdp-skills-no-flags-stall-3.gif` |
| `1m-smdp-skills-no-flags-stall-4` | stall | 349 steps | `smdp-skills-1m-89f3c98f-s3` | 148 | 10018 | `gifs/1m-smdp-skills-no-flags-stall-4.gif` |
| `1m-smdp-skills-no-flags-deep_floor-1` | deep_floor | floor 0 | `smdp-skills-1m-89f3c98f-s7` | 244 | 10052 | `gifs/1m-smdp-skills-no-flags-deep_floor-1.gif` |
| `1m-smdp-skills-no-flags-deep_floor-2` | deep_floor | floor 0 | `smdp-skills-1m-89f3c98f-s4` | 244 | 10022 | `gifs/1m-smdp-skills-no-flags-deep_floor-2.gif` |
| `1m-smdp-skills-no-flags-deep_floor-3` | deep_floor | floor 0 | `smdp-skills-1m-89f3c98f-s5` | 12 | 10025 | `gifs/1m-smdp-skills-no-flags-deep_floor-3.gif` |
| `1m-smdp-skills-no-flags-rare_unlock-1` | rare_unlock | make_wood_pickaxe | `smdp-skills-1m-89f3c98f-s4` | 244 | 10022 | `gifs/1m-smdp-skills-no-flags-rare_unlock-1.gif` |
| `1m-smdp-skills-no-flags-rare_unlock-2` | rare_unlock | make_wood_sword | `smdp-skills-1m-89f3c98f-s6` | 244 | 10002 | `gifs/1m-smdp-skills-no-flags-rare_unlock-2.gif` |
| `1m-smdp-skills-no-flags-rare_unlock-3` | rare_unlock | defeat_zombie | `smdp-skills-1m-89f3c98f-s1` | 244 | 10024 | `gifs/1m-smdp-skills-no-flags-rare_unlock-3.gif` |
| `1m-smdp-skills-no-flags-rare_unlock-4` | rare_unlock | defeat_skeleton | `smdp-skills-1m-89f3c98f-s2` | 244 | 10000 | `gifs/1m-smdp-skills-no-flags-rare_unlock-4.gif` |
| `1m-smdp-skills-no-flags-death-1` | death | zombie | `smdp-skills-1m-89f3c98f-s2` | 244 | 10038 | `gifs/1m-smdp-skills-no-flags-death-1.gif` |
| `1m-smdp-skills-no-flags-death-2` | death | starvation | `smdp-skills-1m-89f3c98f-s8` | 244 | 10006 | `gifs/1m-smdp-skills-no-flags-death-2.gif` |
| `1m-smdp-skills-no-flags-death-3` | death | arrow | `smdp-skills-1m-89f3c98f-s0` | 244 | 10009 | `gifs/1m-smdp-skills-no-flags-death-3.gif` |

### Obstacles: random-top

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 10 / 10 | 16.4k to 999k | 1 drops, deepest drawdown 1.96 (floor 0.1) (run cd256c896311415481814a915bd485e6; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-random-top-death-1.gif` |
| Collapse | 1M | 10 / 10 | 16.4k to 999k | drawdown 1.96, final 0.29 of best 0.41 (floor 0.1) (run cd256c896311415481814a915bd485e6; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-random-top-death-1.gif` |
| Plateau | 1M | 3 / 10 | 500k to 999k | score gain -0.03 from 500k to the end (< 0.04) (run 6b50afddc260447ca91d22428dbed8a8; 2 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-smdp-skills-random-top-death-3.gif` |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 12.3k to 885k | >= 20% to < 5%: collect_drink (111k to 123k), collect_sapling (45.1k to 119k), collect_wood (28.7k to 152k), place_plant (45.1k to 77.8k), wake_up (20.5k to 180k) (run cd256c896311415481814a915bd485e6; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Skill interference | 1M | 10 / 10 | 4.1k to 983k | success from >= 20% to < 5%: collect_wood (4.1k to 73.7k; fight chosen +17%), sleep (45.1k to 86k; unlock_any chosen +5%), unlock_any (4.1k to 119k; descend chosen +20%) (run cd256c896311415481814a915bd485e6; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Deep-skill starvation | 1M | 10 / 10 |  | deep skills chosen fewer than 100 times: collect_diamond about 0, make_iron_pickaxe about 0 (run cd256c896311415481814a915bd485e6; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |
| Survival failure despite flags | 1M | 10 / 10 | 999k to 999k | 50% of 135 low-stat interruptions followed by death within 100 steps (run cd256c896311415481814a915bd485e6; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/death_causes_over_training_tuned.png` |

Cases of random-top (`cases.csv`, replay files under `training-analysis-1m-smdp-skills`):

| Case | Kind | Detail | Run | Checkpoint | Eval seed | GIF |
|---|---|---|---|---|---|---|
| `1m-smdp-skills-random-top-stall-1` | stall | 372 steps | `smdp-skills-1m-1f559f2c-s5` | 244 | 10005 | `gifs/1m-smdp-skills-random-top-stall-1.gif` |
| `1m-smdp-skills-random-top-stall-2` | stall | 317 steps | `smdp-skills-1m-1f559f2c-s3` | 1 | 10048 | `gifs/1m-smdp-skills-random-top-stall-2.gif` |
| `1m-smdp-skills-random-top-stall-3` | stall | 384 steps | `smdp-skills-1m-1f559f2c-s3` | 12 | 10009 | `gifs/1m-smdp-skills-random-top-stall-3.gif` |
| `1m-smdp-skills-random-top-stall-4` | stall | 309 steps | `smdp-skills-1m-1f559f2c-s6` | 148 | 10059 | `gifs/1m-smdp-skills-random-top-stall-4.gif` |
| `1m-smdp-skills-random-top-deep_floor-1` | deep_floor | floor 0 | `smdp-skills-1m-1f559f2c-s8` | 7 | 10014 | `gifs/1m-smdp-skills-random-top-deep_floor-1.gif` |
| `1m-smdp-skills-random-top-deep_floor-2` | deep_floor | floor 0 | `smdp-skills-1m-1f559f2c-s7` | 12 | 10002 | `gifs/1m-smdp-skills-random-top-deep_floor-2.gif` |
| `1m-smdp-skills-random-top-deep_floor-3` | deep_floor | floor 0 | `smdp-skills-1m-1f559f2c-s7` | 12 | 10040 | `gifs/1m-smdp-skills-random-top-deep_floor-3.gif` |
| `1m-smdp-skills-random-top-rare_unlock-1` | rare_unlock | eat_cow | `smdp-skills-1m-1f559f2c-s9` | 244 | 10020 | `gifs/1m-smdp-skills-random-top-rare_unlock-1.gif` |
| `1m-smdp-skills-random-top-rare_unlock-2` | rare_unlock | make_wood_pickaxe | `smdp-skills-1m-1f559f2c-s9` | 244 | 10056 | `gifs/1m-smdp-skills-random-top-rare_unlock-2.gif` |
| `1m-smdp-skills-random-top-rare_unlock-3` | rare_unlock | make_wood_sword | `smdp-skills-1m-1f559f2c-s3` | 244 | 10008 | `gifs/1m-smdp-skills-random-top-rare_unlock-3.gif` |
| `1m-smdp-skills-random-top-rare_unlock-4` | rare_unlock | defeat_zombie | `smdp-skills-1m-1f559f2c-s5` | 244 | 10047 | `gifs/1m-smdp-skills-random-top-rare_unlock-4.gif` |
| `1m-smdp-skills-random-top-rare_unlock-5` | rare_unlock | collect_stone | `smdp-skills-1m-1f559f2c-s8` | 7 | 10014 | `gifs/1m-smdp-skills-random-top-rare_unlock-5.gif` |
| `1m-smdp-skills-random-top-rare_unlock-6` | rare_unlock | defeat_skeleton | `smdp-skills-1m-1f559f2c-s5` | 244 | 10035 | `gifs/1m-smdp-skills-random-top-rare_unlock-6.gif` |
| `1m-smdp-skills-random-top-rare_unlock-7` | rare_unlock | place_furnace | `smdp-skills-1m-1f559f2c-s8` | 7 | 10014 | `gifs/1m-smdp-skills-random-top-rare_unlock-7.gif` |
| `1m-smdp-skills-random-top-rare_unlock-8` | rare_unlock | collect_coal | `smdp-skills-1m-1f559f2c-s8` | 7 | 10014 | `gifs/1m-smdp-skills-random-top-rare_unlock-8.gif` |
| `1m-smdp-skills-random-top-death-1` | death | zombie | `smdp-skills-1m-1f559f2c-s6` | 244 | 10045 | `gifs/1m-smdp-skills-random-top-death-1.gif` |
| `1m-smdp-skills-random-top-death-2` | death | starvation | `smdp-skills-1m-1f559f2c-s3` | 244 | 10001 | `gifs/1m-smdp-skills-random-top-death-2.gif` |
| `1m-smdp-skills-random-top-death-3` | death | arrow | `smdp-skills-1m-1f559f2c-s1` | 244 | 10038 | `gifs/1m-smdp-skills-random-top-death-3.gif` |
