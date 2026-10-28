---
title: "PQN-RNN training analysis"
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
# PQN-RNN training analysis

## Summary

- 1M, 10 seeds, tuned config: final training score 1.356 (sd 0.508), stochastic eval score 1.268; at the final checkpoint only floor 0 (overworld) reached; floors 1-8 (dungeon to graveyard) never reached in any of 640 final eval episodes.
- Achievements: 54 of 67 never unlocked by any seed at any checkpoint (basic 12, intermediate 18, advanced 15, very advanced 9), 6 rare (final rate < 5%), 7 common.
- Deaths at the final checkpoint: zombie 46.6%, starvation 33.6%, arrow 17.7%, dehydration 2.2% (the untrained policy: zombie 48.4%, starvation 37.5%, arrow 12.5%, dehydration 1.6%).
- Stalls: 105 stalls in 640 final eval episodes, 0.164 per episode (seed interval 0.136 to 0.194), median length 320 steps, 4.8% escaped by an unlock; they are started by wake up (41%) and place plant (21%).
- Obstacles at 1M: instability 10/10, collapse 4/10, plateau 2/10; not logged (R78): entropy collapse, exploration failure, reward hacking, value failure. Caveat: skill forgetting 10/10 fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

## Scope

- Craftax-1M results only: `final-1m`, PQN-RNN (control, base of count-bonus-pqn-rnn, smdp-skills and dyna-q), 10 seeds (0-9) of the tuned config; method: [[PQN-RNN design]], plan [[2026-10-20-craftax-method-pqn-rnn]]; analysis of [[2026-10-20-craftax-training-analysis]] over the Craftax floors ([[Craftax floors and progression]]) and achievements ([[Craftax achievements]]).
- Numbers come from `outputs/analysis/1m/pqn-rnn/` (written 2026-10-28 by `analysis.training`, logged as MLflow run `training-analysis-1m-pqn-rnn`, id `a7b67d217dfd48abba749a7b80799119`, experiment `analysis`) and from `outputs/metrics/final-1m` (`runs.parquet`: `final_score` is the training-episode score in % of the maximum, `eval_stochastic_score` the stochastic final eval). 19 case episodes, all replayed from the stored eval files.
- Final eval = checkpoint 244 (999,424 env steps), stochastic mode, 64 eval seeds per run.
- 1B: pending (final-1b not run, R75); sections marked 1B below are placeholders for the same tables at 1B.
- Runs reused from tuning (`reused_from_tuning` in `outputs/metrics/final-1m/runs.parquet`): 3 of this method's 10 tuned runs (3 of 10 over all its units). A detector whose metric family a run never logged reads "not logged" (R78); detectors whose threshold assumes the 1B scale read "uninformative at 1M" (R79). Verdict counts of this method's detectors (all units): absent 55, detected 26, not logged 19; needs replay 0, needs eval actions 0.

## Hard scenarios

- Floors: floor 0 reach rate 100.0% (10 of 10 seeds); floors 1-8 reach rate 0.0% (0 of 640 final stochastic episodes). Source: `training-analysis-1m-pqn-rnn`, `hard/floor_reach.csv`, `hard/achievement_status.csv`, plot `plots/floor_reach.png`.
- Never unlocked (54 of 67; 0 unlocks in training episodes and in every eval of every seed): basic (12): place stone, make stone pickaxe, make stone sword, place furnace, collect coal, collect iron, collect diamond, make iron pickaxe, make iron sword, make arrow, make torch, place torch; intermediate (18): make diamond sword, make iron armour, make diamond armour, make diamond pickaxe, enter dungeon, defeat orc solider, defeat orc mage, eat snail, find bow, fire bow, open chest, drink potion, enter gnomish mines, defeat gnome warrior, defeat gnome archer, eat bat, collect sapphire, collect ruby; advanced (15): enter sewers, defeat lizard, defeat kobold, learn fireball, cast fireball, learn iceball, cast iceball, enchant sword, enchant armour, enter vault, defeat knight, defeat archer, enter troll mines, defeat troll, defeat deep thing; very advanced (9): enter fire realm, defeat pigman, defeat fire elemental, enter ice realm, defeat frost troll, defeat ice elemental, enter graveyard, damage necromancer, defeat necromancer. Source: `training-analysis-1m-pqn-rnn`, `hard/achievement_status.csv` (`status` = never), plot `plots/achievement_status_tuned.png`.
- Rare (final stochastic success rate < 5%, unlocked somewhere; 6): defeat zombie 3.75% (10/10 seeds ever), make wood sword 1.88% (10/10 seeds ever), make wood pickaxe 0.31% (9/10 seeds ever), defeat skeleton 0.31% (10/10 seeds ever), eat plant 0.16% (1/10 seeds ever), collect stone 0.00% (1/10 seeds ever). Same file.
- Common (7): collect sapling 83.0%, wake up 78.3%, collect wood 70.9%, place plant 64.8%, place table 33.9%, collect drink 27.2%, eat cow 12.0%. Same file.
- Death causes at the final checkpoint (checkpoint 244, 999,424 env steps, 640 episodes): zombie 298 (46.6%); starvation 215 (33.6%); arrow 113 (17.7%); dehydration 14 (2.2%). At checkpoint 0 (untrained): zombie 48.4%, starvation 37.5%, arrow 12.5%, dehydration 1.6%. Source: `training-analysis-1m-pqn-rnn`, `hard/death_causes.csv`, plot `plots/death_causes_over_training_tuned.png`.
- Stalls (>= 300 steps without an unlock): 105 stalls in 640 final eval episodes, 0.164 per episode (seed interval 0.136 to 0.194), median length 320 steps, 4.8% escaped by an unlock. All started on floor 0; started by wake up 43 (41.0%), place plant 22 (21.0%), collect drink 13 (12.4%), episode start 8 (7.6%). Source: `training-analysis-1m-pqn-rnn`, `hard/stall_stats.csv`, `hard/stall_starts.csv`, plot `plots/stall_lengths.png`.
- Stall endings: 5 by an unlock, 100 by death. Longest stall 448 steps (run `09abe70a68ba4fa8975dc3057aab5a0d`, seed 2, eval seed 10023, steps 148 to 596, ended by death). Source: `hard/stalls.csv`.

## How complex scenarios were solved

One paragraph per case of the tuned config, from `cases/<id>.json` (`describe_case`: stats, inventory, action counts in the window) and `cases.csv`; action counts are the steps not forced in the window (a chosen `make_*` or `place_*` is an attempt, not a success). Replay files are paths inside the artifacts of MLflow run `training-analysis-1m-pqn-rnn` (`masters/` stays on disk only). A stall case is replayed from stall start to the unlock; the other kinds from 150 steps before the event.

### Reading

Across this method's tuned stall cases and its stall table (plan question: what changed in behaviour just before the unlock); counts, not causes.

- Stall endings (`hard/stalls.csv`, final eval): 105 stalls, 5 ended by an unlock, 100 by death (95.2%).
- Before the unlock (8 stall cases; `last_actions` = the 50 actions before it): one action fills at least half of those 50 steps in 5 cases; most frequent actions do 28%, drink potion red 15%, place table 7%; movement keys are 18% of the last 50 steps against 20% of the whole stall window, `do` 28% against 15%; the most frequent action of the last 50 differs from the window's in 2 of 8 cases.
- State at the unlock: health <= 4 in 7 of 8 cases (median health 2.5 at the unlock, 9 at the stall start), food 0 in 6 of 8 (median food 0, 8.5 at the start), drink 0 in 6 of 8.
- Achievements that ended the stalls: wake up 4, eat plant 1, collect wood 1, eat cow 1, collect sapling 1.
- Deepest floor: the deep-floor cases are all overworld (floor 0) episodes, so the route question is moot at 1M.

### Stalls broken by an unlock

The longest escaped stalls of the final checkpoint and the longest one at earlier checkpoints (`cases.csv` kind `stall`).

**`1m-pqn-rnn-stall-1`** (run `pqn-rnn-1m-10cbfc16-s7`, `c1c0ffa15b934c299dde5925ff8ebff5`, checkpoint 244 (999,424 env steps), eval seed 10057, stochastic). Situation: a stall of 419 steps from step 353 on floor 0, begun after eat cow; at its start health 5, food 6, drink 7, energy 9, inventory sapling 4. Behaviour during the stall: most chosen actions do 82, up 63, down 62 over the 419 steps; 0.00% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 5. What broke it: eat plant at step 772 (then health 7, food 4, drink 5, energy 9, inventory sapling 7). Outcome: the episode lasted 1062 steps. Replay: `gifs/1m-pqn-rnn-stall-1.gif`, `overlays/1m-pqn-rnn-stall-1.mp4`, `masters/case-1m-pqn-rnn-stall-1.mkv`, `rollouts/case-1m-pqn-rnn-stall-1.npz`.

**`1m-pqn-rnn-stall-2`** (run `pqn-rnn-1m-10cbfc16-s6`, `8fc68412e96a425595a5f63d14dd0999`, checkpoint 244 (999,424 env steps), eval seed 10042, stochastic). Situation: a stall of 347 steps from step 43 on floor 0, begun after wake up; at its start health 9, food 8, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 108, down 48, up 47 over the 347 steps; 0.43% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 2. What broke it: collect wood at step 390 (then health 2, food 0, drink 9, energy 8, inventory wood 1). Outcome: the episode lasted 432 steps. Replay: `gifs/1m-pqn-rnn-stall-2.gif`, `overlays/1m-pqn-rnn-stall-2.mp4`, `masters/case-1m-pqn-rnn-stall-2.mkv`, `rollouts/case-1m-pqn-rnn-stall-2.npz`.

**`1m-pqn-rnn-stall-3`** (run `pqn-rnn-1m-10cbfc16-s9`, `e10e2e4707a641d6b684777dc23bb97a`, checkpoint 244 (999,424 env steps), eval seed 10031, stochastic). Situation: a stall of 337 steps from step 11 on floor 0, begun after place plant; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions drink potion pink 96, drink potion red 74, place plant 40 over the 337 steps; 0.39% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: wake up at step 348 (then health 3, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 382 steps. Replay: `gifs/1m-pqn-rnn-stall-3.gif`, `overlays/1m-pqn-rnn-stall-3.mp4`, `masters/case-1m-pqn-rnn-stall-3.mkv`, `rollouts/case-1m-pqn-rnn-stall-3.npz`.

**`1m-pqn-rnn-stall-4`** (run `pqn-rnn-1m-10cbfc16-s9`, `e10e2e4707a641d6b684777dc23bb97a`, checkpoint 244 (999,424 env steps), eval seed 10057, stochastic). Situation: a stall of 305 steps from step 11 on floor 0, begun after place plant; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions drink potion red 154, cast fireball 61, do 6 over the 305 steps; 1.17% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 4. What broke it: wake up at step 316 (then health 4, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 375 steps. Replay: `gifs/1m-pqn-rnn-stall-4.gif`, `overlays/1m-pqn-rnn-stall-4.mp4`, `masters/case-1m-pqn-rnn-stall-4.mkv`, `rollouts/case-1m-pqn-rnn-stall-4.npz`.

**`1m-pqn-rnn-stall-5`** (run `pqn-rnn-1m-10cbfc16-s2`, `09abe70a68ba4fa8975dc3057aab5a0d`, checkpoint 244 (999,424 env steps), eval seed 10002, stochastic). Situation: a stall of 304 steps from step 46 on floor 0, begun after place table; at its start health 9, food 8, drink 8, energy 9, inventory empty. Behaviour during the stall: most chosen actions do 102, down 59, up 39 over the 304 steps; 13.67% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: eat cow at step 350 (then health 1, food 6, drink 0, energy 3, inventory wood 4, sapling 10). Outcome: the episode lasted 354 steps. Replay: `gifs/1m-pqn-rnn-stall-5.gif`, `overlays/1m-pqn-rnn-stall-5.mp4`, `masters/case-1m-pqn-rnn-stall-5.mkv`, `rollouts/case-1m-pqn-rnn-stall-5.npz`.

**`1m-pqn-rnn-stall-6`** (run `pqn-rnn-1m-10cbfc16-s2`, `09abe70a68ba4fa8975dc3057aab5a0d`, checkpoint 0 (0 env steps), eval seed 10044, stochastic). Situation: a stall of 303 steps from step 43 on floor 0; at its start health 9, food 8, drink 8, energy 9, inventory wood 1. Behaviour during the stall: most chosen actions place stone 11, make arrow 11, make torch 11 over the 303 steps; 2.21% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 3. What broke it: collect sapling at step 346 (then health 3, food 0, drink 0, energy 9, inventory wood 1, sapling 1). Outcome: the episode lasted 391 steps. Replay: `gifs/1m-pqn-rnn-stall-6.gif`, `overlays/1m-pqn-rnn-stall-6.mp4`, `masters/case-1m-pqn-rnn-stall-6.mkv`, `rollouts/case-1m-pqn-rnn-stall-6.npz`.

**`1m-pqn-rnn-stall-7`** (run `pqn-rnn-1m-10cbfc16-s1`, `95e53cb0fea34b1d8f6f6a6693fef307`, checkpoint 12 (49,152 env steps), eval seed 10031, stochastic). Situation: a stall of 368 steps from step 0 on floor 0; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions place table 156, left 7, right 6 over the 368 steps; 2.65% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: wake up at step 368 (then health 1, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-pqn-rnn-stall-7.gif`, `overlays/1m-pqn-rnn-stall-7.mp4`, `masters/case-1m-pqn-rnn-stall-7.mkv`, `rollouts/case-1m-pqn-rnn-stall-7.npz`.

**`1m-pqn-rnn-stall-8`** (run `pqn-rnn-1m-10cbfc16-s5`, `d9708f4ad9af4d0cb5c585c1b1b40897`, checkpoint 148 (606,208 env steps), eval seed 10005, stochastic). Situation: a stall of 367 steps from step 0 on floor 0; at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour during the stall: most chosen actions make diamond sword 207, level up strength 32, make iron pickaxe 20 over the 367 steps; 0.78% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 1. What broke it: wake up at step 367 (then health 1, food 0, drink 0, energy 9, inventory empty). Outcome: the episode lasted 381 steps. Replay: `gifs/1m-pqn-rnn-stall-8.gif`, `overlays/1m-pqn-rnn-stall-8.mp4`, `masters/case-1m-pqn-rnn-stall-8.mkv`, `rollouts/case-1m-pqn-rnn-stall-8.npz`.

### Deepest episodes

The 3 episodes with the deepest floor over all checkpoints, ties by achievements unlocked (kind `deep_floor`); on floor 0 the case is the episode with the most achievements.

**`1m-pqn-rnn-deep_floor-1`** (run `pqn-rnn-1m-10cbfc16-s2`, `09abe70a68ba4fa8975dc3057aab5a0d`, checkpoint 244 (999,424 env steps), eval seed 10019, stochastic). Situation: an episode with 7 achievements unlocked on floor 0; window steps 96 to 246, at its start health 9, food 6, drink 5, energy 9, inventory wood 1. Behaviour: most chosen actions do 72, place table 30, down 16; 7.90% of overworld tiles first seen in the window; combat damage 9, close calls 0, lowest health 2. At step 246 (the last unlock) health 2, food 0, drink 3, energy 4, inventory wood 1. Outcome: the episode lasted 254 steps. Replay: `gifs/1m-pqn-rnn-deep_floor-1.gif`, `masters/case-1m-pqn-rnn-deep_floor-1.mkv`, `rollouts/case-1m-pqn-rnn-deep_floor-1.npz`.

**`1m-pqn-rnn-deep_floor-2`** (run `pqn-rnn-1m-10cbfc16-s2`, `09abe70a68ba4fa8975dc3057aab5a0d`, checkpoint 244 (999,424 env steps), eval seed 10023, stochastic). Situation: an episode with 7 achievements unlocked on floor 0; window steps 0 to 148, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions sleep 31, do 24, down 15; 2.73% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 9. At step 148 (the last unlock) health 9, food 7, drink 4, energy 9, inventory sapling 2. Outcome: the episode lasted 596 steps. Replay: `gifs/1m-pqn-rnn-deep_floor-2.gif`, `masters/case-1m-pqn-rnn-deep_floor-2.mkv`, `rollouts/case-1m-pqn-rnn-deep_floor-2.npz`.

**`1m-pqn-rnn-deep_floor-3`** (run `pqn-rnn-1m-10cbfc16-s2`, `09abe70a68ba4fa8975dc3057aab5a0d`, checkpoint 244 (999,424 env steps), eval seed 10024, stochastic). Situation: an episode with 7 achievements unlocked on floor 0; window steps 64 to 214, at its start health 9, food 7, drink 7, energy 9, inventory wood 1. Behaviour: most chosen actions place table 93, do 15, down 12; 4.30% of overworld tiles first seen in the window; combat damage 0, close calls 0, lowest health 9. At step 214 (the last unlock) health 9, food 8, drink 1, energy 4, inventory empty. Outcome: the episode lasted 281 steps. Replay: `gifs/1m-pqn-rnn-deep_floor-3.gif`, `masters/case-1m-pqn-rnn-deep_floor-3.mkv`, `rollouts/case-1m-pqn-rnn-deep_floor-3.npz`.

### Rare unlocks

One episode per rare achievement that unlocked it (kind `rare_unlock`).

**`1m-pqn-rnn-rare_unlock-1`** (run `pqn-rnn-1m-10cbfc16-s6`, `8fc68412e96a425595a5f63d14dd0999`, checkpoint 244 (999,424 env steps), eval seed 10025, stochastic). Situation: make wood pickaxe (rare at the final checkpoint) is unlocked at step 95; window steps 0 to 95, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions place table 23, make iron armour 22, do 14; combat damage 4, close calls 0, lowest health 5. At the unlock health 5, food 6, drink 5, energy 6, inventory empty. Outcome: the episode lasted 111 steps. Replay: `gifs/1m-pqn-rnn-rare_unlock-1.gif`, `masters/case-1m-pqn-rnn-rare_unlock-1.mkv`, `rollouts/case-1m-pqn-rnn-rare_unlock-1.npz`.

**`1m-pqn-rnn-rare_unlock-2`** (run `pqn-rnn-1m-10cbfc16-s0`, `c31ef16dec694eeb8e53e11b6dc28c84`, checkpoint 244 (999,424 env steps), eval seed 10053, stochastic). Situation: make wood sword (rare at the final checkpoint) is unlocked at step 87; window steps 0 to 87, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions up 26, do 24, sleep 8; combat damage 2, close calls 0, lowest health 7. At the unlock health 7, food 6, drink 6, energy 8, inventory wood 2. Outcome: the episode lasted 334 steps. Replay: `gifs/1m-pqn-rnn-rare_unlock-2.gif`, `masters/case-1m-pqn-rnn-rare_unlock-2.mkv`, `rollouts/case-1m-pqn-rnn-rare_unlock-2.npz`.

**`1m-pqn-rnn-rare_unlock-3`** (run `pqn-rnn-1m-10cbfc16-s8`, `a5f94ec675954425ba431f18216f03b6`, checkpoint 244 (999,424 env steps), eval seed 10047, stochastic). Situation: defeat zombie (rare at the final checkpoint) is unlocked at step 24; window steps 0 to 24, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions do 11, up 4, down 3; combat damage 2, close calls 0, lowest health 7. At the unlock health 7, food 9, drink 8, energy 9, inventory sapling 1. Outcome: the episode lasted 206 steps. Replay: `gifs/1m-pqn-rnn-rare_unlock-3.gif`, `masters/case-1m-pqn-rnn-rare_unlock-3.mkv`, `rollouts/case-1m-pqn-rnn-rare_unlock-3.npz`.

**`1m-pqn-rnn-rare_unlock-4`** (run `pqn-rnn-1m-10cbfc16-s7`, `c1c0ffa15b934c299dde5925ff8ebff5`, checkpoint 244 (999,424 env steps), eval seed 10057, stochastic). Situation: eat plant (rare at the final checkpoint) is unlocked at step 772; window steps 622 to 772, at its start health 9, food 4, drink 7, energy 8, inventory sapling 5. Behaviour: most chosen actions do 31, up 21, down 21; combat damage 0, close calls 0, lowest health 7. At the unlock health 7, food 4, drink 5, energy 9, inventory sapling 7. Outcome: the episode lasted 1062 steps. Replay: `gifs/1m-pqn-rnn-rare_unlock-4.gif`, `masters/case-1m-pqn-rnn-rare_unlock-4.mkv`, `rollouts/case-1m-pqn-rnn-rare_unlock-4.npz`.

**`1m-pqn-rnn-rare_unlock-5`** (run `pqn-rnn-1m-10cbfc16-s0`, `c31ef16dec694eeb8e53e11b6dc28c84`, checkpoint 244 (999,424 env steps), eval seed 10054, stochastic). Situation: defeat skeleton (rare at the final checkpoint) is unlocked at step 133; window steps 0 to 133, at its start health 9, food 9, drink 9, energy 9, inventory empty. Behaviour: most chosen actions up 29, do 21, down 20; combat damage 4, close calls 0, lowest health 7. At the unlock health 7, food 5, drink 4, energy 8, inventory wood 1. Outcome: the episode lasted 287 steps. Replay: `gifs/1m-pqn-rnn-rare_unlock-5.gif`, `masters/case-1m-pqn-rnn-rare_unlock-5.mkv`, `rollouts/case-1m-pqn-rnn-rare_unlock-5.npz`.

### Deaths

One representative episode for each of the 3 most frequent causes of death (kind `death`).

**`1m-pqn-rnn-death-1`** (run `pqn-rnn-1m-10cbfc16-s6`, `8fc68412e96a425595a5f63d14dd0999`, checkpoint 244 (999,424 env steps), eval seed 10035, stochastic). Situation: death by zombie at step 220; the window (steps 70 to 220) opens with health 9, food 7, drink 6, energy 9, inventory wood 2. Behaviour: most chosen actions left 33, right 24, make diamond sword 16; combat damage 9, close calls 0, lowest health 0. At death health 0, food 2, drink 0, energy 9, inventory empty. Outcome: the episode ended after 220 steps. Replay: `gifs/1m-pqn-rnn-death-1.gif`, `masters/case-1m-pqn-rnn-death-1.mkv`, `rollouts/case-1m-pqn-rnn-death-1.npz`.

**`1m-pqn-rnn-death-2`** (run `pqn-rnn-1m-10cbfc16-s1`, `95e53cb0fea34b1d8f6f6a6693fef307`, checkpoint 244 (999,424 env steps), eval seed 10009, stochastic). Situation: death by starvation at step 356; the window (steps 206 to 356) opens with health 9, food 2, drink 0, energy 6, inventory empty. Behaviour: most chosen actions cast fireball 60, make iron armour 33, drink potion green 25; combat damage 0, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 1, inventory empty. Outcome: the episode ended after 356 steps. Replay: `gifs/1m-pqn-rnn-death-2.gif`, `masters/case-1m-pqn-rnn-death-2.mkv`, `rollouts/case-1m-pqn-rnn-death-2.npz`.

**`1m-pqn-rnn-death-3`** (run `pqn-rnn-1m-10cbfc16-s5`, `d9708f4ad9af4d0cb5c585c1b1b40897`, checkpoint 244 (999,424 env steps), eval seed 10035, stochastic). Situation: death by arrow at step 268; the window (steps 118 to 268) opens with health 9, food 5, drink 4, energy 8, inventory wood 1, sapling 1. Behaviour: most chosen actions make torch 71, rest 33, level up strength 11; combat damage 7, close calls 0, lowest health 0. At death health 0, food 0, drink 0, energy 5, inventory wood 2, sapling 1. Outcome: the episode ended after 268 steps. Replay: `gifs/1m-pqn-rnn-death-3.gif`, `masters/case-1m-pqn-rnn-death-3.mkv`, `rollouts/case-1m-pqn-rnn-death-3.npz`.


## Obstacles

One table for Craftax-1M (`final-1m`, 10 seeds). Rows exist where a detector fired, could not read a run, or is uninformative or not applicable at 1M; the detectors listed as absent below found nothing in any run. The tuning-runs table (`tune-coarse`, `tune-fine`) is deferred at 1M: not analysed (tuning runs, R107), because analysing them would mean 494 more runs. The ablation units are in the Ablations section, and no config change after tuning was made, so no entry appends to [[Experiment decisions]]. Replay or plot paths are artifacts of MLflow run `training-analysis-1m-pqn-rnn`: a plot of the obstacle's family, or a case episode (GIF) of the first affected run, or the method's first death case.

Plateau at 1M uses a relative limit: the score gain from 50% of the budget to the end must stay below 0.1 x the run's best smoothed score (`PLATEAU_SHARE`; each row shows the limit it used), not the plan's absolute 1 point, which applies only at 1B (`ABSOLUTE_BUDGETS`, R79).

Caveat: skill forgetting fires on every seed of every method that runs it at 1M; threshold likely too sensitive, not counted as a method-specific obstacle.

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| Instability | 1M | 10 / 10 | 94.2k to 922k | 3 drops, deepest drawdown 1.09 (floor 0.44) (run c31ef16dec694eeb8e53e11b6dc28c84; 9 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-pqn-rnn-rare_unlock-2.gif` |
| Collapse | 1M | 4 / 10 | 127k to 999k | drawdown 1.09, final 1.59 of best 1.77 (floor 0.44) (run c31ef16dec694eeb8e53e11b6dc28c84; 3 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-pqn-rnn-rare_unlock-2.gif` |
| Plateau | 1M | 2 / 10 | 500k to 999k | score gain -0.74 from 500k to the end (< 0.15) (run 95e53cb0fea34b1d8f6f6a6693fef307; 1 more) | not analysed (tuning runs, R107) | open at 1M | `gifs/1m-pqn-rnn-stall-7.gif` |
| Entropy collapse | 1M | 0 / 7; not logged: 3 |  | not logged: ppo/entropy | - | - | - |
| Value failure | 1M | not logged: 10 |  | not logged: ppo/explained_variance | - | - | - |
| Reward hacking | 1M | 0 / 7; not logged: 3 |  | not logged: train/ach_reward_share | - | - | - |
| Exploration failure | 1M | 0 / 7; not logged: 3 |  | not logged: first_unlock/ | - | - | - |
| Skill forgetting | 1M | 10 / 10 | 8.19k to 995k | >= 20% to < 5%: collect_drink (45.1k to 53.2k), collect_sapling (12.3k to 20.5k), collect_wood (12.3k to 250k), eat_cow (610k to 651k), place_plant (12.3k to 20.5k), wake_up (12.3k to 172k) (run c31ef16dec694eeb8e53e11b6dc28c84; 9 more) | not analysed (tuning runs, R107) | open at 1M | `plots/achievement_status_tuned.png` |

- Absent in every run: Numerical failure, Survival failure.
- Detector verdicts over the tuned runs (100 run-detector pairs): absent 55, detected 26, not logged 19; needs replay 0; needs eval actions 0. Runs with a metric family not logged (R78, runs reused from tuning, or a base control that lacks it): 10.

### 1B

1B: pending (final-1b not run, R75). The 1B obstacles table (3 seeds) is added by rerunning `analysis.training` once the 1B finals exist.

## Comparison with the controls

The three questions for the three controls, one line each (tuned configs, final checkpoint, stochastic eval; floors reached, never or rare achievements, cause of death and stalls). "floor 0 only" means no final stochastic episode of 640 reaches floor 1 or deeper. Source: the same `hard/*.csv` files of each control's analysis run.

- This method, [[PQN-RNN training analysis]] (this note): floors - floor 0 only; achievements - 54 never, 6 rare, 7 common; killer and stalls - zombie 46.6% of deaths, 0.164 stalls per episode (median 320 steps, 4.8% escaped); final score 1.356.
- [[PPO training analysis]]: floors - floor 0 only; achievements - 48 never, 10 rare, 9 common; killer and stalls - zombie 61.1% of deaths, 0.095 stalls per episode (median 332 steps, 8.2% escaped); final score 2.173.
- [[PPO-RNN training analysis]]: floors - floor 0 only; achievements - 47 never, 11 rare, 9 common; killer and stalls - zombie 56.6% of deaths, 0.113 stalls per episode (median 334.5 steps, 11.1% escaped); final score 2.188.

## Ablations

No ablation of this method ran in `final-1m`.
