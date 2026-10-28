---
title: "Video cases 1M"
type: concept
domain: analysis
parent: "[[2026-10-20-craftax-project]]"
tags: [concept, results, analysis, video, craftax-1m]
created: 2026-10-28
updated: 2026-10-28
related-code:
  - "[[src/wut_murl_craftax/analysis/cases.py]]"
  - "[[src/wut_murl_craftax/analysis/training.py]]"
---
# Video cases 1M

The 36 case ids of `outputs/analysis/1m/cases.csv` flagged `video = yes` for [[2026-10-20-craftax-visualisation-production]] ("Stuck, then solved", deepest floor, deaths); every other row of the 291 reads `no`. `outputs/` is not in git, and `write_cases` or `analysis.training --force` resets the column, so this list is the tracked source: set `video` to `yes` for exactly these ids in the budget table and in the per-method copies `outputs/analysis/1m/<method>/cases.csv`, `no` elsewhere. Case ids are deterministic. Module: [[analysis]].

## Selection rule

- Only the `tuned` unit of each method; ablation units are `no`.
- 2 `stall` cases: latest checkpoint first, then the longest stall (the final checkpoint gives 2 for every method but dyna-q, whose second is its checkpoint 148 case).
- 1 `deep_floor` case: the first one (the floor 1 episode where one exists, else the overworld episode with the most achievements).
- 1 `death` case: the first one (the most frequent cause, zombie for all methods).
- Every flagged row has its rollout, master, GIF and (stalls) overlay on disk.

## Ids

| Method | Stall cases | Deep-floor case | Death case |
|---|---|---|---|
| ppo | `1m-ppo-stall-1` (468 steps, checkpoint 244)<br>`1m-ppo-stall-2` (351 steps, checkpoint 244) | `1m-ppo-deep_floor-1` (floor 0, checkpoint 244) | `1m-ppo-death-1` (zombie, checkpoint 244) |
| ppo-rnn | `1m-ppo-rnn-stall-1` (449 steps, checkpoint 244)<br>`1m-ppo-rnn-stall-2` (372 steps, checkpoint 244) | `1m-ppo-rnn-deep_floor-1` (floor 0, checkpoint 244) | `1m-ppo-rnn-death-1` (zombie, checkpoint 244) |
| pqn-rnn | `1m-pqn-rnn-stall-1` (419 steps, checkpoint 244)<br>`1m-pqn-rnn-stall-2` (347 steps, checkpoint 244) | `1m-pqn-rnn-deep_floor-1` (floor 0, checkpoint 244) | `1m-pqn-rnn-death-1` (zombie, checkpoint 244) |
| count-bonus-ppo-rnn | `1m-count-bonus-ppo-rnn-stall-1` (334 steps, checkpoint 244)<br>`1m-count-bonus-ppo-rnn-stall-2` (331 steps, checkpoint 244) | `1m-count-bonus-ppo-rnn-deep_floor-1` (floor 1, checkpoint 90) | `1m-count-bonus-ppo-rnn-death-1` (zombie, checkpoint 244) |
| count-bonus-pqn-rnn | `1m-count-bonus-pqn-rnn-stall-1` (352 steps, checkpoint 244)<br>`1m-count-bonus-pqn-rnn-stall-2` (335 steps, checkpoint 244) | `1m-count-bonus-pqn-rnn-deep_floor-1` (floor 0, checkpoint 244) | `1m-count-bonus-pqn-rnn-death-1` (zombie, checkpoint 244) |
| sil | `1m-sil-stall-1` (377 steps, checkpoint 244)<br>`1m-sil-stall-2` (342 steps, checkpoint 244) | `1m-sil-deep_floor-1` (floor 0, checkpoint 244) | `1m-sil-death-1` (zombie, checkpoint 244) |
| smdp-skills | `1m-smdp-skills-stall-1` (373 steps, checkpoint 244)<br>`1m-smdp-skills-stall-2` (331 steps, checkpoint 244) | `1m-smdp-skills-deep_floor-1` (floor 0, checkpoint 244) | `1m-smdp-skills-death-1` (zombie, checkpoint 244) |
| dyna-q | `1m-dyna-q-stall-1` (358 steps, checkpoint 244)<br>`1m-dyna-q-stall-4` (343 steps, checkpoint 148) | `1m-dyna-q-deep_floor-1` (floor 0, checkpoint 244) | `1m-dyna-q-death-1` (zombie, checkpoint 244) |
| combo | `1m-combo-stall-1` (354 steps, checkpoint 244)<br>`1m-combo-stall-2` (347 steps, checkpoint 244) | `1m-combo-deep_floor-1` (floor 0, checkpoint 244) | `1m-combo-death-1` (zombie, checkpoint 244) |
