---
title: "evaluation"
type: module
domain: evaluation
parent: "[[wut_murl_craftax]]"
tags: [architecture, evaluation]
created: 2026-10-21
updated: 2026-10-21
related-code:
  - "[[src/wut_murl_craftax/evaluation/__init__.py]]"
  - "[[src/wut_murl_craftax/evaluation/events.py]]"
  - "[[src/wut_murl_craftax/evaluation/milestones.py]]"
  - "[[src/wut_murl_craftax/evaluation/episode_metrics.py]]"
  - "[[src/wut_murl_craftax/evaluation/reproduction.py]]"
---
# evaluation

## Responsibility

- Define what a Craftax step and episode are worth: game tables, per-step events, online episode metrics, the reproduction check ([[2026-10-20-craftax-evaluation-and-metrics]]).

## Public surface

- `step_events(game, s, action, s2) -> StepEvents`: pure, runs under `jit` and `vmap`; `Game`, `game_of`, `achievement_names`.
- `init_acc`, `episode_step`, `update_summary`, `score_pct`: online accumulation of episode metrics.
- `band`, `Reproduction`: the reproduction check.

## Modules

One line each; signatures and details: [[Modules]].

- `milestones`: game tables (tech ladder, floors, creatures, actions, radar axes); names only, no Craftax import.
- `events`: `StepEvents` of one step from the states before and after it; deviations from the plan's definitions: [[2026-10-20-craftax-evaluation-and-metrics]] "Shared step events".
- `episode_metrics`: online score and achievement metrics of training episodes, read from step events.
- `reproduction`: does a control's final score match the published one (`band`, CLI)?

## Data flow

- In: states from the env wrappers of [[training]] and from replays of [[visualisation]].
- Out: `StepEvents` per step (training metrics, `eval/<mode>/<update>.npz`), per-update summaries, reproduction verdicts tagged on MLflow runs.

## Invariants

- Events are computed from the pre-reset next state, so a done step keeps its unlocks ([[Experiment decisions]]).
- Importing a table module never starts the JAX backends.

## Known issues

- None.

## Tests

- [[tests/evaluation/test_events.py]], [[tests/evaluation/test_milestones.py]], [[tests/evaluation/test_episode_metrics.py]], [[tests/evaluation/test_reproduction.py]]
