---
title: "evaluation"
type: module
domain: evaluation
parent: "[[wut_murl_craftax]]"
tags: [architecture, evaluation]
created: 2026-10-21
updated: 2026-10-24
related-code:
  - "[[src/wut_murl_craftax/evaluation/__init__.py]]"
  - "[[src/wut_murl_craftax/evaluation/events.py]]"
  - "[[src/wut_murl_craftax/evaluation/milestones.py]]"
  - "[[src/wut_murl_craftax/evaluation/episode_metrics.py]]"
  - "[[src/wut_murl_craftax/evaluation/reproduction.py]]"
  - "[[src/wut_murl_craftax/evaluation/stats.py]]"
---
# evaluation

## Responsibility

- Define what a Craftax step and episode are worth: game tables, per-step events, online episode metrics, the reproduction check ([[2026-10-20-craftax-evaluation-and-metrics]]).

## Public surface

- `step_events(game, s, action, s2) -> StepEvents`: pure, runs under `jit` and `vmap`; `Game`, `game_of`, `achievement_names`.
- `init_acc`, `episode_step`, `update_summary`, `score_pct`: online accumulation of episode metrics.
- `band`, `Reproduction`: the reproduction check.
- `iqm`, `mean_se`, `probability_of_improvement`, `seed_spread`: point estimates over the seeds of one stratum; `Interval` is `(estimate, lo, hi)`.
- `stratified_bootstrap_ci(scores_by_stratum, statistic, n=10_000, seed=0) -> Interval`: percentile 95% interval, seeds resampled inside each stratum; `statistic` takes the resampled arrays by stratum key (IQM of the pool, or a probability of improvement between two strata).
- `t_ci(scores) -> Interval`: t-based 95% interval (1B, three seeds).
- `holm(p_values)`, `mann_whitney_holm(comparisons, alpha=0.05) -> DataFrame` (`u`, `p`, `p_holm`, `reject`): one Holm family per mapping.
- `step_grid(budget)`, `curve_band(curves, grid, statistic="mean"|"iqm") -> DataFrame` (`env_steps`, `estimate`, `lo`, `hi`): learning curves on the common grid with a bootstrap band.

## Modules

One line each; signatures and details: [[Modules]].

- `milestones`: game tables (tech ladder, floors, creatures, actions, radar axes); names only, no Craftax import.
- `events`: `StepEvents` of one step from the states before and after it; deviations from the plan's definitions: [[2026-10-20-craftax-evaluation-and-metrics]] "Shared step events".
- `episode_metrics`: online score and achievement metrics of training episodes, read from step events.
- `reproduction`: does a control's final score match the published one (`band`, CLI)?
- `stats`: the statistics protocol (IQM, bootstrap and t intervals, probability of improvement, Holm-corrected Mann-Whitney, seed spread, curve bands); `select_pair` uses `iqm` and `probability_of_improvement`.

## Data flow

- In: states from the env wrappers of [[training]] and from replays of [[visualisation]].
- Out: `StepEvents` per step (training metrics, `eval/<mode>/<update>.npz`), per-update summaries, reproduction verdicts tagged on MLflow runs.

## Invariants

- Statistics take one score per run; `stratified_bootstrap_ci` never mixes strata, and a bootstrap seed fixes its draws. A mapping given to `mann_whitney_holm` is one Holm family. No test is run at 1B (three seeds cannot reach 0.05; [[2026-10-20-craftax-evaluation-and-metrics]] "Statistics protocol").
- `curve_band` is NaN where some seed has no data (before its first update or past its last logging gap), never extrapolated.
- Events are computed from the pre-reset next state, so a done step keeps its unlocks ([[Experiment decisions]]).
- Importing a table module never starts the JAX backends.

## Known issues

- None.

## Tests

- [[tests/evaluation/test_events.py]], [[tests/evaluation/test_milestones.py]], [[tests/evaluation/test_episode_metrics.py]], [[tests/evaluation/test_reproduction.py]], [[tests/evaluation/test_stats.py]]
