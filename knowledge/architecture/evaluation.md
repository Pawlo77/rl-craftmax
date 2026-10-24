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
  - "[[src/wut_murl_craftax/evaluation/stability.py]]"
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
- `holm(p_values)`, `mann_whitney_holm(comparisons, alpha=0.05) -> DataFrame` (`u`, `p`, `p_holm`, `reject`): one Holm family per mapping; logs a warning for a pair that cannot reach alpha (3 seeds a side).
- `step_grid(budget, n=200)`: 50 log-spaced points from 0.05% to 5% of the budget, then 150 evenly spaced up to the budget (our reading of the plan's "log-spaced for the first 5%"; points below a seed's first update are NaN in `curve_band`).
- `curve_band(curves, grid, statistic="mean"|"iqm") -> DataFrame` (`env_steps`, `estimate`, `lo`, `hi`): learning curves on the common grid with a bootstrap band.
- `smoothed(curve)`, `drawdowns(c)`, `drops(c) -> [(start, stop)]`, `collapsed(c)`, `recovery_times(c, steps)`, `dispersion(c)`: learning stability of one per-update score curve (`c` is the smoothed curve; indices are updates).
- `config_scores(tuning_table)`, `robust_share(scores)`, `sensitivity(tuning_table) -> DataFrame`: config score `r_c`, robust share and `S_h` per hyperparameter with its noise floor; the table is the `runs` rows of one method's tuning phase (`config_hash`, `final_score`, `config_json`).

## Modules

One line each; signatures and details: [[Modules]].

- `milestones`: game tables (tech ladder, floors, creatures, actions, radar axes); names only, no Craftax import.
- `events`: `StepEvents` of one step from the states before and after it; deviations from the plan's definitions: [[2026-10-20-craftax-evaluation-and-metrics]] "Shared step events".
- `episode_metrics`: online score and achievement metrics of training episodes, read from step events.
- `reproduction`: does a control's final score match the published one (`band`, CLI)?
- `stats`: the statistics protocol (IQM, bootstrap and t intervals, probability of improvement, Holm-corrected Mann-Whitney, seed spread, curve bands); `select_pair` uses `iqm` and `probability_of_improvement`.
- `stability`: drawdown, drop, collapse and recovery of a run's curve; config scores and hyperparameter sensitivity of a tuning grid ([[2026-10-20-craftax-evaluation-and-metrics]] "Learning stability").

## Data flow

- In: states from the env wrappers of [[training]] and from replays of [[visualisation]].
- Out: `StepEvents` per step (training metrics, `eval/<mode>/<update>.npz`), per-update summaries, reproduction verdicts tagged on MLflow runs.

## Invariants

- Statistics take one score per run; `stratified_bootstrap_ci` never mixes strata, and a bootstrap seed fixes its draws. A mapping given to `mann_whitney_holm` is one Holm family. No test is run at 1B (three seeds cannot reach 0.05; [[2026-10-20-craftax-evaluation-and-metrics]] "Statistics protocol").
- `smoothed` is a centred rolling median over 1% of the updates; a drop is a drawdown of at least 0.2 lasting at least 2% of the updates; recovery is `inf` when the drawdown never falls below 0.05. Drawdowns are NaN while the running max is below 1% score and never count.
- Stability functions take the smoothed curve in percent score. NaN scores (updates without a finished episode) are skipped by `smoothed`; `drawdowns` carries the last known score across NaN, so a NaN stretch neither hides a collapse nor splits a drop.
- Hyperparameters of `sensitivity` are the `config_json` keys constant within a `config_hash` and varying across them, so seeds drop out; a table with no varying key gives an empty result.
- `curve_band` skips non-finite scores per seed and is NaN where some seed has no data: before its first finite update, or later than one median logging gap after its last. Between those it interpolates linearly; nothing else is extrapolated.
- Events are computed from the pre-reset next state, so a done step keeps its unlocks ([[Experiment decisions]]).
- Importing a table module never starts the JAX backends.

## Known issues

- None.

## Tests

- [[tests/evaluation/test_events.py]], [[tests/evaluation/test_milestones.py]], [[tests/evaluation/test_episode_metrics.py]], [[tests/evaluation/test_reproduction.py]], [[tests/evaluation/test_stats.py]], [[tests/evaluation/test_stability.py]]
