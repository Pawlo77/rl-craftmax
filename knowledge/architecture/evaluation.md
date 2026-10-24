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
  - "[[src/wut_murl_craftax/evaluation/episode_records.py]]"
  - "[[src/wut_murl_craftax/evaluation/coverage.py]]"
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
- `episode_record(events, actions, trace=None, *, game) -> EpisodeRecord`: one eval episode from its stored `StepEvents` and actions; `EpisodeRecord.row()` gives its eval-episode table columns (identity columns excepted), with nested `Stall` and `Fight` lists. `stored_records(npz arrays, game)` records every seed of an eval file without replay; `trace_of(states, game) -> Trace` reads the replayed states the trace-only fields need.
- `init_seen(s0, game)`, `update_seen(seen, s2, game, valid=True)` (pure, `jit`, `vmap`; `valid` False skips a padded step), `coverage_summary(seen, positions, floors)`: exploration coverage per floor, new tiles per 100 steps, revisit rate and unique positions of one episode.
- `band`, `Reproduction`: the reproduction check.
- `iqm`, `mean_se`, `probability_of_improvement`, `seed_spread`: point estimates over the seeds of one stratum; `Interval` is `(estimate, lo, hi)`.
- `stratified_bootstrap_ci(scores_by_stratum, statistic, n=10_000, seed=0) -> Interval`: percentile 95% interval, seeds resampled inside each stratum; `statistic` takes the resampled arrays by stratum key (IQM of the pool, or a probability of improvement between two strata).
- `t_ci(scores) -> Interval`: t-based 95% interval (1B, three seeds).
- `holm(p_values)`, `mann_whitney_holm(comparisons, alpha=0.05) -> DataFrame` (`u`, `p`, `p_holm`, `reject`): one Holm family per mapping; logs a warning for a pair that cannot reach alpha (3 seeds a side).
- `step_grid(budget, n=200)`: 50 log-spaced points from 0.05% to 5% of the budget, then 150 evenly spaced up to the budget (our reading of the plan's "log-spaced for the first 5%"; points below a seed's first update are NaN in `curve_band`).
- `curve_band(curves, grid, statistic="mean"|"iqm") -> DataFrame` (`env_steps`, `estimate`, `lo`, `hi`): learning curves on the common grid with a bootstrap band.
- `smoothed(curve)`, `drawdowns(c, floor=1.0)`, `drops(c, floor=1.0) -> [(start, stop)]`, `collapsed(c, floor=1.0)`, `recovery_times(c, steps, floor=1.0)`, `dispersion(c)`: learning stability of one per-update score curve (`c` is the smoothed curve; indices are updates).
- `config_scores(tuning_table)`, `robust_share(scores)`, `sensitivity(tuning_table) -> DataFrame`: config score `r_c`, robust share and `S_h` per hyperparameter with its noise floor; the table is the `runs` rows of one method's tuning phase (`config_hash`, `final_score`, `config_json`).

## Modules

One line each; signatures and details: [[Modules]].

- `milestones`: game tables (tech ladder, floors, creatures, actions, radar axes); names only, no Craftax import.
- `events`: `StepEvents` of one step from the states before and after it; deviations from the plan's definitions: [[2026-10-20-craftax-evaluation-and-metrics]] "Shared step events".
- `episode_metrics`: online score and achievement metrics of training episodes, read from step events.
- `episode_records`: per-episode records of eval episodes (end, cause, stalls, fights, splits, combat, needs and survival timing, actions, resources, wasted crafts).
- `coverage`: tiles seen per floor (first-seen step per tile), new tiles and revisits of an episode.
- `reproduction`: does a control's final score match the published one (`band`, CLI)?
- `stats`: the statistics protocol (IQM, bootstrap and t intervals, probability of improvement, Holm-corrected Mann-Whitney, seed spread, curve bands); `select_pair` uses `iqm` and `probability_of_improvement`.
- `stability`: drawdown, drop, collapse and recovery of a run's curve; config scores and hyperparameter sensitivity of a tuning grid ([[2026-10-20-craftax-evaluation-and-metrics]] "Learning stability").

## Data flow

- In: states from the env wrappers of [[training]] and from replays of [[visualisation]].
- Out: `StepEvents` per step (training metrics, `eval/<mode>/<update>.npz`), per-update summaries, reproduction verdicts tagged on MLflow runs.
- Eval files -> `stored_records` (no replay); replayed states -> `trace_of` -> `episode_record(..., trace)` adds resources, wasted crafts and stall-start stats; `update_seen` scanned over the replayed states -> `coverage_summary` with the trace's positions and floors.

## Invariants

- Statistics take one score per run; `stratified_bootstrap_ci` never mixes strata, and a bootstrap seed fixes its draws. A mapping given to `mann_whitney_holm` is one Holm family. No test is run at 1B (three seeds cannot reach 0.05; [[2026-10-20-craftax-evaluation-and-metrics]] "Statistics protocol").
- `smoothed` is a centred rolling median over 1% of the updates; a drop is a drawdown of at least 0.2 lasting at least 2% of the updates; recovery is `inf` when the drawdown never falls below 0.05. Drawdowns are NaN while the running max is below `floor` (default 1% score, `MIN_MAX`) and never count; a 1M caller passes a lower floor (R92).
- Stability functions take the smoothed curve in percent score. NaN scores (updates without a finished episode) are skipped by `smoothed`; `drawdowns` carries the last known score across NaN, so a NaN stretch neither hides a collapse nor splits a drop.
- Hyperparameters of `sensitivity` are the `config_json` keys constant within a `config_hash` and varying across them, so seeds drop out; a table with no varying key gives an empty result.
- `curve_band` skips non-finite scores per seed and is NaN where some seed has no data: before its first finite update, or later than one median logging gap after its last. Between those it interpolates linearly; nothing else is extrapolated.
- Records count steps as frames: an unlock in action `t` has unlock step `t + 1`; stalls span previous unlock (or 0) to the next unlock (or the end), fights first to last fight frame. Without a trace, `gathered`, `used`, `wasted_crafts` and stall-start stats are None (NaN in `row`), never zeros.
- `seen` holds the `timestep` a tile was first in view and lit (`light_map > 0.05`; Classic has no light map), -1 never; its shape is the game's map (48 x 48 x 9 floors Craftax, Classic's own `map` shape). Coverage is NaN on a floor never visited; new tiles per 100 steps is the time-weighted mean of the 100-step windows; standing still counts as a revisit.
- Wasted crafts narrow the plan's "crafting or placing actions": a `MAKE_*` step that consumed resources and raised no owned level (tool, armour, arrows, torches), plus a `PLACE_TABLE` with a table already on the 8 tiles around the player. Other placements (stone, furnace, plant, torch) never count. Craftax refuses a worse pickaxe or sword, so that case arises only on Classic.
- Stall action histograms skip forced steps, as the episode histogram does. Survival timing (`survival/count|gap|level/<stat>`) pairs food with eats, drink with drinks and energy with sleep starts; gaps come from the stored events, levels (the stat in the state before) need the trace.
- Record rules shared with training (cause window, stall gap, close calls, action shares) follow `episode_metrics`; a test checks the record against `episode_step` on one stream.
- Events are computed from the pre-reset next state, so a done step keeps its unlocks ([[Experiment decisions]]).
- Importing a table module never starts the JAX backends.

## Known issues

- None.

## Tests

- [[tests/evaluation/test_events.py]], [[tests/evaluation/test_milestones.py]], [[tests/evaluation/test_episode_metrics.py]], [[tests/evaluation/test_episode_records.py]], [[tests/evaluation/test_coverage.py]], [[tests/evaluation/test_reproduction.py]], [[tests/evaluation/test_stats.py]], [[tests/evaluation/test_stability.py]]
