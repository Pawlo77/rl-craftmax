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
  - "[[src/wut_murl_craftax/evaluation/rollout_metrics.py]]"
  - "[[src/wut_murl_craftax/evaluation/reproduction.py]]"
  - "[[src/wut_murl_craftax/evaluation/stats.py]]"
  - "[[src/wut_murl_craftax/evaluation/stability.py]]"
  - "[[src/wut_murl_craftax/evaluation/hardware.py]]"
  - "[[src/wut_murl_craftax/evaluation/export.py]]"
---
# evaluation

## Responsibility

- Define what a Craftax step and episode are worth: game tables, per-step events, online episode metrics, the offline replay of eval files, the reproduction check ([[2026-10-20-craftax-evaluation-and-metrics]]).

## Public surface

- `step_events(game, s, action, s2) -> StepEvents`: pure, runs under `jit` and `vmap`; `Game`, `game_of`, `achievement_names`.
- `init_acc`, `episode_step`, `update_summary`, `score_pct`: online accumulation of episode metrics.
- `episode_record(events, actions, trace=None, *, game) -> EpisodeRecord`: one eval episode from its stored `StepEvents` and actions; `EpisodeRecord.row()` gives its eval-episode table columns (identity columns excepted), with nested `Stall` and `Fight` lists. `stored_records(npz arrays, game)` records every seed of an eval file without replay; `trace_of(states, game, xp=np) -> Trace` reads the replayed states the trace-only fields need (`xp=jnp` under `jit`).
- `init_seen(s0, game)`, `update_seen(seen, s2, game, valid=True)` (pure, `jit`, `vmap`; `valid` False skips a padded step), `coverage_summary(seen, positions, floors)`: exploration coverage per floor, new tiles per 100 steps, revisit rate and unique positions of one episode.
- `Replayer(run)`, `replay_file(rp, arrays) -> FileReplay`: every seed of an eval file replayed together and checked against the file (`ReplayMismatchError`). `file_tables(rp, arrays, update, mode, Identity) -> Tables` gives the eval-episode rows in `episode_columns(game)` order (`EPISODE_COLUMNS`, the contracts.md section 1 order with families expanded; the per-run rows lack `config_ref`), `skill_ends` (`analysis.obstacles.SKILL_ENDS_COLUMNS`, Dyna-Q adds `pair_count` from `pair_counts`) and `eval_inventory` (`obstacles.INVENTORY_COLUMNS`; `at_cap` over `HOARD_CAPS`) of contracts.md sections 1 and 9. `process_file` writes them to `out_paths(out, run_id, update, mode)` and logs them; `phase_runs(tracking_uri, phases)`, `plan_jobs(runs, updates, modes)`, `pending` and the CLI (`python -m wut_murl_craftax.evaluation.rollout_metrics --phase P --tracking-uri URI [--updates all|final|u,...] [--mode both] [--no-log]`) run a phase.
- `band`, `Reproduction`: the reproduction check.
- `iqm`, `mean_se`, `probability_of_improvement`, `seed_spread`: point estimates over the seeds of one stratum; `Interval` is `(estimate, lo, hi)`.
- `stratified_bootstrap_ci(scores_by_stratum, statistic, n=10_000, seed=0) -> Interval`: percentile 95% interval, seeds resampled inside each stratum; `statistic` takes the resampled arrays by stratum key (IQM of the pool, or a probability of improvement between two strata).
- `t_ci(scores) -> Interval`: t-based 95% interval (1B, three seeds).
- `holm(p_values)`, `mann_whitney_holm(comparisons, alpha=0.05) -> DataFrame` (`u`, `p`, `p_holm`, `reject`): one Holm family per mapping; logs a warning for a pair that cannot reach alpha (3 seeds a side).
- `step_grid(budget, n=200)`: 50 log-spaced points from 0.05% to 5% of the budget, then 150 evenly spaced up to the budget (our reading of the plan's "log-spaced for the first 5%"; points below a seed's first update are NaN in `curve_band`).
- `curve_band(curves, grid, statistic="mean"|"iqm") -> DataFrame` (`env_steps`, `estimate`, `lo`, `hi`): learning curves on the common grid with a bootstrap band.
- `smoothed(curve)`, `drawdowns(c, floor=1.0)`, `drops(c, floor=1.0) -> [(start, stop)]`, `collapsed(c, floor=1.0)`, `recovery_times(c, steps, floor=1.0)`, `dispersion(c)`: learning stability of one per-update score curve (`c` is the smoothed curve; indices are updates).
- `config_scores(tuning_table)`, `robust_share(scores)`, `sensitivity(tuning_table) -> DataFrame`: config score `r_c`, robust share and `S_h` per hyperparameter with its noise floor; the table is the `runs` rows of one method's tuning phase (`config_hash`, `final_score`, `config_json`).
- `hardware_summary(store, run_id, *, budget=None) -> dict`: hardware metrics of one run read from the SQLite store (path, `sqlite:///` URI or connection), keys `profile`, `device_kind`, `actor_device`, `learner_device`, `reused_from_tuning`, `wall_clock_s`, `steady_sps`, `overall_sps`, `compile_s`, `eval_s`, `cpu_util_mean|p95`, `gpu_util_mean|p95`, `peak_rss_mb`, `peak_host_mem_mb`, `peak_device_mem_mb`, `energy_j`, `steps_per_joule`; NaN is "not measured". `profile_of(learner_device)`, `reused_from_tuning(phases)`, `show_metric(value)` ("n.m." for NaN), `system_keys(store, run_id)`, `HARDWARE_KEYS` (stored names per profile and metric), `NOT_MEASURED`. CLI: `python -m wut_murl_craftax.evaluation.hardware --run <id> [--list-keys] [--tracking-uri ...]`.

- `export_phase(phase, out_dir, *, tracking_uri="sqlite:///mlruns.db", replay_root=outputs/metrics, phases_dir=configs/phases, runs=None) -> dict[str, Path]`: the phase's `runs`, `curves` and `eval_episodes` (contracts.md section 1) as parquet plus a CSV copy each (list columns as JSON strings; the curves copy gzipped as `curves.csv.gz`, R104), and the replay's `skill_ends` and `eval_inventory` tables when it wrote any; built in `<out_dir>.staging` and swapped in as a whole. `config_refs(phase_file)` maps (registry, `config_hash`) to `tuned` or the ablation name (`ABLATIONS`); `log_export(folder, phase, tracking_uri, *, csv=False)` logs the parquet files (all copies with `csv`) to run `export-<phase>` of experiment `ANALYSIS_EXPERIMENT` (`analysis`). CLI, run from the main checkout (the default paths are relative to the working directory): `python -m wut_murl_craftax.evaluation.export --phase P --tracking-uri URI[?timeout=300] [--out outputs/metrics/P] [--replay-root outputs/metrics] [--no-log] [--log-csv] [--run ID ... with --no-log and --out]`. `rollout_metrics.arrow_table(df)` gives the fixed Arrow types of the per-run tables; `rollout_metrics.phase_runs(uri, phases)` gives each `RunRef` with its status, experiment and tags; `hardware.open_store(store)` is the one read-only connector (300 s busy timeout, `?timeout=` overrides).

## Modules

One line each; signatures and details: [[Modules]].

- `milestones`: game tables (tech ladder, floors, creatures, actions, radar axes); names only, no Craftax import.
- `events`: `StepEvents` of one step from the states before and after it; deviations from the plan's definitions: [[2026-10-20-craftax-evaluation-and-metrics]] "Shared step events".
- `episode_metrics`: online score and achievement metrics of training episodes, read from step events.
- `episode_records`: per-episode records of eval episodes (end, cause, stalls, fights, splits, combat, needs and survival timing, actions, resources, wasted crafts).
- `coverage`: tiles seen per floor (first-seen step per tile), new tiles and revisits of an episode.
- `rollout_metrics`: offline replay of the stored eval files (EV6): records with the trace, coverage, the training-analysis frames `skill_ends` and `eval_inventory`, their files, MLflow logging and the phase CLI.
- `reproduction`: does a control's final score match the published one (`band`, CLI)?
- `stats`: the statistics protocol (IQM, bootstrap and t intervals, probability of improvement, Holm-corrected Mann-Whitney, seed spread, curve bands); `select_pair` uses `iqm` and `probability_of_improvement`.
- `stability`: drawdown, drop, collapse and recovery of a run's curve; config scores and hyperparameter sensitivity of a tuning grid ([[2026-10-20-craftax-evaluation-and-metrics]] "Learning stability").
- `export`: a phase's tidy tables `runs`, `curves`, `eval_episodes` (EV10) from the store and the eval files, merged with the replay tables, and their MLflow run.
- `hardware`: wall-clock, SPS, compile time, utilisation, memory and energy of a run from its stored series ([[2026-10-20-craftax-evaluation-and-metrics]] "Hardware usage").

## Data flow

- In: states from the env wrappers of [[training]] and from replays of [[visualisation]].
- Out: `StepEvents` per step (training metrics, `eval/<mode>/<update>.npz`), per-update summaries, reproduction verdicts tagged on MLflow runs.
- Eval files -> `stored_records` (no replay); replayed states -> `trace_of` -> `episode_record(..., trace)` adds resources, wasted crafts and stall-start stats; `update_seen` scanned over the replayed states -> `coverage_summary` with the trace's positions and floors.
- `rollout_metrics`: `mlruns/<exp>/<run_id>/artifacts/eval/<mode>/<update>.npz` (+ `eval/dyna_model_graph.parquet`) -> replay in jitted 256-step scans of `EvalRun.env_step` ([[visualisation]] `eval_rollouts`), 64 seeds vmapped and split over the CPU devices -> `outputs/metrics/<run_id>/{eval,skill_ends,inventory}_<update>_<mode>.parquet` -> the run's `eval_tables/` artifacts and `eval/<mode>/<column>` episode means of the replay-only columns at step = the checkpoint's env steps. EV10 concatenates the `eval_*` files of a phase into `eval_episodes` and inserts `config_ref`.
- `export`: tag `phases` -> the phase's runs (read-only store, one short connection per run) -> `runs` (tags, `config.json`, last points of the stored series, `hardware_summary`), `curves` (every non-`system/*` series, streamed run by run), `eval_episodes` (per eval file the replay table if `outputs/metrics/<run_id>/eval_<update>_<mode>.parquet` exists, else `stored_records`) -> `outputs/metrics/<phase>.staging/` -> swapped into `outputs/metrics/<phase>/` -> parquet artifacts of run `export-<phase>` in experiment `analysis` -> `analysis.load`.

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
- A replay must reproduce the stored events of every seed exactly up to its `length`, and end a skill (`skill_step` on the replayed observations, as `SMDPPolicy.step`) exactly before each stored `steps_in_skill == 0`; otherwise `ReplayMismatchError` names run, update, mode and seed and the CLI skips that file (exit status 1 at the end). Past its length a seed keeps stepping in the batch, but `update_seen` gets `valid` False and its rows are cut.
- A skill runs from the frame of its decision to the next decision (or the episode end, reason `episode_end`); `key` is `smdp.top_key` of the decision observation; the other reasons are the `skill_step` reason of the step before the next decision. `capped_steps` counts the states before each action with a material at its `progress_key.CAPS` cap, only for materials whose cap is above 1 (`HOARD_CAPS`: wood, stone, coal, iron, diamond; R98: one sapling, gem, arrow, torch, book or potion is no hoarding).
- The per-run eval tables leave out `config_ref`: it depends on the exported phase. `stalls` and `fights` have fixed Arrow types, so files with no stall concatenate.
- Resumable and idempotent: the eval table is written last (atomic writes); a `.logging` marker before the MLflow logging becomes `.logged` after it, and a rerun after a cut logs only the summaries the store lacks at that step. The CLI skips what exists and loads a run only for pending files. Final checkpoints of every run come first.
- A file with `extras/skill` whose run policy is not an `SMDPPolicy` raises `ReplayMismatchError` instead of silently dropping `skill_ends`. The eval table must hold exactly the `episode_columns` set, or `file_tables` raises.
- Hardware: one run per call, read-only SQL by `run_uuid` (never a scan of `metrics`); a series the store lacks is NaN, never 0 (the laptop store has no power, GPU or eval-time series and no `profile` tag, so `gpu_util_*`, `peak_device_mem_mb`, `energy_j`, `steps_per_joule` and `eval_s` are all "n.m." there; the runner logs `peak_device_mb` whenever the device reports a peak; `eval_s` is a placeholder no runner code logs yet). `peak_device_mem_mb` is the largest of the runner's `peak_device_mb` and the sampled GPU memory series. `profile` comes from the `learner_device` tag (`MetalDevice` laptop, `cuda`/`CudaDevice` cuda, else unknown) and runs are compared inside one profile only.
- `train/wall_s` is cumulative across resume segments (the runner restores it), so the wall-clock is its last value; a re-logged update keeps its latest value, and a `wall_s` that runs backwards raises. MLflow's step of those points is the env steps. Steady SPS runs from the point at 10 times the first point's steps (update 10 at 1M, a window of updates at 1B) to the last. Compile time is the stored `compile_s`, else the first point's `wall_s` minus its steps at the median per-step time (they agree on the final-1m runs checked).
- Energy is the trapezoid integral over the sample timestamps (the sysmetrics step counts samples, not time) of every `system/gpu_*_power_usage_watts` series; an interval above 10 times the median one is a resume gap and is skipped, so a resumed run's energy is a lower bound. It covers logged GPU power only (no CPU or system power).
- Context only (R97): every laptop run trained with 2 queue workers, fresh `final-1m` runs included, so laptop wall-clock, SPS and compile time, and the machine-wide `cpu_util_*` and `peak_host_mem_mb`, are context, never a clean comparison. `peak_rss_mb` is the peak of the whole worker process (several runs of a worker share it). Reports say "two concurrent workers" beside them.
- `reused_from_tuning` is True for a run whose `phases` tag lists `tune-coarse` or `tune-fine` (38 of the 140 `final-1m` runs, including the 26 without online families). It only separates reused from freshly trained runs; it says nothing about which timings are clean. Reused runs were in fact faster than fresh ones of the same method.
- Export: membership by tag `phases` only (R77), so a run reused by a later phase is exported with each phase that lists it. `config_ref` is `tuned` for the first `configs` item of a final or combo phase entry (it must be named `tuned`, `combined` or `base`, never an override) and the `ABLATIONS` name of each other item's `set`; a run hash the phase file does not produce, or a missing `phases`, `method`, `budget`, `seed` or `config_hash` tag, raises before any work; tuning phases use the hash, and only they keep tag `best`. `final_score` and the eval scores are stored metrics at their largest step, never recomputed. A curve point re-logged at the same step keeps its latest value. Replay tables win over stored records, which leave the replay-only columns (`coverage/*`, `new_tiles_per_100`, `revisit_rate`, `unique_positions`) null (any other missing column raises); `unique_positions` stays int64 in the parquet schema; a replay table whose identity disagrees with the run's tags raises. Atomic per phase: every file is written into `<out_dir>.staging`, which replaces `out_dir` in one directory move (the old export is moved to `<out_dir>.previous` first and restored by the next run if a crash fell between the two moves); a failure removes the staging folder and leaves the previous export loadable. A rerun of `log_export` keeps the run RUNNING while it logs the new files and deletes the artifacts the new export lacks only after that, raising if a delete fails. The store is read in one short connection per run with a 300 s busy timeout (rollback-journal store, R103), never in a transaction held across runs.
- Importing a table module never starts the JAX backends; `rollout_metrics` imports the replay modules inside functions, so the CLI can split the CPU (`--host-devices`) first.

## Known issues

- Hardware (EV9 Step 2, open with 1B, R75): the cuda key patterns of `HARDWARE_KEYS` come from mlflow's source, not from a cluster store; run `--list-keys` on a cluster smoke run and confirm them.
- On a multi-GPU node the pooled `gpu_util_*` and the summed `energy_j` cover every GPU sysmetrics sees, not only the learner's; check at Step 2.

## Tests

- [[tests/evaluation/test_events.py]], [[tests/evaluation/test_milestones.py]], [[tests/evaluation/test_episode_metrics.py]], [[tests/evaluation/test_episode_records.py]], [[tests/evaluation/test_coverage.py]], [[tests/evaluation/test_reproduction.py]], [[tests/evaluation/test_stats.py]], [[tests/evaluation/test_stability.py]], [[tests/evaluation/test_hardware.py]], [[tests/evaluation/test_rollout_metrics.py]], [[tests/evaluation/test_export.py]]
