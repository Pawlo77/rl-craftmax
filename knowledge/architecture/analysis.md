---
title: "analysis"
type: module
domain: analysis
parent: "[[wut_murl_craftax]]"
tags: [architecture, analysis]
created: 2026-10-24
updated: 2026-10-24
related-code:
  - "[[src/wut_murl_craftax/analysis/__init__.py]]"
  - "[[src/wut_murl_craftax/analysis/cases.py]]"
  - "[[src/wut_murl_craftax/analysis/compare.py]]"
  - "[[src/wut_murl_craftax/analysis/hard.py]]"
  - "[[src/wut_murl_craftax/analysis/literature.py]]"
  - "[[src/wut_murl_craftax/analysis/load.py]]"
  - "[[src/wut_murl_craftax/analysis/obstacles.py]]"
---
# analysis

## Responsibility

- Phase 4 analysis of finished runs: load the exported tables, per-method training analysis, obstacles, cases ([[2026-10-20-craftax-training-analysis]]) and the method comparison ([[2026-10-20-craftax-comparison-and-results]]).
- Filled by the training analysis and comparison tasks; so far the loaders, the literature table, the obstacle detectors, the hard-scenario summaries and the comparison tables.
- Training-analysis cases (TA3): the case finder, its replay export and the case facts.

## Public surface

- `load_phase(phase_dir)`, `load_phases(phases, root, skip_missing=)`: the export tables as `{"runs", "curves", "eval_episodes"}` frames; `load_phases` keeps a run of two phases once, from the first phase listing it, and warns when the dropped copy differs; pass `final-*` before tuning phases.
- `select(frames, method=, config_ref=, budget=, phase=, pool=False)`: the matching runs and their rows in every frame; `phase` matches the `phases` tag (`in_phase(runs, phase)` is its mask). It raises when the result mixes `config_ref` values of a method and no `config_ref` or `pool=True` is given.
- `units(frames)`: each (`method`, `config_ref`) with its own frames.
- `BASE_CONTROL` (non-combo method -> control), `CONTROLS`, `base_control(row)`, `pair_of(row)`: the control a run is compared with; a combo run's comes from its own pair.
- `final_checkpoint(episodes, run=None)`: the last checkpoint's eval episodes per run.
- `load_literature(path=)`: the published rows of blocks A, A2 and D of [[Existing Implementations]] from `configs/literature/existing_implementations.csv` (columns `block, implementation, note, env, observation, budget, protocol, score_pct, spread, spread_kind, seeds, score_text`; `note` is the linked note's file stem; `protocol` is the block's (A: training-episode return, A2: separate evaluation of the trained policy), `n.r.` where the row's source note says the protocol is unstated (PPO-GTrXL, PQN-RNN, SimBa, EASimulus), and per row in block D from the notes). `LITERATURE_CSV` and `LITERATURE_COLUMNS` are the path and the column list.
- `comparable_rows(score_kind, budget)`: the like-for-like block (`COMPARABLE_BLOCKS`): `("training", "1b")` is A, `("training", "1m")` is D, `("stochastic", "1b")` is A2; any other pair raises `ValueError`.
- `MissingPhaseError` (a `FileNotFoundError` naming the phases), `SchemaError`, `FINAL_PHASES`, `METRICS_ROOT`, `UNIT_COLUMNS`.
- `detect_all(frames, method, budget, config_ref="tuned")`: one `Obstacle(name, run, start, end, evidence, status)` per run and detector of `DETECTORS[method]`; `status` is `detected`, `absent`, `not logged`, `needs replay (evaluation Task 6)`, `needs eval actions`, `not applicable` or `uninformative at <budget>`.
- `DETECTORS` (registry method -> `Detector` list), built from `shared_detectors()`, `count_bonus_detectors()`, `sil_detectors()`, `skill_detectors()`, `dyna_detectors()`; a `Detector` declares `REQUIRED_METRICS`, `BASE_METRICS`, optional `inputs`, informative `budgets` and owner `methods`.
- `ALIASES`: plan metric name -> stored names in order (contracts section 5; the comparison figures import it).
- `table_rows(obstacles, budget)`, `obstacles_table(rows)`: the obstacles table with exactly `TABLE_COLUMNS` of the plan template.
- `drawdown_floor(budget, c)`: the R92 drawdown floor of a smoothed score curve (1 point at 1B, else `max(0.1, 0.25 x` best)); the detectors and comparison T3 share it.
- `repeated_cycle(actions)` -> `Cycle(period, start, length)`: the shortest action period repeated over >= 50 steps; `eval_actions(run_dir, run_id, update, mode)` reads an eval file into the `eval_actions` input.
- `floor_reach(episodes, mode="stochastic", n=10000, floors=9)`: per unit and floor, the share of final eval episodes with `max_floor >= f` (`rate`, the mean over seeds of each seed's share, fractions) with the bootstrap interval `lo`, `hi` over seeds, `reached_seeds` and `status` (`never`, `rare` below `RARE_RATE` = 5%, `common`); every summary records its basis in `mode` and, where it has one, `scope` (`final` = each run's last checkpoint, `all`, `checkpoint`).
- `achievement_status(episodes, curves=None, mode=)`: per unit and achievement, `final_rate`, `best_rate` and `best_update` over the checkpoints, `seeds_ever` of `n_seeds` (any eval, or training given `curves`), `eval_seeds`, `train_seeds`, `train_logged` and `status` (`never`, `never in eval`, `rare`, `common`), in tier, floor, value order. `ACHIEVEMENTS` (name -> `Achievement(tier, floor)`, value order) is the tier and floor table.
- `death_causes(episodes, mode=)`: cause-of-death shares per unit, as one row set per eval checkpoint (`scope` `checkpoint`) and one over each run's own last checkpoint (`scope` `final`); a death with an empty cause is `not recorded` (the real `unknown` cause stays apart); a set without deaths has one `(no deaths)` row with NaN share.
- `stall_table(episodes, mode=, final=True)`: one row per stall (frames, floor, `ending`, `ended_by`, `started_by`); `stall_stats(episodes, mode=, n=, final=)`: per unit stalls per episode (interval over seeds), median length, escape rate and `start_floor/<f>` counts; `stall_starts(table)`: where stalls start (floor, unlock); `stall_ccdf(table)`: the length CCDF.
- `plot_floor_reach`, `plot_achievement_status`, `plot_death_causes`, `plot_stall_lengths` (table, `out_dir`) -> the PNG path (`floor_reach.png`, `achievement_status.png`, `death_causes_over_training.png`, `stall_lengths.png`), dark style of `visualisation.charts`; one budget per plot, a unit per line or panel.
- Comparison tables (`compare`), each a DataFrame with the listed column constant; `n_boot` and `seed` set the bootstrap (protocol: 10,000):
  - `main_table(frames)` (T1, `main_columns()`): per (benchmark, block, method) of the tuned runs: seeds, mean +- s.e., mean CI, IQM [CI] at 1M or median [CI] and t interval at 1B, final-checkpoint eval means, and per control `poi_<c>`, `poi_lo_<c>`, `poi_hi_<c>`, `p_<c>`, `p_holm_<c>` (1M) or `diff_<c>` with its Welch t interval (1B); `indicative` marks 1B.
  - `poi_matrix(frames)` (F3, `POI_COLUMNS`): PoI with CI between every two members of one block at 1M.
  - `ablation_table(frames)` (`ABLATION_COLUMNS`, R77): each ablation of a final phase (`simhash`, `single-buffer`, `random-top`, `monte-carlo`, `no-flags`, `kappa0`) against its method's tuned runs: difference of means with a bootstrap CI (1M) or Welch t interval (1B), and at 1M PoI and Holm p within the block.
  - `reproduction_gate(frames, published=, published_hash=)` (`GATE_COLUMNS`): per control and benchmark with runs, over the runs of the published config hash (`reproduction.published_config_hashes`): `pass` or `fail` with gap and `h` of `reproduction.band`, `too few seeds`, `no published spread`, `published config not run`, or the PQN-RNN 1M reference-code note (`REFERENCE_CODE`); never raises.
  - `literature_table(frames)` (T2, `LITERATURE_TABLE_COLUMNS`): per comparable (score kind, benchmark) our rows (mean, s.e., CI, gate on training control rows), then the one literature block; `ours_above`, `ours_within`, `ours_below` only on rows of our protocol (`protocol_match`: `same`, `different protocol`, or `protocol not reported` for `n.r.`), `indicative` on 1B groups.
  - `stability_table` (T3, `STABILITY_COLUMNS`), `hardware_table` (T4 per profile, `HARDWARE_COLUMNS`, laptop rows carry `CONTEXT_NOTE`), `race_table` (`RACE_COLUMNS`, `wall_note`), `game_summary_table` (T6, `GAME_COLUMNS`, `wall_note`), `sensitivity_table` (T7 per tuning phase, `SENSITIVITY_TABLE_COLUMNS`; raises when no run of a phase is tagged `best`); T2, T3, T4 and T6 carry `indicative`.
  - `tuning_optimism(frames)` (`TUNING_OPTIMISM_COLUMNS`, R101, R102): per tuned unit the 3-seed tuning mean of its config hash (tune-fine, else tune-coarse, `TUNING_ORDER`), the mean of its fresh final seeds (not reused from tuning, `fresh_seeds`) and `optimism` = tuning - fresh; NaN with an `n/a` `optimism_note` when the config was selected on all its final seeds (the controls), has fewer than `MIN_FRESH` = 5 fresh seeds or was never tuned.
  - `block_of(method, combo_pair)`: `skills` for `SKILL_METHODS` (select_pair's) and a combo whose own pair holds one, else `main`; `BLOCKS`.
- `find_cases(frames, method, budget, config_ref="tuned", mode="stochastic")`: the case episodes of one unit as `cases.csv` rows (`CASE_COLUMNS`: contracts section 4, then `env_steps`, `length`, `detail`); kinds `stall`, `deep_floor`, `rare_unlock`, `death`; `id` `<budget>-<method>-<kind>-<n>`, an ablation `<budget>-<method>-<config_ref>-<kind>-<n>`.
- `export_case(case, out_dir, run=None, *, tracking_uri=, root="outputs", overlay_kw=)`: replays one case and returns it with `rollout`, `master`, `gif` and `overlay` filled (paths relative to `root`); `run` is an `EvalRun` (shared compiles), a run directory, or looked up by `run_dirs`.
- `describe_case(case, run=None, *, tracking_uri=)`: the write-up facts of a case from one replay: floor, stats and inventory at `start` and `peak`, action counts in the window and the episode, coverage gained per floor, the 50 actions before the peak, combat damage, close calls and lowest health in the window.
- `write_cases(cases, root)`: `analysis/<budget>/cases.csv` (all methods of the budget) and `analysis/<budget>/<method>/cases.csv`; `run_dirs(run_ids, tracking_uri)`: artifact directories from a SQLite store, read-only.

## Modules

One line each; signatures and details: [[Modules]].

- `cases`: case episodes per unit from the `hard` tables, their replay export (rollout, master, GIF, internals clip) and their facts.
- `compare`: the comparison's result tables (T1-T4, T6, T7, the PoI matrix, the ablation table, tuning optimism) and the reproduction gate.
- `hard`: floor reach, achievement status, cause of death and stall summaries per unit, and their plots; the comparison figures (F9) import the death and stall functions.
- `literature`: the published scores of blocks A, A2 and D of [[Existing Implementations]] and the like-for-like block of each of our score kinds.
- `load`: loaders, schema checks, run selectors and base controls over the export tables of [[evaluation]].
- `obstacles`: the 10 shared and 23 method obstacle detectors, their registry and the obstacles table.

## Data flow

- In: the export tables of [[evaluation]] under `outputs/metrics/<phase>/` (`runs`, `curves`, `eval_episodes` parquet; column contract in the Phase 4 contracts); eval files and checkpoints through [[visualisation]]; `configs/literature/existing_implementations.csv` (rows copied from [[Existing Implementations]]).
- Detectors read `curves` (stored names, `ALIASES`), `eval_episodes`, the base control's tuned runs of the same frames (one pool per method, seed medians; a control has one pool per other control, a combo run its pair base), and the optional frames below.
- Optional input frames of `detect_all` (a missing one reads `needs eval actions` or `needs replay (evaluation Task 6)`; contracts "Replay inputs for training analysis"):

| Frame | Column | Type | Meaning | Source |
|---|---|---|---|---|
| `eval_actions` | `run_id`, `update`, `mode`, `eval_seed` | str, int, str, int | the eval episode | `eval_actions(run_dir, ...)` from `eval/<mode>/<update>.npz` |
| `eval_actions` | `actions` | int array | the episode's actions, padding cut at `length` | eval file |
| `skill_ends` | `run_id`, `update`, `env_steps`, `mode`, `eval_seed` | str, int, int64, str, int | the eval episode and checkpoint | replay |
| `skill_ends` | `start`, `end` | int | episode steps where the skill started and ended | replay |
| `skill_ends` | `skill` | str | `skills.SKILLS` name | eval `extras/skill` |
| `skill_ends` | `key` | int64 | `smdp.top_key` table row at the decision (`low_flags` as the run's config) | replay |
| `skill_ends` | `reason` | str | `skills.REASONS` name of the end (`success`, `failure`, `timeout`, `episode_end`, `floor_change`, `low_stat`) | replay |
| `skill_ends` | `pair_count` | int | Dyna-Q only: real outcomes of (`key`, `skill`) in the checkpoint's model, `count` of `eval/dyna_model_graph.parquet` (0 if absent) | model graph of the checkpoint |
| `eval_inventory` | `run_id`, `update`, `mode`, `eval_seed` | str, int, str, int | the eval episode | replay |
| `eval_inventory` | `steps` | int | episode steps | replay |
| `eval_inventory` | `capped_steps` | int | steps with at least one material at or above its `progress_key.CAPS` cap | replay |
- Cases replay eval files through [[visualisation]] (`as_rollout`, `render_master`, `cuts.gif`, `render_overlay`, `stored_episode`) and read state facts with `episode_records.trace_of` and `coverage.update_seen`.
- Case files (`export_case`, `out_dir` = `outputs/analysis/<budget>/<method>/`): `rollouts/case-<id>.npz`, `masters/case-<id>.mkv` (whole episode, frame = step), `gifs/<id>.gif`, stall `overlays/<id>.mp4`; the rollout is copied to `outputs/visualisation/runs/case-<id>.npz` (flat, for the video `runs/*.npz` glob), and that copy is the `rollout` column. A master's stem equals its rollout's, as `cuts.cut_kept` looks masters up.
- The comparison tables read `runs` (final score, eval scores, hardware columns, `config_json` for combo pairs and sensitivity), `curves` (`train/score_pct`, `first_unlock/*`, `first_unlock_wall/*`), final-checkpoint stochastic `eval_episodes`, `configs/reference/published_scores.yaml` and the literature CSV.
- Out: `outputs/analysis/<budget>/`, `outputs/results/`, MLflow experiment `analysis`.

## Invariants

- Runs belong to a phase by their `phases` tag, never by experiment name; statistics are per (method, `config_ref`): `select` refuses to pool them unless told to.
- Loading checks the contract columns, a unique `run_id`, the phase tag, that curves and episodes belong to a run of `runs`, and that the episodes' `method`, `config_ref` and `budget` equal the run's.
- A phase that was not exported (`final-1b`, `combo-1b`) raises `MissingPhaseError` naming it; `skip_missing=True` loads only the phases that exist, and an incomplete export still raises.
- Parquet-only use imports neither JAX nor MLflow: `CONTROLS`, `BASE_CONTROL` (module attributes built on first access) and `pair_of` import the registry when used; `CONTROLS` never holds combo.
- Controls come from `select_pair.controls_of` and `SKILL_METHODS`, never copies; a combo run's base control is read from its own `config_json` pair (`combo_pair`), never from combo.yaml.
- Literature is compared like with like only: training-episode score at 1B with block A, at 1M with block D, stochastic evaluation at 1B with block A2; greedy evaluation and stochastic at 1M have no block and raise. Blocks A3, B, B2, C, C2, C3 and E to I are not in the CSV.
- The CSV copies the note's cells as written; `spread` is set only when the score cell states an sd or s.e. of the % score itself (a raw-return sd, a CI or an sd of another variant gives `spread_kind` `none`; `score_text` keeps the cell). Nothing is recomputed from figures. After a note edit the test against the note shows which rows drifted.
- `protocol` follows one rule: the block's protocol, except `n.r.` when the row's source note says it is unstated or undocumented; block D has no heading protocol, so it is read per row from the notes. T2 shows it, because block D's Simulus row is a test-episode score.
- Every `note` value is the stem of a file in `research/Existing Implementations/`.
- A detector whose series, base-control series or input frame is missing reports `not logged`, `needs replay (evaluation Task 6)` or `needs eval actions`, never `absent` (R78); `table_rows` counts runs as detected over judged (detected or absent) and lists the others.
- Thresholds are module constants. Plateau is relative to the run's best score except at `ABSOLUTE_BUDGETS` (1B: 1 point); bonus starvation is `uninformative at 1M` (`STARVATION_BUDGETS`) (R79).
- `detect_all` judges one (method, `config_ref`) unit per call (R77); combo runs keep the method detectors of their own pair. Base pools are per method, never mixed: a control's exploration failure needs fewer floors than every other control, each named in the evidence.
- A base run that never logged a series (`first_unlock/*` in the reused runs) is left out of the base median, never counted as never unlocked; the evidence gives the runs used and left out (R78).
- Per-skill stats are required one by one (`smdp/skill/*/<stat>`, each deep skill's `chosen_share`), so one missing stat reads `not logged`.
- The `hard` summaries group by (`method`, `config_ref`, `budget`) themselves, so tuned and ablation runs in one frame stay apart (R77); intervals are `stats.stratified_bootstrap_ci` over seeds (one run per seed, 10,000 resamples), skipped only when every seed has the same value.
- `hard` reads only the event stream (`max_floor`, `ach/*`, `unlock_step/*`, `cause_of_death`, stall frames); the replay-only fields (`gathered/*`, `used/*`, `coverage/*`, `survival/level/*`, a stall's `intrinsics` and `inventory`) are never read, so a missing replay never becomes 0. Undefined values are NaN or a status: a checkpoint without deaths has a NaN share, a unit without stalls NaN median length and escape rate, a missing training series `train_seeds` NaN and status `never in eval` (never asserted as `never`, R78).
- Final checkpoint and stochastic mode are the defaults of every `hard` summary (thresholds' basis); an achievement's `seeds_ever` counts both modes and every checkpoint. `ACHIEVEMENTS` floors are home floors checked against the game: mob floors from `FLOOR_MOB_MAPPING`, ore (sapphire, ruby on 2), chest loot on its first floor (bow, chest, potion 1; books, so spells and enchanting, 3), crafting on 0; `started_by` ties break by `.value`. Plots name mode and scope in their title and refuse a table that mixes them; `achievement_status.png` draws `never` (solid cross) and `never in eval` (hollow diamond) apart.
- Cases are found per (method, `config_ref`), tuned unless an ablation is named (R77), in stochastic final-checkpoint episodes as the plan's thresholds say; earlier stall checkpoints are spread evenly over those with an escaped stall; ties break by fixed columns, so a rerun gives the same table. A deep-floor case on floor 0 peaks at its last unlock (every episode starts on floor 0).
- Windows: a stall case runs from the stall start to 20 steps after the unlock at 8x; other kinds 150 steps before to 49 after the peak at 1x; a GIF keeps the last 20 s of its window, so it ends with the peak. An export whose master replay diverges from the stored episode raises.
- `find_cases` and `write_cases` import neither JAX nor MLflow; the replay side imports JAX inside the functions. Exports write only under `out_dir` and `root`, never the store (R85).
- Instability and collapse pass `stability`'s drawdown floor: 1 point at `ABSOLUTE_BUDGETS`, else `max(0.1, 0.25 x` the run's best smoothed score), named in the evidence (R92).
- Flat score (bonus distraction, reward hacking, re-trying) is a documented choice: the method plan says "flat (drawdown definition of evaluation-and-metrics)" and evaluation defines only drawdowns, so flat is a gain over the trailing 10% of the budget below 5% of the run's best smoothed score (`FLAT_WINDOW`, `FLAT_GAIN`).
- Bonus-driven re-trying takes the pair count from the Dyna-Q model graph and the success rate from the eval ends of that pair at the checkpoint: the model stores no success flag.
- `DETECTORS` is built on first access (owners of the skill detectors are `select_pair.SKILL_METHODS`); importing `obstacles` imports neither JAX nor MLflow.

- Comparison: every table runs per benchmark and block; the three controls are the reference rows of both blocks and a method never meets one of the other block (PoI matrix, race). IQM, PoI and Mann-Whitney exist at 1M only, with one Holm family per block (T1: block methods x controls; ablations: the block's ablations); 1B rows give mean, median and CIs and are indicative.
- T1, T2, T3, T4, T6 and the PoI matrix use the tuned runs; ablations appear only in `ablation_table`, each against its own method's tuned runs (R77). T2 shows our training score beside block D (1M) or A (1B) and the stochastic eval beside A2 (1B); never two blocks in one group.
- The gate reads only the published config's runs, skips budgets without runs (R86) and reports instead of raising; T2 carries its status, gap and `h` on the control rows. T2 gives a CI verdict only against rows of our protocol.
- Race (T6): per method `n` = runs that logged `first_unlock/*` (R78), `k` = runs that unlocked; medians are lower medians over all n runs with +inf for a never-unlocked run (as the obstacles base medians); eligible at k >= n / 2; lowest median wins, wall-clock only within one profile; never-logged methods read NaN wins, not 0. Laptop wall-clock (race, T4) is context only and says so in `wall_note` or `note` (R97, two concurrent workers).
- T6 weights seeds equally, the top cause of death included (mean per-seed share of deaths).
- T3 uses `obstacles.drawdown_floor` (R92), the detectors' own rule.
- Tuning optimism (R101, R102) compares the tuning mean only with fresh final seeds: reused seeds are the very runs the selection saw, so including them would shrink the winner's curse.

## Known issues

- `export_case` replays a stall case's episode five times: `as_rollout`, the master render and `render_overlay` (its own `EvalRun`, which replays for the observations, the network outputs and the frames); sharing one replay needs an `EvalRun` or episode parameter on `as_rollout` and `render_overlay` ([[visualisation]]).
- Key aliasing, survival failure despite flags and bonus-driven re-trying need the `skill_ends` replay frame, and hoarding under log2 keys (the tuned count-bonus-pqn-rnn config) the `eval_inventory` frame; until evaluation Task 6 builds them they read `needs replay (evaluation Task 6)`.

## Tests

- [[tests/analysis/test_load.py]], [[tests/analysis/test_literature.py]] (the CSV against its source note and the block mapping), [[tests/analysis/test_compare.py]] (blocks, Holm families per block, 1B rows and Welch intervals, PoI matrix, ablations, the gate and its published config, T2 blocks, CI kind and protocol match, T3 collapse, T4 profiles, the race rule with censored seeds, T6, T7, tuning optimism) on `_compare_fixture.py` (in-memory frames: 10 seeds at 1M, 3 at 1B, the six ablations, tuning-reused seeds, a skill-pair combo option) and the shared export, [[tests/analysis/test_hard.py]] (hand-built episode frames: rates, intervals against the shared bootstrap, thresholds, NaN and status cases, units kept apart, the tier table against the game, the plots), [[tests/analysis/test_obstacles.py]] (a firing and a quiet synthetic run per detector, not-logged and needs-replay statuses, the cycle check, the table); `tests/analysis/conftest.py` (fixtures) and `_analysis_export.py` (its builder) write a synthetic export of every registry method, two ablations and a combo run, for the other analysis tests.
- [[tests/analysis/test_cases.py]] (TA3): hand-built episode frames for each case threshold and the table contract; export, GIF cap and facts on the tiny PPO eval run of the visualisation tests.
