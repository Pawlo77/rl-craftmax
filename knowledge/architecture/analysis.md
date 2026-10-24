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
  - "[[src/wut_murl_craftax/analysis/load.py]]"
  - "[[src/wut_murl_craftax/analysis/obstacles.py]]"
---
# analysis

## Responsibility

- Phase 4 analysis of finished runs: load the exported tables, per-method training analysis, obstacles, cases ([[2026-10-20-craftax-training-analysis]]) and the method comparison ([[2026-10-20-craftax-comparison-and-results]]).
- Filled by the training analysis and comparison tasks; so far the loaders and the obstacle detectors.

## Public surface

- `load_phase(phase_dir)`, `load_phases(phases, root, skip_missing=)`: the export tables as `{"runs", "curves", "eval_episodes"}` frames; `load_phases` keeps a run of two phases once, from the first phase listing it, and warns when the dropped copy differs; pass `final-*` before tuning phases.
- `select(frames, method=, config_ref=, budget=, phase=, pool=False)`: the matching runs and their rows in every frame; `phase` matches the `phases` tag (`in_phase(runs, phase)` is its mask). It raises when the result mixes `config_ref` values of a method and no `config_ref` or `pool=True` is given.
- `units(frames)`: each (`method`, `config_ref`) with its own frames.
- `BASE_CONTROL` (non-combo method -> control), `CONTROLS`, `base_control(row)`, `pair_of(row)`: the control a run is compared with; a combo run's comes from its own pair.
- `final_checkpoint(episodes, run=None)`: the last checkpoint's eval episodes per run.
- `MissingPhaseError` (a `FileNotFoundError` naming the phases), `SchemaError`, `FINAL_PHASES`, `METRICS_ROOT`, `UNIT_COLUMNS`.
- `detect_all(frames, method, budget, config_ref="tuned")`: one `Obstacle(name, run, start, end, evidence, status)` per run and detector of `DETECTORS[method]`; `status` is `detected`, `absent`, `not logged`, `needs replay (evaluation Task 6)`, `needs eval actions`, `not applicable` or `uninformative at <budget>`.
- `DETECTORS` (registry method -> `Detector` list), built from `shared_detectors()`, `count_bonus_detectors()`, `sil_detectors()`, `skill_detectors()`, `dyna_detectors()`; a `Detector` declares `REQUIRED_METRICS`, `BASE_METRICS`, optional `inputs`, informative `budgets` and owner `methods`.
- `ALIASES`: plan metric name -> stored names in order (contracts section 5; the comparison figures import it).
- `table_rows(obstacles, budget)`, `obstacles_table(rows)`: the obstacles table with exactly `TABLE_COLUMNS` of the plan template.
- `repeated_cycle(actions)` -> `Cycle(period, start, length)`: the shortest action period repeated over >= 50 steps; `eval_actions(run_dir, run_id, update, mode)` reads an eval file into the `eval_actions` input.

## Modules

One line each; signatures and details: [[Modules]].

- `load`: loaders, schema checks, run selectors and base controls over the export tables of [[evaluation]].
- `obstacles`: the 10 shared and 23 method obstacle detectors, their registry and the obstacles table.

## Data flow

- In: the export tables of [[evaluation]] under `outputs/metrics/<phase>/` (`runs`, `curves`, `eval_episodes` parquet; column contract in the Phase 4 contracts); eval files and checkpoints through [[visualisation]].
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
- Out: `outputs/analysis/<budget>/`, `outputs/results/`, MLflow experiment `analysis`.

## Invariants

- Runs belong to a phase by their `phases` tag, never by experiment name; statistics are per (method, `config_ref`): `select` refuses to pool them unless told to.
- Loading checks the contract columns, a unique `run_id`, the phase tag, that curves and episodes belong to a run of `runs`, and that the episodes' `method`, `config_ref` and `budget` equal the run's.
- A phase that was not exported (`final-1b`, `combo-1b`) raises `MissingPhaseError` naming it; `skip_missing=True` loads only the phases that exist, and an incomplete export still raises.
- Parquet-only use imports neither JAX nor MLflow: `CONTROLS`, `BASE_CONTROL` (module attributes built on first access) and `pair_of` import the registry when used; `CONTROLS` never holds combo.
- Controls come from `select_pair.controls_of` and `SKILL_METHODS`, never copies; a combo run's base control is read from its own `config_json` pair (`combo_pair`), never from combo.yaml.
- A detector whose series, base-control series or input frame is missing reports `not logged`, `needs replay (evaluation Task 6)` or `needs eval actions`, never `absent` (R78); `table_rows` counts runs as detected over judged (detected or absent) and lists the others.
- Thresholds are module constants. Plateau is relative to the run's best score except at `ABSOLUTE_BUDGETS` (1B: 1 point); bonus starvation is `uninformative at 1M` (`STARVATION_BUDGETS`) (R79).
- `detect_all` judges one (method, `config_ref`) unit per call (R77); combo runs keep the method detectors of their own pair. Base pools are per method, never mixed: a control's exploration failure needs fewer floors than every other control, each named in the evidence.
- A base run that never logged a series (`first_unlock/*` in the reused runs) is left out of the base median, never counted as never unlocked; the evidence gives the runs used and left out (R78).
- Per-skill stats are required one by one (`smdp/skill/*/<stat>`, each deep skill's `chosen_share`), so one missing stat reads `not logged`.
- Instability and collapse pass `stability`'s drawdown floor: 1 point at `ABSOLUTE_BUDGETS`, else `max(0.1, 0.25 x` the run's best smoothed score), named in the evidence (R92).
- Flat score (bonus distraction, reward hacking, re-trying) is a documented choice: the method plan says "flat (drawdown definition of evaluation-and-metrics)" and evaluation defines only drawdowns, so flat is a gain over the trailing 10% of the budget below 5% of the run's best smoothed score (`FLAT_WINDOW`, `FLAT_GAIN`).
- Bonus-driven re-trying takes the pair count from the Dyna-Q model graph and the success rate from the eval ends of that pair at the checkpoint: the model stores no success flag.
- `DETECTORS` is built on first access (owners of the skill detectors are `select_pair.SKILL_METHODS`); importing `obstacles` imports neither JAX nor MLflow.

## Known issues

- Key aliasing, survival failure despite flags and bonus-driven re-trying need the `skill_ends` replay frame, and hoarding under log2 keys (the tuned count-bonus-pqn-rnn config) the `eval_inventory` frame; until evaluation Task 6 builds them they read `needs replay (evaluation Task 6)`.

## Tests

- [[tests/analysis/test_load.py]]; [[tests/analysis/test_obstacles.py]] (a firing and a quiet synthetic run per detector, not-logged and needs-replay statuses, the cycle check, the table); `tests/analysis/conftest.py` (fixtures) and `_analysis_export.py` (its builder) write a synthetic export of every registry method, two ablations and a combo run, for the other analysis tests.
