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
---
# analysis

## Responsibility

- Phase 4 analysis of finished runs: load the exported tables, per-method training analysis, obstacles, cases ([[2026-10-20-craftax-training-analysis]]) and the method comparison ([[2026-10-20-craftax-comparison-and-results]]).
- Filled by the training analysis and comparison tasks; so far only the loaders.

## Public surface

- `load_phase(phase_dir)`, `load_phases(phases, root, skip_missing=)`: the export tables as `{"runs", "curves", "eval_episodes"}` frames; `load_phases` keeps a run of two phases once, from the first phase listing it, and warns when the dropped copy differs; pass `final-*` before tuning phases.
- `select(frames, method=, config_ref=, budget=, phase=, pool=False)`: the matching runs and their rows in every frame; `phase` matches the `phases` tag (`in_phase(runs, phase)` is its mask). It raises when the result mixes `config_ref` values of a method and no `config_ref` or `pool=True` is given.
- `units(frames)`: each (`method`, `config_ref`) with its own frames.
- `BASE_CONTROL` (non-combo method -> control), `CONTROLS`, `base_control(row)`, `pair_of(row)`: the control a run is compared with; a combo run's comes from its own pair.
- `final_checkpoint(episodes, run=None)`: the last checkpoint's eval episodes per run.
- `MissingPhaseError` (a `FileNotFoundError` naming the phases), `SchemaError`, `FINAL_PHASES`, `METRICS_ROOT`, `UNIT_COLUMNS`.

## Modules

One line each; signatures and details: [[Modules]].

- `load`: loaders, schema checks, run selectors and base controls over the export tables of [[evaluation]].

## Data flow

- In: the export tables of [[evaluation]] under `outputs/metrics/<phase>/` (`runs`, `curves`, `eval_episodes` parquet; column contract in the Phase 4 contracts); eval files and checkpoints through [[visualisation]].
- Out: `outputs/analysis/<budget>/`, `outputs/results/`, MLflow experiment `analysis`.

## Invariants

- Runs belong to a phase by their `phases` tag, never by experiment name; statistics are per (method, `config_ref`): `select` refuses to pool them unless told to.
- Loading checks the contract columns, a unique `run_id`, the phase tag, that curves and episodes belong to a run of `runs`, and that the episodes' `method`, `config_ref` and `budget` equal the run's.
- A phase that was not exported (`final-1b`, `combo-1b`) raises `MissingPhaseError` naming it; `skip_missing=True` loads only the phases that exist, and an incomplete export still raises.
- Parquet-only use imports neither JAX nor MLflow: `CONTROLS`, `BASE_CONTROL` (module attributes built on first access) and `pair_of` import the registry when used; `CONTROLS` never holds combo.
- Controls come from `select_pair.controls_of` and `SKILL_METHODS`, never copies; a combo run's base control is read from its own `config_json` pair (`combo_pair`), never from combo.yaml.

## Known issues

- None.

## Tests

- [[tests/analysis/test_load.py]]; `tests/analysis/conftest.py` (fixtures) and `_analysis_export.py` (its builder) write a synthetic export of every registry method, two ablations and a combo run, for the other analysis tests.
