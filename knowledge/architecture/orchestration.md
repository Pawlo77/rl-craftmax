---
title: "orchestration"
type: module
domain: orchestration
parent: "[[wut_murl_craftax]]"
tags: [architecture, orchestration]
created: 2026-10-20
updated: 2026-10-28
related-code:
  - "[[src/wut_murl_craftax/orchestration/__init__.py]]"
  - "[[src/wut_murl_craftax/orchestration/runtime.py]]"
  - "[[src/wut_murl_craftax/orchestration/reproducibility.py]]"
  - "[[src/wut_murl_craftax/orchestration/tracking.py]]"
  - "[[src/wut_murl_craftax/orchestration/queue.py]]"
  - "[[src/wut_murl_craftax/orchestration/runner.py]]"
  - "[[src/wut_murl_craftax/orchestration/profiles.py]]"
  - "[[src/wut_murl_craftax/orchestration/store.py]]"
  - "[[src/wut_murl_craftax/orchestration/abort.py]]"
  - "[[src/wut_murl_craftax/orchestration/train_metrics.py]]"
  - "[[src/wut_murl_craftax/orchestration/greedy_tables.py]]"
  - "[[src/wut_murl_craftax/orchestration/prune.py]]"
---
# orchestration

## Responsibility

- Set up the JAX runtime for fast Craftax runs: compile cache, CPU env shards, actor and learner devices, texture caches ([[Craftax runtime setup]]).
- Snapshot the environment of each run (platform, packages, git state) for MLflow.
- Track experiment runs in a local MLflow store.
- Expand method and phase YAML files into runs, execute them resumably and abort on non-finite losses.
- Storage retention: keep only the run artifacts a later step reads (`prune`, `Profile.drop_state_on_finish`).

## Public surface

- `configure_runtime(cache_dir, host_devices)`: persistent compile cache; optional CPU host devices; call before the first JAX computation.
- `select_platforms(mode)`: `JAX_PLATFORMS=metal,cpu` for the `cpu-actor` mode on a machine with the Apple GPU plugin; before the first JAX computation.
- `import_craftax()`: imports every Craftax env module after `configure_runtime`; a first import inside a trace leaks tracers (`UnexpectedTracerError`).
- `device_roles() -> DeviceRoles`: `actor` (env rollouts) and `learner` (updates); CPU actor with a metal learner on Apple GPU, both on a CUDA GPU, both on CPU otherwise.
- `env_mesh(devices) -> Mesh` and `shard_envs(tree, mesh)`: split an env batch over host CPU devices.
- `warm_texture_cache() -> list[Path]`: build both games' texture caches; `make warm` runs it.
- `collect_reproducibility_report() -> ReproducibilityReport`: environment snapshot.
- `build_mlflow_tracker(...) -> MlflowRunTracker`: resumable MLflow run tracking.
- `tracking.resolve_tracking_uri(uri)`: a local store URI made absolute; refuses the repo's store under pytest (the CLIs' client constructor).
- `merge_stores(src, dst=None, names=None) -> {"copied", "joined", "refused"}`; `evaluation_complete(run_dir, log_spaced)`: both modes' eval files of every (or the final) checkpoint, pruned ones included (the prune keeps them, so a gap shows as `eval-incomplete`).
- `plan_prune(runs, exported, cases, phases=None) -> Plan`, `seed_picks(runs) -> {run_id: roles}`; CLI `python -m wut_murl_craftax.orchestration.prune [--tracking-uri URI] [--phases P ...] [--metrics-root DIR] [--analysis-root DIR] [--apply]` (dry run by default; reads the store read-only; relative paths resolve against the repo root; prints the export rows and case updates it loaded; `--apply` exits 2 when either is empty).

## Modules

One line each; signatures and details: [[Modules]].

- `runtime`: compile cache, device roles, CPU env shards, `select_platforms`, `import_craftax`, texture caches.
- `profiles`: machine profiles (`configs/profiles/<name>.yaml`): host devices, role mode, memory limit, save cadence, `drop_state_on_finish` (true in `cuda.yaml`); no hyperparameters.
- `queue`: run identity (`<registry>-<budget>-<hash8>-s<seed>`), `RunSpec`, expansion of method and phase YAML files; a combo spec carries its unhashed `combine` pair, which the runner saves in `config.json`.
- `runner`: resumable training, evaluation and MLflow logging of every `RunSpec` of a worker; the `Trainer` protocol.
- `abort`: non-finite-loss aborts (opt-in method metrics via `abort.finite`) and flag rules.
- `store`: phase status and the merge of node MLflow stores; the merge never copies an unfinished node run without `state/`: refused when new to the main store, joined (phases only) beside an unfinished main copy, even one with an unreadable state.
- `tracking`: resumable MLflow run tracking and the store location.
- `train_metrics`: windowed MLflow points of the per-update metrics.
- `reproducibility`: environment snapshot (platform, packages, git state) for MLflow.
- `greedy_tables`: eval-only rebuild of the SMDP greedy tables of one config hash from their checkpoints; clears only the schema-failure `warn/artifacts` tag (ruling R81).
- `prune`: storage retention of FINISHED, fully evaluated runs: deletes `state/`; every checkpoint of a tuning-only run (a `smoke` run keeps its last, the Classic PPO footage source); of a final run (`final-*`, `combo-*`) all but the last checkpoint, the case updates (`outputs/analysis/*/cases.csv`) and every checkpoint of the seeds `seed_picks` keeps whole (per budget, method and `config_ref`: best and lower-median `final_score`, best `eval_stochastic_score`, from `outputs/metrics/{final,combo}-*/runs.parquet`); a final run missing from those tables keeps every checkpoint. It writes `ckpt/pruned.json` before deleting and prints sizes per class (`state`, `ckpt_tuning`, `ckpt_final`). Kept apart and listed: runs tagged `warn/artifacts`, `ckpt/missing` or `warn/state_unreadable`, runs with an `owner.json` (a live queue; checked again just before deleting), runs without a `phases` tag, runs whose folder is outside the store's folder or whose `ckpt/`, `state/` or indexed files are links or leave `ckpt/`.

## Data flow

- In: environment variables (`JAX_PLATFORMS`, `XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR`), the experiment config.
- Out: JAX configuration and device handles to training code; run metadata to MLflow.
- Retention: the runner drops `state/` of a run that closes FINISHED (with `drop_state_on_finish`); `prune` reads the store, the export `runs` tables and `cases.csv` and deletes artifacts in the run folders.

## Invariants

- Sharding never changes results: a sharded rollout equals the unsharded one.
- Environment rollouts never default to the Apple GPU.
- A FINISHED run is never trained again: a lost (not pruned) checkpoint is rebuilt from `state/` without training, and without `state/` the run raises and stays FINISHED; pruned checkpoints (`ckpt/pruned.json`) are known-absent and never send a run back through `_train`.
- A run with checkpoints past its first chunk (or a prune marker) but no `state/` is never restarted from update 0 (`_refuse_stateless` raises before `config.json` or a tag is rewritten); a kill before the first state save (checkpoints 0 and the first chunk only) still restarts. An unfinished stateless run with its final checkpoint indexed and its score logged (a FINISHED run a cut queue call reopened) is only evaluated.
- `drop_state_on_finish` deletes `state/` only after the FINISHED close, with every eval file the phase needs and no lost checkpoint; killed, failed and unevaluated runs keep it.

## Known issues

- None.

## Tests

- [[tests/test_runtime.py]], [[tests/orchestration/test_runner.py]], [[tests/orchestration/test_queue.py]], [[tests/orchestration/test_abort.py]], [[tests/orchestration/test_profiles.py]], [[tests/orchestration/test_store.py]], [[tests/orchestration/test_training_metrics.py]], [[tests/orchestration/test_greedy_tables.py]], [[tests/orchestration/test_prune.py]]
