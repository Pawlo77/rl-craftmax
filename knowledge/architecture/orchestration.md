---
title: "orchestration"
type: module
domain: orchestration
parent: "[[wut_murl_craftax]]"
tags: [architecture, orchestration]
created: 2026-10-20
updated: 2026-10-24
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
---
# orchestration

## Responsibility

- Set up the JAX runtime for fast Craftax runs: compile cache, CPU env shards, actor and learner devices, texture caches ([[Craftax runtime setup]]).
- Snapshot the environment of each run (platform, packages, git state) for MLflow.
- Track experiment runs in a local MLflow store.
- Expand method and phase YAML files into runs, execute them resumably and abort on non-finite losses.

## Public surface

- `configure_runtime(cache_dir, host_devices)`: persistent compile cache; optional CPU host devices; call before the first JAX computation.
- `select_platforms(mode)`: `JAX_PLATFORMS=metal,cpu` for the `cpu-actor` mode on a machine with the Apple GPU plugin; before the first JAX computation.
- `import_craftax()`: imports every Craftax env module after `configure_runtime`; a first import inside a trace leaks tracers (`UnexpectedTracerError`).
- `device_roles() -> DeviceRoles`: `actor` (env rollouts) and `learner` (updates); CPU actor with a metal learner on Apple GPU, both on a CUDA GPU, both on CPU otherwise.
- `env_mesh(devices) -> Mesh` and `shard_envs(tree, mesh)`: split an env batch over host CPU devices.
- `warm_texture_cache() -> list[Path]`: build both games' texture caches; `make warm` runs it.
- `collect_reproducibility_report() -> ReproducibilityReport`: environment snapshot.
- `build_mlflow_tracker(...) -> MlflowRunTracker`: resumable MLflow run tracking.

## Modules

One line each; signatures and details: [[Modules]].

- `runtime`: compile cache, device roles, CPU env shards, `select_platforms`, `import_craftax`, texture caches.
- `profiles`: machine profiles (`configs/profiles/<name>.yaml`): host devices, role mode, memory limit, save cadence; no hyperparameters.
- `queue`: run identity (`<registry>-<budget>-<hash8>-s<seed>`), `RunSpec`, expansion of method and phase YAML files; a combo spec carries its unhashed `combine` pair, which the runner saves in `config.json`.
- `runner`: resumable training, evaluation and MLflow logging of every `RunSpec` of a worker; the `Trainer` protocol.
- `abort`: non-finite-loss aborts (opt-in method metrics via `abort.finite`) and flag rules.
- `store`: phase status and the merge of node MLflow stores.
- `tracking`: resumable MLflow run tracking and the store location.
- `train_metrics`: windowed MLflow points of the per-update metrics.
- `reproducibility`: environment snapshot (platform, packages, git state) for MLflow.

## Data flow

- In: environment variables (`JAX_PLATFORMS`, `XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR`), the experiment config.
- Out: JAX configuration and device handles to training code; run metadata to MLflow.

## Invariants

- Sharding never changes results: a sharded rollout equals the unsharded one.
- Environment rollouts never default to the Apple GPU.

## Known issues

- None.

## Tests

- [[tests/test_runtime.py]], [[tests/orchestration/test_runner.py]], [[tests/orchestration/test_queue.py]], [[tests/orchestration/test_abort.py]], [[tests/orchestration/test_profiles.py]], [[tests/orchestration/test_store.py]], [[tests/orchestration/test_training_metrics.py]]
