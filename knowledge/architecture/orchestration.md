---
title: "orchestration"
type: module
domain: orchestration
parent: "[[wut_murl_craftax]]"
tags: [architecture, orchestration]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/orchestration/__init__.py]]"
  - "[[src/wut_murl_craftax/orchestration/runtime.py]]"
  - "[[src/wut_murl_craftax/orchestration/reproducibility.py]]"
  - "[[src/wut_murl_craftax/orchestration/tracking.py]]"
---
# orchestration

## Responsibility

- Set up the JAX runtime for fast Craftax runs: compile cache, CPU env shards, actor and learner devices, texture caches ([[Craftax runtime setup]]).
- Snapshot the environment of each run (platform, packages, git state) for MLflow.
- Track experiment runs in a local MLflow store.

## Public surface

- `configure_runtime(cache_dir, host_devices)`: persistent compile cache; optional CPU host devices; call before the first JAX computation.
- `device_roles() -> DeviceRoles`: `actor` (env rollouts) and `learner` (updates); CPU actor with a metal learner on Apple GPU, both on a CUDA GPU, both on CPU otherwise.
- `env_mesh(devices) -> Mesh` and `shard_envs(tree, mesh)`: split an env batch over host CPU devices.
- `warm_texture_cache() -> list[Path]`: build both games' texture caches; `make warm` runs it.
- `collect_reproducibility_report() -> ReproducibilityReport`: environment snapshot.
- `build_mlflow_tracker(...) -> MlflowRunTracker`: resumable MLflow run tracking.

## Data flow

- In: environment variables (`JAX_PLATFORMS`, `XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR`), the experiment config.
- Out: JAX configuration and device handles to training code; run metadata to MLflow.

## Invariants

- Sharding never changes results: a sharded rollout equals the unsharded one.
- Environment rollouts never default to the Apple GPU.

## Known issues

- None.

## Tests

- [[tests/test_runtime.py]]
