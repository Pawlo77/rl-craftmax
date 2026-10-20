---
title: "Craftax runtime setup"
type: handbook
domain: craftax
parent: "[[Craftax]]"
tags: [handbook, craftax, performance, jax]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/orchestration/runtime.py]]"
  - "[[tests/test_runtime.py]]"
---
# Craftax runtime setup

The fastest tested configuration for running Craftax in this repository. Device comparison behind it: [[Craftax CPU vs GPU performance]]. Code: [[orchestration]].

## Steps

1. `make install`: syncs every dependency group, incl. `metaljax` on Apple Silicon and `jax[cuda12]` on Linux.
2. `make warm`: builds the Craftax and Craftax-Classic texture caches and prints the actor and learner devices.
3. `cp .env.example .env`: turns on the persistent XLA compilation cache for every `make` target.
4. In training code, before the first JAX computation:

   ```python
   from wut_murl_craftax.orchestration import (
       configure_runtime, device_roles, env_mesh, shard_envs,
   )

   configure_runtime(host_devices=8)  # compile cache + 8 CPU shards
   roles = device_roles()             # actor: env rollouts, learner: updates
   with jax.default_device(roles.actor):
       obs, state = jax.vmap(env.reset, (0, None))(keys, params)
       obs, state = shard_envs((obs, state), env_mesh())
   # rollout on roles.actor; jax.device_put(batch, roles.learner) for the update
   ```

5. On Apple Silicon, start training scripts with `JAX_PLATFORMS=metal,cpu` so `device_roles()` sends updates to the GPU; leave it unset for CPU only.
6. On a CUDA machine run with the default backends: `device_roles()` puts rollouts and updates on the GPU.
7. Use `auto_reset=False` with Craftax_Baselines' `OptimisticResetVecEnvWrapper` for training ([[Craftax API and wrappers]]).

## Settings and their measured effect

| Setting | Where | Effect measured here |
|---|---|---|
| Texture cache warm-up | `make warm` | `make warm` rebuilding the Craftax cache: 40.3 s; with both caches present: 2.6 s (the Classic cache alone took 9.6 s in [[Rendering Craftax frames]]) |
| Persistent compile cache | `.env` (`JAX_COMPILATION_CACHE_DIR`) or `configure_runtime()` | CPU, Craftax-Symbolic 1024-env rollout: compile part of the first call about 6.3 s cold, 1.9 s warm (first call 29.0 s vs 24.5 s, of which 22.6 s is the rollout itself); cache 12 MB |
| CPU env shards | `configure_runtime(host_devices=8)` + `shard_envs` | rollouts 1.7-2.2x faster on Craftax-Classic, 1.3-1.4x on Craftax ([[Craftax CPU vs GPU performance#Rollouts with a policy (env + MLP forward)]]) |
| Actor/learner split | `JAX_PLATFORMS=metal,cpu` + `device_roles()` | end-to-end PPO 10,557 vs 10,017 steps/s with a 3 x 256 MLP (+5%) |
| Optimistic resets | `OptimisticResetVecEnvWrapper` | not measured here; the authors report about 2x ([[Craftax performance]]) |
| `OMP_NUM_THREADS=1` (exported by the Makefile) | `Makefile` | no effect: 20,253 vs 19,621 steps/s unset |

## Pitfalls

- Host shards must be set before JAX starts its CPU backend; `configure_runtime` after the first computation changes nothing.
- Extra host devices slow unsharded code: 11,044 vs 11,871 steps/s with 8 devices and no sharding, so shards are set in code, not in `.env`.
- The env batch must divide by the shard count (10 shards fail for 1024 envs).
- `env_mesh` uses an `Auto` axis: an `Explicit` axis fails inside Craftax with `ShardingTypeError` on `dynamic_update_slice`.
- `JAX_PLATFORMS=metal,cpu` makes metal the default device: code that does not use `device_roles()` then steps envs on the GPU, about 3.4x slower end to end.
- The compile cache does not speed up metal runs (first call 49.8 s cold, 49.5 s warm; compile part about 2 s either way).
- Texture caches live inside the installed package (`.venv/.../craftax/*/assets/texture_cache*.pbz2`); recreating `.venv` deletes them, so run `make warm` again. Set `CRAFTAX_RELOAD_TEXTURES=true` after editing game textures.
- metaljax refuses to run when other processes hold unified memory; raise `METALJAX_MEM_SYS_MB` ([[Craftax on Apple GPU]]).

## Tests

- [[tests/test_runtime.py]]: compile-cache config, CPU device roles, metal device roles (Apple Silicon), sharded rollout equals unsharded rollout over 4 host devices, both texture caches written.
