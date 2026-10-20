---
title: "Craftax on Apple GPU"
type: handbook
domain: craftax
parent: "[[Craftax]]"
tags: [handbook, craftax, apple-silicon, jax]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax on Apple GPU

## Steps

1. Keep plain CPU JAX as the default backend: `craftax==1.6.1`, `jax==0.11.2`, `jaxlib==0.11.2`, `flax==0.12.10`, `optax==0.2.8`; no `[tool.uv]` overrides are needed.
2. For an opt-in GPU backend, add `metaljax==0.11.9` with marker `sys_platform == 'darwin' and platform_machine == 'arm64'` in a separate dependency group (it needs `jax<0.12`).
3. Select the GPU per run with `JAX_PLATFORMS=metal`; with the variable unset, metaljax leaves CPU as the default backend.
4. Check the device: `JAX_PLATFORMS=metal uv run python -c "import jax; print(jax.devices())"` prints `[MetalDevice(id=0)]`.
5. Never install `jax-mps` or `jax-metal` in the project environment: `jax-mps` makes `mps` the default backend on import, `jax-metal` forces `jax==0.4.35`.
6. For more CPU throughput, shard the env batch over host devices: `XLA_FLAGS=--xla_force_host_platform_device_count=8`, a 1-D mesh built with `axis_types=(jax.sharding.AxisType.Auto,)` (the default Explicit axis type fails inside Craftax with `ShardingTypeError` on `dynamic_update_slice`), batch size divisible by the device count.

## Hardware tested

- Apple M4 Pro, 14 CPU cores (10 performance, 4 efficiency).
- GPU: 20 cores, Metal 4.
- 48 GB unified memory.
- macOS 26.7.1 (build 25G241), Python 3.12, uv 0.12.23.

## Routes tried

### Apple jax-metal 0.1.1

- Last release 2024-10-08; Apple's table lists jaxlib `>=0.4.26` for 0.1.0 and nothing newer; `ENABLE_PJRT_COMPATIBILITY=1` is Apple's switch for newer jaxlib.
- Installed with `jax==0.4.35`, `jaxlib==0.4.35`, `flax<0.10.3` (resolved `flax==0.10.2`, `optax==0.2.5`); `jax.devices()` prints `[METAL(id=0)]` without `ENABLE_PJRT_COMPATIBILITY`.
- Coexists with craftax 1.6.1 only by holding jax, jaxlib and flax at 2024 versions.
- Silent wrong results: the first Craftax-Classic-Symbolic reset observation differs from CPU (same jax 0.4.35, same seed) at index 1341, the player-direction one-hot, `-1.0` on Metal against `1.0` on CPU; the step-1 reward differs (0 against 1).
- Same bug on Craftax-Symbolic: one of 8268 reset observation elements is off by 2.
- MLP update (batch 4096) loss differs from CPU by up to 0.0075 after 20 steps.
- A jitted rollout of 64 envs for 20 steps hung for over 10 minutes at 0% CPU, blocked in `xla::AwaitBuffersReady`; killed.
- Outcome: unusable.

### jax-mps (tillahoffmann/jax-mps, MLX-based PJRT plugin)

- Alpha; 0.11.0 released 2026-09-15 needs `jax>=0.11,<0.12`; enables itself on install and becomes the default backend (`[MpsDevice(id=0)] mps`).
- 0.11.0 fails on Craftax-Symbolic: `stablehlo.scatter: multi-axis window scatter with batching dims is not supported`, then `Output count mismatch: expected 4, got 0`.
- 0.11.0 runs Craftax-Classic-Symbolic.
- Pre-release `0.11.1.dev861` (2026-10-13, fixes batched scatter, issue 124) runs Craftax-Symbolic.
- `JAX_MPS_ASYNC_DISPATCH=1` (experimental) is suggested by the plugin; it changed SPS at N=64 from 389 to 404.
- First call for 4096 Craftax-Symbolic envs took 520 s.
- Un-jitted `vmap(env.reset)` over 4096 Craftax-Symbolic envs failed: `[metal::malloc] Resource limit (499000) exceeded`.
- Open upstream issues match the slowness: `while` loops sync the GPU per iteration (issue 83), `lax.scan` with stacked outputs grows quadratically (issue 215).
- Outcome: correct but 4 to 25 times slower than CPU on Craftax-Symbolic.

### metaljax (eterevsky/metaljax, MLX-based PJRT plugin)

- Beta; 0.11.9 released 2026-09-29 needs `jax>=0.11,<0.12`; wheel only.
- CPU stays default; `JAX_PLATFORMS=metal` gives `[MetalDevice(id=0)] metal`.
- Runs Craftax-Symbolic, Craftax-Classic-Symbolic, both pixel envs and a flax MLP update with no errors.
- Its README says scan bodies with gather or scatter fall back to per-timestep graph replay, which explains the low env throughput.
- Its memory governor refused to run while another process held GPU memory: `RESOURCE_EXHAUSTED: metaljax out of memory at execute ... over the 36.0 GB METALJAX_MEM_SYS_MB ceiling`; `METALJAX_MEM_SYS_MB` raises the ceiling.
- Outcome: works and matches CPU; env stepping is 1.4 to 8 times slower than CPU (closest at N = 1024 to 4096), MLP updates are 3 to 4 times faster.

### jax-mlx-plugin 0.0.4 (tsumme1/jax-mlx)

- Last release 2026-03-18, sdist only.
- Build fails against current MLX: `error: no member named 'raw_ptr' in 'mlx::core::allocator::Data'`.
- Outcome: not installable.

### Plain CPU JAX

- `jax==0.11.2`, `jaxlib==0.11.2`, `flax==0.12.10`, `craftax==1.6.1`.
- Fastest working route for env stepping at every batch size tested.

## Benchmark

- Rollout: `jax.jit` of `vmap(reset)` then `lax.scan` over `vmap(step)` with random actions, auto-reset envs, seed 0; SPS is env steps per second, mean of 3 timed calls after one warm-up call, `block_until_ready`.
- MLP: 3 x 512 tanh layers, policy and value heads, softmax cross-entropy plus value loss, Adam, batch 4096 symbolic observations, mean of 20 jitted updates.
- CPU Classic SPS varied up to 1.5 times between two runs on the same machine; both runs are listed.

| Env | N | T | CPU | CPU, 8 host shards | metaljax 0.11.9 | jax-mps 0.11.1.dev861 |
|---|---|---|---|---|---|---|
| Craftax-Symbolic | 1 | 20 or 100 | 331 | - | 6 | 13 |
| Craftax-Symbolic | 64 | 20 or 100 | 2,916 | - | 378 | 404 |
| Craftax-Symbolic | 1024 | 20 | 3,050 | 4,400 | 1,397 | 740 |
| Craftax-Symbolic | 4096 | 10 | 2,881 | 3,861 | 1,575 | 401 |
| Craftax-Classic-Symbolic | 1 | 10 or 100 | 1,227 to 1,293 | - | 16 | 2 (0.11.0) |
| Craftax-Classic-Symbolic | 64 | 20 or 100 | 5,128 to 7,334 | - | 986 | 242 (0.11.0) |
| Craftax-Classic-Symbolic | 1024 | 50 | 8,690 to 13,217 | 18,934 | 5,508 | - |
| Craftax-Classic-Symbolic | 4096 | 20 | 8,637 to 11,207 | 19,130 | 6,242 | - |
| Craftax-Pixels | 16 | 20 | 1,266 | - | 104 | - |
| Craftax-Classic-Pixels | 16 | 20 | 2,393 | - | 267 | - |

| MLP update, batch 4096 | CPU | metaljax 0.11.9 | jax-mps |
|---|---|---|---|
| Craftax-Symbolic obs (8268) | 65.06 ms | 15.36 ms | - |
| Craftax-Classic-Symbolic obs (1345) | 19.55 to 26.84 ms | 6.56 ms | 7.17 ms (0.11.0) |

- First-call (compile) time for 4096 Craftax-Symbolic envs: CPU 20.29 s, metaljax 27.99 s, jax-mps 520.14 s.
- jax-mps 0.11.0 Classic rows ran alongside a CPU benchmark; jax-metal produced no timed rows (wrong results, then a hang).

## Correctness

- metaljax against CPU (both jax 0.11.2): rewards, dones and reset observations bit-exact for Craftax-Symbolic at N = 1, 64, 1024, 4096.
- metaljax Craftax-Classic-Symbolic: dones and observations bit-exact; rewards bit-exact except 48 of 51,200 at N=1024, max abs difference 2.98e-08.
- metaljax pixels: observations within 1.19e-07 of CPU, rewards bit-exact.
- metaljax MLP: losses within 1.1e-05 (Symbolic) and 3.1e-06 (Classic) of CPU after 20 updates.
- jax-mps: Craftax-Symbolic rewards, dones and observations bit-exact at N = 1, 64, 1024, 4096; Classic bit-exact except 6 of 6,400 rewards (max 2.98e-08); MLP losses within 2.9e-06.
- jax-metal: wrong observation and reward values against CPU on the same jax version.
- Comparisons across jax versions are meaningless: jax 0.4.35 and jax 0.11 produce different random streams for the same seed.

## Known problems and workarounds

- Craftax env stepping is dominated by loop dispatch on all three Apple GPU plugins; no batch size up to 4096 made the GPU faster than CPU.
- jax-metal returns wrong numbers without any error; do not use it.
- jax-mps 0.11.0 cannot run Craftax-Symbolic; the batched-scatter fix is only in pre-releases (`--prerelease allow`).
- jax-mps compiles slowly at large batch (520 s for 4096 envs).
- metaljax is single-device; `jax.sharding` over the GPU is unsupported.
- metaljax's memory governor fails a run when other processes hold unified memory; raise `METALJAX_MEM_SYS_MB` or free memory.
- Training that fuses env and network in one jit (PureJaxRL style) runs at env speed, so metaljax's faster MLP does not help it.

## Recommendation

- Use plain CPU JAX (`jax==0.11.2`) for Craftax rollouts and end-to-end training on this Mac.
- Shard the env batch over 8 host devices for 1.3 to 2.2 times more CPU throughput at N = 1024 to 4096.
- Keep `metaljax==0.11.9` as an opt-in dependency group for network-heavy work outside the env loop; select it with `JAX_PLATFORMS=metal`.
- Reach for a CUDA GPU for full-scale Craftax runs; the authors' 4096-env speeds ([[Craftax performance]]) are about 90 times (Craftax-Symbolic) and 36 to 47 times (Classic) this Mac's single-device CPU.
