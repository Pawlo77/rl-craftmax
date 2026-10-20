---
title: "Craftax CPU vs GPU performance"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [concept, craftax, performance, jax, apple-silicon]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax CPU vs GPU performance

Which device to use for each part of Craftax training. Measured on an Apple M4 Pro (14 CPU cores, 20-core GPU, 48 GB) with craftax 1.6.1 and jax 0.11.2; Apple GPU through `metaljax` 0.11.9 ([[Craftax on Apple GPU]]). NVIDIA figures are the Craftax authors' ([[Craftax performance]]), not measured here. How to apply the result: [[Craftax runtime setup]].

## Summary

| Work | Fastest on Apple Silicon | Why |
|---|---|---|
| Stepping environments | CPU, env batch split over 4-8 host devices | metaljax replays scan bodies with gather/scatter step by step; Craftax is full of them |
| Network updates | Apple GPU (`metal`) | 3-4x faster than CPU for a 3 x 512 MLP at batch 4096 |
| End-to-end PPO, small network | CPU rollouts + GPU updates, by a small margin | env stepping dominates, so the update speed-up shows only as +4-5% |
| Everything on the Apple GPU | never | 3.4x slower than CPU end to end |
| NVIDIA GPU (CUDA) | everything on the GPU | authors' RTX 4090: 266,961 steps/s (Craftax) and 405,618 (Classic) end to end |

## End-to-end PPO training

- Setup: PPO adapted from Craftax_Baselines `ppo.py`, `Craftax-Classic-Symbolic-v1`, MLP 3 x 256, 64 steps per rollout, 1M environment steps, idle machine, 2026-10-20.
- Split = `JAX_PLATFORMS=metal,cpu`, env rollout on the CPU device, PPO update on the Metal device.

| Setup | Envs | Steps/s | Wall-clock for 1M steps |
|---|---|---|---|
| CPU only | 256 | 10,017 | 100 s |
| Split | 256 | 10,557 | 95 s |
| CPU only | 1024 | 9,711 | 102 s |
| Split | 1024 | 10,131 | 97 s |
| Metal only | 256 | 2,932 | 101 s for 0.3M |

- The split gains more as the network grows, since only the update moves to the GPU; the measured gain is for this small MLP.
- Final training returns agree across setups (3.4-4.8 at 0.3-1M steps); runs differ in random streams per backend, so they are not bit-identical.

## Rollouts with a policy (env + MLP forward)

- Jitted `lax.scan` of MLP 3 x 256 forward, action sampling and `vmap(env.step)`, 64 steps, best of 3 timed calls after one warm-up, CPU only.
- Shards = CPU split into host devices (`--xla_force_host_platform_device_count`), env batch sharded over them; rewards are identical with and without shards.

| Env | Envs | 1 shard | 4 shards | 8 shards |
|---|---|---|---|---|
| Craftax-Classic-Symbolic | 1024 | 12,123 | 20,762 | 20,070 |
| Craftax-Classic-Symbolic | 4096 | 9,027 | 17,560 | 19,600 |
| Craftax-Symbolic | 1024 | 2,917 | 3,865 | 4,212 |
| Craftax-Symbolic | 4096 | 2,942 | 3,788 | 4,101 |

- Gain: 1.7-2.2x on Craftax-Classic, 1.3-1.4x on Craftax.
- Ten shards cannot split 1024 or 4096 envs evenly; the batch must divide by the shard count.
- Same rollout on metal (Craftax-Symbolic, 1024 envs): 1,367 steps/s against 2,890 on one CPU device.

## Environment stepping and network updates alone

- Random-action rollouts and MLP updates from [[Craftax on Apple GPU]] (measured while other jobs shared the machine).

| Measurement | CPU | metal |
|---|---|---|
| Craftax-Symbolic, 4096 envs, steps/s | 2,881 | 1,575 |
| Craftax-Classic-Symbolic, 4096 envs, steps/s | 8,637-11,207 | 6,242 |
| Craftax-Pixels, 16 envs, steps/s | 1,266 | 104 |
| MLP 3 x 512 update, batch 4096, Craftax-Symbolic observations | 65.06 ms | 15.36 ms |
| MLP 3 x 512 update, batch 4096, Classic observations | 19.55-26.84 ms | 6.56 ms |

## Correctness

- metaljax matches the CPU: rewards and observations bit-exact on Craftax-Symbolic; on Classic 48 of 51,200 rewards differ by at most 2.98e-08 ([[Craftax on Apple GPU#Correctness]]).
- A 1024-env, 100-step random rollout in the repo environment gave the same reward sum (1091.4) on CPU and metal.

## What decides the choice

- Craftax spends most of a training step in environment code (world update, creatures, rendering of the symbolic view), which is gather/scatter heavy.
- On a CUDA GPU these run in parallel across thousands of envs; on metaljax they serialise, so the CPU wins on Apple Silicon.
- Bigger networks (wider MLPs, RNNs, transformers, world models) shift the balance towards the GPU learner.

## Not measured

- CUDA GPUs and the cluster nodes: no runs here; only the authors' RTX 4090 figures.
- CPU shards combined with the metal learner in one run.
- Pixel-observation training end to end.
