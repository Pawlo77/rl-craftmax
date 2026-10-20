---
title: "Craftax performance"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, performance, throughput]
created: 2026-10-20
---
# Craftax performance

All figures below are the authors' or issue reporters' numbers, not measured here. Apple-silicon specifics: [[Craftax on Apple GPU]].

## Authors' speed test

Hardware: one machine with a GeForce RTX 4090 (24 GB VRAM), Intel i9-13900K (24 cores, 32 threads), 32 GB RAM ([paper App. B](https://arxiv.org/html/2402.16801v2#A2)).

| Environment | Steps per second (best case) | Env workers (best case) |
|---|---|---|
| Craftax-Classic (symbolic) | 405,618 | 4096 |
| Craftax (symbolic) | 266,961 | 4096 |
| Procgen | 7,638 | 1024 |
| NetHack | 5,628 | 64 |
| Crafter | 1,580 | 1024 |
| MineRL | 133 | 8 |

- Source: [paper Table 1](https://arxiv.org/html/2402.16801v2#A2.T1).
- Method: steps per second averaged over a whole RL training run (end-to-end compiled PureJaxRL PPO for Craftax), not a random-agent loop; worker count doubled until the machine ran out of memory.
- Crafter and Procgen used stable-baselines3 PPO with `SubprocVecEnv`; NetHack the built-in IMPALA; MineRL random actions only, so its number is an overestimate.
- Craftax-Classic and Craftax are 257x and 169x faster than Crafter ([paper Fig. 2](https://arxiv.org/html/2402.16801v2#S1.F2)); the abstract rounds to "up to 250x".

## Wall-clock claims

| Claim | Source |
|---|---|
| PPO, 1B steps on Craftax-Classic: under an hour on one GPU, about 90% of optimal reward | [paper abstract](https://arxiv.org/abs/2402.16801) |
| Craftax-Classic solved to within 90% of max return by PPO in 51 minutes | [paper §1](https://arxiv.org/html/2402.16801v2#S1) |
| A Craftax-1B run "on a single RTX 4090 can finish in an hour" | [paper §3.4](https://arxiv.org/html/2402.16801v2#S3.SS4) |
| Symbolic observations run "around 10x faster" than pixels | [paper §3.1](https://arxiv.org/html/2402.16801v2#S3.SS1) |
| Optimistic resets: "around 2x" speed-up in Craftax | [paper App. C](https://arxiv.org/html/2402.16801v2#A3) |
| PPO on Craftax-Pixels: "of the order of ~10k SPS, so about a day to run 1e9 steps" (hardware not stated); adding an RNN "only slightly reduced" SPS | maintainer in [issue #34](https://github.com/MichaelTMatthews/Craftax/issues/34) |
| Up to 4096 parallel environment workers used | [paper §2.1](https://arxiv.org/html/2402.16801v2#S2.SS1) |

## Costs and settings that matter

- Resets are the costly step: a reset generates all 9 floors, and with gymnax-style auto-reset over 99% of generated worlds are discarded ([paper App. C](https://arxiv.org/html/2402.16801v2#A3)); use `OptimisticResetVecEnvWrapper` ([[Craftax API and wrappers]]).
- Reset ratio 16 (1 new world per 16 workers) is the paper's and `ppo.py`'s default; at 1024 workers and an assumed mean episode length of 200, P(more finished episodes than new worlds) is below 1e-10 per step.
- Compilation: the play scripts take about 30 s to the first frame and 20 s more to the first action ([README Play](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md)).
- Un-jitted code is very slow (a few seconds per step in `--debug`), because every `x.at[i].set(v)` copies the array outside `jit` (maintainer in [issue #17](https://github.com/MichaelTMatthews/Craftax/issues/17)).
- First use of a pixel env builds the texture atlas ("will take a minute") and caches it ([constants.py `load_all_textures`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L1119)).
- On import JAX preallocates 75% of GPU memory (60 GB seen on an H100); control it with JAX's memory settings ([issue #28](https://github.com/MichaelTMatthews/Craftax/issues/28), [JAX GPU memory allocation](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)).
- CPU reference point from an issue reporter: about 2,250 env-steps/s for Craftax-Classic at 256 envs, "indicative only", single machine ([issue #57](https://github.com/MichaelTMatthews/Craftax/issues/57)).
- The per-step cost does not shrink with fewer live creatures: JAX simulates the maximum number of each creature every step and masks unspawned ones ([paper App. A.1](https://arxiv.org/html/2402.16801v2#A1.SS1)).
