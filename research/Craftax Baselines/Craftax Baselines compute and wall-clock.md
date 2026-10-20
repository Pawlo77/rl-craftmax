---
title: "Craftax Baselines compute and wall-clock"
type: benchmark
parent: "[[Craftax Baselines]]"
tags:
  - craftax
  - compute
  - hardware
  - jax
source: "https://arxiv.org/pdf/2402.16801v2"
created: 2026-10-20
---
# Craftax Baselines compute and wall-clock

> [!abstract] Summary
> The authors measured speed on one RTX 4090; Craftax-Symbolic reaches about 267k steps per second end to end, so a Craftax-1B run takes about an hour on that GPU. No per-algorithm wall-clock or total GPU-hours are reported.

## Reference machine

- GeForce RTX 4090 with 24 GB VRAM, i9-13900K (24 cores, 32 threads), 32 GB RAM ([paper](https://arxiv.org/pdf/2402.16801v2) Appendix B).
- The [project page](https://craftaxenv.github.io/) repeats: "All experiments were run on a single machine with an RTX 4090 and i9-13900K."
- `run_docker.sh` launches one container per GPU and treats `all` as GPUs 0-7 ([run_docker.sh](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/run_docker.sh)); the paper does not say which machine ran the full sweeps.

## Speed test (paper Appendix B, Table 1)

| Environment | Best-case steps per second | Workers at best case |
|---|---|---|
| Craftax-Classic | 405,618 | 4096 |
| Craftax | 266,961 | 4096 |
| Procgen | 7,638 | 1024 |
| NetHack | 5,628 | 64 |
| Crafter | 1,580 | 1024 |
| MineRL | 133 | 8 |

- Craftax-Classic and Craftax are 257x and 169x faster than Crafter (Figure 2).
- Craftax rows used `Craftax-Classic-Symbolic` and `Craftax-Symbolic` with a PureJaxRL end-to-end PPO pipeline.
- Steps per second are averaged over the whole RL learning process, not a random-agent rollout.
- Workers were doubled from 1 until the machine ran out of memory.
- Network size, epochs and minibatch size were matched across environments as far as possible.
- Best case uses 4096 workers; the baselines' default is 1024 (Craftax-1B) or 256 (Craftax-1M).

## Run times reported

| Run | Time | Source |
|---|---|---|
| Craftax-1B (1e9 steps, Craftax-Symbolic) | "a run on a single RTX 4090 can finish in an hour" | Section 3.4 |
| Craftax-Classic, PPO, 1e9 steps | under an hour on a single GPU; 51 minutes to reach 90% of max return | Abstract; Introduction |
| Craftax-1M | "experiments can take only seconds to finish" | Section 3.4 |
| PPO with pixels, 1e9 steps | roughly 10k steps per second, about a day (maintainer estimate) | [Craftax issue #34](https://github.com/MichaelTMatthews/Craftax/issues/34) |
| PPO-RNN vs PPO | the RNN "only slightly reduced steps per second" | Craftax issue #34 |

- Derived: 1e9 / 266,961 steps per second is about 62 minutes for Craftax and 1e9 / 405,618 is about 41 minutes for Classic, consistent with the claims.
- Derived: 1e9 steps at 10k steps per second is about 28 hours.
- Pixel environments: no measured wall-clock in the paper; symbolic is about 10x faster than pixels (Section 3.1).

## Speed features in the code

- Optimistic resets: only one new world per 16 workers is generated each step (64 worlds per step at 1024 workers); about a 2x speedup on Craftax with "essentially no effect" on learning (Appendix C).
- Paper estimate: with mean episode length 200 and 1024 workers, the chance of more resets than generated worlds in one step is below 1e-10 (Appendix C).
- The whole training loop, including rollouts, is one jitted function; `--num_repeats` vmaps several seeds on one GPU ([ppo.py L635-L642](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L635-L642)).
- The printed `SPS` divides total steps by time including JIT compilation; the W&B `sps` key measures time between logging calls.

## Memory

- JAX preallocates 75% of GPU memory at import by default; the maintainer points to the JAX GPU memory allocation docs to change this ([Craftax issue #28](https://github.com/MichaelTMatthews/Craftax/issues/28)).
- Derived from code: one Craftax-1B rollout holds 64 x 1024 symbolic observations of 8268 float32, about 2.17 GB; `ppo.py` stores both `obs` and `next_obs`, about 4.33 GB per seed before activations and other buffers.
- Derived: the Craftax-1M rollout (16 x 256) holds about 0.14 GB of observations.
- Derived: the PPO MLP for Craftax-Symbolic has about 9.54M parameters (actor 4,781,099, critic 4,759,553).
- These derived sizes scale linearly with `--num_repeats`.

## Not reported

- Wall-clock per algorithm (PPO-RNN, ICM, E3B, RND) for Craftax-1B or 1M.
- GPU type or count used for the 10-seed runs and the hyperparameter sweeps.
- Total GPU-hours for the paper.
- Peak memory per algorithm.
