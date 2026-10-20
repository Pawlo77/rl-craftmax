---
title: "Craftax"
type: concept
domain: craftax
parent: "[[wut_murl_craftax]]"
tags: [craftax, concept, environment, overview]
created: 2026-10-20
aliases: [Craftax framework]
---
# Craftax

Craftax is an open-ended RL benchmark written entirely in JAX: a 2D, top-down, procedurally generated survival and dungeon-crawling game whose `reset` and `step` compile with `jax.jit` and batch with `jax.vmap`, so the whole RL loop runs on the accelerator ([README](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md); [paper §1, §2.1](https://arxiv.org/html/2402.16801v2#S1)).

## Lineage

| Environment | What it is | Source |
|---|---|---|
| Crafter (Hafner, 2021) | Python 2D "simplified Minecraft": one 64x64 world, 17 actions, 22 achievements, +1 per first unlock, 1M-step protocol | [paper §2.2](https://arxiv.org/html/2402.16801v2#S2.SS2) |
| Craftax-Classic | JAX rewrite of Crafter, mimics its dynamics "as closely as possible"; 257x faster than Crafter in the authors' end-to-end test | [paper §3.1, Fig. 2](https://arxiv.org/html/2402.16801v2#S3.SS1), [Appendix A](https://arxiv.org/html/2402.16801v2#A1) |
| Craftax | Main benchmark: Crafter mechanics plus NetHack-like ones (9 floors, 19 creatures, combat with physical/fire/ice damage, potions, enchantments, attributes, boss); 169x faster than Crafter | [paper §3.2](https://arxiv.org/html/2402.16801v2#S3.SS2) |

- Craftax-Classic is "solved" by PPO-RNN in 1B steps (all achievements but `COLLECT_DIAMOND` and `EAT_PLANT` reliable), which is why the authors propose Craftax as the benchmark ([paper App. F.3](https://arxiv.org/html/2402.16801v2#A6.SS3)).
- Interface follows gymnax (`reset(key, params)`, `step(key, state, action, params)`); since v1.6.0 the gymnax dependency is gone and the base classes are vendored ([release v1.6.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.6.0)).
- Baseline agents (PPO, PPO-RNN, ICM, E3B, RND) and the wrappers live in the separate [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines) repo.

## Key facts

| Fact | Craftax | Craftax-Classic | Source |
|---|---|---|---|
| Registered names | `Craftax-Symbolic-v1`, `Craftax-Pixels-v1` | `Craftax-Classic-Symbolic-v1`, `Craftax-Classic-Pixels-v1` | [craftax_env.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_env.py) |
| Actions | 43 | 17 | [paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3) |
| Symbolic obs | flat 8268 | flat 1345 | [paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3) |
| Pixel obs (H, W, C) | 130 x 110 x 3 | 63 x 63 x 3 | env `observation_space` |
| Achievements | 67 in code (paper text says 65) | 22 | [constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L410), [paper Table 6](https://arxiv.org/html/2402.16801v2#A4.T6) |
| Max reward | 226 | 22 | [README Scoreboard](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md) |
| Episode cap | 100,000 steps | 10,000 steps | `EnvParams.max_timesteps` |
| Map | 9 floors of 48x48 | 1 map of 64x64 | `StaticEnvParams.map_size` |
| Benchmarks | Craftax-1B, Craftax-1M (symbolic only) | not a benchmark | [paper §3.4](https://arxiv.org/html/2402.16801v2#S3.SS4) |
| Latest release | 1.6.1 (2026-06-20) | same package | [PyPI](https://pypi.org/project/craftax/) |

## Notes

- [[Craftax environments]]: variants, registered names, classes, params.
- [[Craftax API and wrappers]]: install, `reset`/`step`, `jit`/`vmap`, wrappers, play scripts, rendering.
- [[Craftax observations and actions]]: symbolic and pixel layouts, all actions.
- [[Craftax game content and achievements]]: floors, creatures, crafting, intrinsics, achievements by tier.
- [[Craftax scoring and evaluation]]: reward, % of 226, Craftax-1B/1M, scoreboard, logging.
- [[Craftax world generation]]: what `reset` builds, seeding, UED hooks.
- [[Craftax performance]]: reported throughput and hardware.
- [[Craftax on Apple GPU]]: running Craftax on Apple silicon.
- [[Craftax versions and dependencies]]: releases, dependency lists, Python support.
- [[Craftax known issues]]: open bugs, errata, gotchas.

## Licence and citation

- Licence: MIT, "Copyright (c) 2024 Michael Matthews" ([LICENSE](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/LICENSE)); PyPI classifier `License :: OSI Approved :: MIT License` ([pyproject.toml](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/pyproject.toml)).
- Paper: Matthews, Beukman, Ellis, Samvelyan, Jackson, Coward, Foerster, "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning", ICML 2024, [arXiv:2402.16801](https://arxiv.org/abs/2402.16801) (v1 2024-02-26, v2 2024-06-03).
- Citation requested by the README:

```bibtex
@inproceedings{matthews2024craftax,
    author={Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
    title = {Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
    booktitle = {International Conference on Machine Learning ({ICML})},
    year = {2024}
}
```

## Sources

- Code read at commit [`c3c2e0d`](https://github.com/MichaelTMatthews/Craftax/tree/c3c2e0d038c4e641f9481320c158f457f30c28f3) (2026-06-20, same code as PyPI 1.6.1) and Craftax_Baselines at [`7ce36fa`](https://github.com/MichaelTMatthews/Craftax_Baselines/tree/7ce36fa05b84a2c9e758012f1e6da402e1e3a891).
- Paper read from [arXiv HTML v2](https://arxiv.org/html/2402.16801v2).
- Issues read from [GitHub issues](https://github.com/MichaelTMatthews/Craftax/issues?q=is%3Aissue) (all 26, up to #60); releases from [GitHub releases](https://github.com/MichaelTMatthews/Craftax/releases); versions from [PyPI JSON](https://pypi.org/pypi/craftax/json).
