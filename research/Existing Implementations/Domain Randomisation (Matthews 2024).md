---
title: "Domain Randomisation (Matthews 2024)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Michael Matthews
  - Michael Beukman
  - Benjamin Ellis
  - Mikayel Samvelyan
  - Matthew Jackson
  - Samuel Coward
  - Jakob Foerster
year: 2024
venue: "ICML 2024 (PMLR 235)"
paper: "https://arxiv.org/abs/2402.16801"
code: "https://github.com/DramaCow/jaxued/blob/main/examples/craftax/craftax_plr.py"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ued
  - ppo
created: 2026-10-20
---
# Domain Randomisation (Matthews 2024)

## TL;DR

- Control arm of the Craftax paper's UED study: PPO on uniformly sampled, normally generated levels, run in the JaxUED pipeline.
- Competitive with all UED methods; only [[PLR (Matthews 2024)]] is significantly better ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Exact reward: not reported (curves only); 10 runs on 20 fixed evaluation levels.

## Method

- Level sampling: the standard Craftax generator, as in ordinary training ([paper](https://arxiv.org/abs/2402.16801) §4.2).
- Same Craftax-specific rollout changes as the UED arms: short inner rollouts, episode cap 4096 ([paper](https://arxiv.org/abs/2402.16801) App. G.1).
- Policy: feed-forward MLP actor-critic, width 512; no recurrence.
- Differs from [[PPO (Matthews 2024)]] in hyperparameters (below) and in the evaluation protocol.

## Training setup and cost

| Hyperparameter | Value ([paper](https://arxiv.org/abs/2402.16801) Table 14, DR column) |
|---|---|
| Network width | 512 |
| Environments | 1024 |
| γ | 0.99 |
| GAE λ | 0.9 |
| PPO inner / outer rollout | 64 / 64 |
| PPO epochs | 5 |
| Minibatches per epoch | 2 |
| Clip range | 0.2 |
| Adam learning rate | 0.0002 |
| Anneal LR | yes |
| Max gradient norm | 1.0 |
| Value clipping | yes |
| Return normalisation | no |
| Entropy coefficient | 0.01 |

- Budget: 1B environment steps.
- Hardware, wall-clock, parameters, memory: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | evaluation reward on 20 fixed levels, % of max (226) | not reported as a number; second to PLR | 10 | [paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7 |

## Problems and limitations

- Evaluation on in-distribution levels favours DR, per the authors ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Not the same configuration or metric as the scoreboard PPO, so the two are not interchangeable.

## Code

- Repository: https://github.com/DramaCow/jaxued, `examples/craftax/craftax_plr.py` (DR mode).
- Framework: JAX. Licence: Apache-2.0 (declared in `pyproject.toml`).
- Citation: Matthews et al., ICML 2024, arXiv:2402.16801; Coward et al., JaxUED, arXiv:2403.13091.
