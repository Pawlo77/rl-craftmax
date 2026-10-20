---
title: "PLR (Matthews 2024)"
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
  - curriculum
  - ppo
created: 2026-10-20
---
# PLR (Matthews 2024)

## TL;DR

- Prioritized Level Replay (Jiang et al. 2021) on Craftax-1B, run with JaxUED as an unsupervised environment design (UED) baseline in the Craftax paper.
- Best of the UED methods tested; the only one significantly better than plain domain randomisation ([paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7).
- Exact reward: not reported (curves only); 10 runs, evaluated on 20 fixed levels ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Related: [[Domain Randomisation (Matthews 2024)]], [[Robust PLR (Matthews 2024)]], [[ACCEL (Matthews 2024)]].

## Method

- Keeps a buffer of training levels scored by estimated regret; each step either evaluates new random levels (adding high-regret ones) or trains on a sample of high-regret buffer levels ([paper](https://arxiv.org/abs/2402.16801) §4.2).
- Vanilla PLR also takes gradient steps on the newly generated random levels (exploratory gradient updates) ([paper](https://arxiv.org/abs/2402.16801) §4.2).
- Level = the full Craftax state, exposed as one JAX object, so UED methods apply directly ([paper](https://arxiv.org/abs/2402.16801) §2.4).
- Craftax-specific change: several short inner PPO rollouts per level, with regret estimated from the whole trajectory afterwards, so the agent updates often despite long episodes ([paper](https://arxiv.org/abs/2402.16801) App. G.1).
- Episode length capped at 4096 for UED ([paper](https://arxiv.org/abs/2402.16801) App. G.1).
- Policy: feed-forward MLP actor-critic (JaxUED `craftax_plr.py`); no recurrence (RNN policy listed as "coming soon" in the [JaxUED README](https://github.com/DramaCow/jaxued)).

## Training setup and cost

| Hyperparameter | Value ([paper](https://arxiv.org/abs/2402.16801) Table 14) |
|---|---|
| Network width | 512 |
| Environments | 1024 |
| γ | 0.99 |
| GAE λ | 0.9 |
| PPO inner rollout / outer rollout | 64 / 64 |
| PPO epochs | 5 |
| Minibatches per epoch | 2 |
| Clip range | 0.2 |
| Adam learning rate | 0.0003 |
| Anneal LR | yes |
| Max gradient norm | 1.0 |
| Value clipping | yes |
| Entropy coefficient | 0.01 |
| Replay rate p | 0.5 |
| Buffer size K | 4000 |
| Scoring function | MaxMC |
| Prioritisation | rank |
| Temperature β | 1.0 |
| Staleness coefficient | 0.3 |

- Tuning: learning rate over {0.0002, 0.0003}; grid over temperature, replay probability, score function (MaxMC, PVL), prioritisation (rank, top-k) and k ([paper](https://arxiv.org/abs/2402.16801) App. G.3).
- JaxUED defaults differ from Table 14 (e.g. `--lr 1e-4`, `--gamma 0.995`), so Table 14 values must be passed explicitly ([JaxUED craftax_plr.py](https://github.com/DramaCow/jaxued/blob/main/examples/craftax/craftax_plr.py)).
- Budget: 1B environment steps ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Hardware, wall-clock, parameters, memory: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | evaluation reward on 20 fixed levels, % of max (226) | not reported as a number; highest UED curve | 10 | [paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7 |

- Ranking reported: PLR > DR, then Robust PLR, ACCEL (RSwap) and ACCEL (Noise) roughly equal, ACCEL (Swap) worst ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Curated levels tend to place resources closer to the spawn point ([paper](https://arxiv.org/abs/2402.16801) App. G.2).

- Third-party rerun (2026-10-20 pass): [[PACE (Yuan 2026)]] reruns PLR with a JaxUED PPO student on Craftax-Symbolic for 1.07B steps and evaluates the final checkpoints on 20 held-out levels: mean episodic reward 25.380 (sd across levels 5.233), i.e. 11.2% of 226 over 10 seeds; IQM of min-max normalised reward 0.484 [0.372, 0.615] ([PACE paper](https://arxiv.org/abs/2605.01358) Tables 4, 5). Same kind of protocol as the table above but different levels, hyperparameters and budget; in that rerun DR ranks above PLR, unlike the Craftax paper.

## Problems and limitations

- Evaluation is on in-distribution levels, which the authors note favours DR over UED ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Curriculum-induced covariate shift: replay levels differ from the normal generator ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Metric is not the training-episode return used by the other Craftax baselines, so it is not directly comparable to the scoreboard numbers.
- Episode cap: App. G.1 says 4096 "instead of the full 10000", while App. D gives the maximum episode length as 100,000; the paper is inconsistent.

## Code

- Repository: https://github.com/DramaCow/jaxued, `examples/craftax/craftax_plr.py` (DR, PLR, ACCEL on Craftax-Symbolic); mutators in `examples/craftax/mutators.py`.
- Framework: JAX.
- Licence: Apache-2.0 (declared in `pyproject.toml`; the repository has no LICENSE file and GitHub reports none).
- Citation: Matthews et al., "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning", ICML 2024, arXiv:2402.16801; Coward, Beukman, Foerster, "JaxUED: A simple and useable UED library in Jax", arXiv:2403.13091; method: Jiang et al., "Prioritized Level Replay", ICML 2021.
