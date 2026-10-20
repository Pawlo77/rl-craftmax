---
title: "PACE (Yuan 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Fang Yuan
  - Quanjun Yin
  - Siqi Shen
  - Yuxiang Xie
  - Junqiang Yang
  - Long Qin
  - Junjie Zeng
  - Qinglun Li
year: 2026
venue: "ICML 2026"
paper: "https://arxiv.org/abs/2605.01358"
code: "not found (built on JaxUED, https://github.com/DramaCow/jaxued)"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - unsupervised-environment-design
  - curriculum
  - ppo
created: 2026-10-20
---
# PACE (Yuan 2026)

## TL;DR

- Parameter Change for UED: a PLR-style level replay whose level score is the squared parameter change, (1/α)·‖θ_new − θ_old‖², after a provisional PPO update on that level, as a first-order proxy for policy improvement; no regret estimate and no extra rollouts ([paper](https://arxiv.org/abs/2605.01358) §3, Eq. 6).
- Craftax (Symbolic), 1.07B steps, 20 held-out levels: mean episodic reward 31.198 (13.8% of 226) against Domain Randomisation 27.723, PLR 25.380, Robust PLR 23.780 and ACCEL 22.661; 10 seeds ([paper](https://arxiv.org/abs/2605.01358) Table 4).
- Same kind of protocol as block B of [[Existing Implementations]] (UED, 20 levels), and the first source in this folder with numbers for the UED baselines.

## Method

- Level score: squared norm of the parameter change after a provisional PPO update on the level, divided by the learning rate ([paper](https://arxiv.org/abs/2605.01358) §3.1, Eq. 6).
- Level buffer of size K with rank-based sampling (temperature β) and staleness, as in PLR ([paper](https://arxiv.org/abs/2605.01358) §3.2).
- Built on JaxUED with a PPO student ([paper](https://arxiv.org/abs/2605.01358) §4); student architecture not stated.

## Training setup and cost

| Hyperparameter | Value ([paper](https://arxiv.org/abs/2605.01358) Table 6) |
|---|---|
| Steps | 1.07B |
| Learning rate | 3e-4 (2e-4 for DR) |
| Environments | 1024 |
| Rollout | 64 + 64 |
| Minibatches | 2 |
| γ / GAE λ | 0.995 / 0.90 |
| Entropy coefficient | 0.01 |
| Buffer size / replay probability / staleness / β | 1000 / 0.7 / 0.6 / 0.8 |

- Hardware: one A100 (80 GB) for Craftax ([paper](https://arxiv.org/abs/2605.01358) §4).
- Wall-clock and parameters: not reported.

## Results

Craftax-Symbolic, 1.07B steps, final checkpoints, 10 training seeds ([paper](https://arxiv.org/abs/2605.01358) Tables 4, 5, App. C.2).

| Method | Mean reward over 20 levels (raw) ± sd across levels | % of 226 | IQM of min-max normalised reward [95% CI] | Optimality gap [95% CI] |
|---|---|---|---|---|
| PACE | 31.198 ± 2.439 | 13.8 | 0.750 [0.706, 0.788] | 0.268 [0.232, 0.305] |
| Domain Randomisation | 27.723 ± 3.839 | 12.3 | 0.603 [0.530, 0.675] | 0.407 [0.343, 0.470] |
| PLR | 25.380 ± 5.233 | 11.2 | 0.484 [0.372, 0.615] | 0.495 [0.409, 0.580] |
| Robust PLR (PLR⟂) | 23.780 ± 4.688 | 10.5 | 0.409 [0.320, 0.520] | 0.556 [0.479, 0.633] |
| ACCEL | 22.661 ± 4.005 | 10.0 | 0.364 [0.283, 0.458] | 0.599 [0.534, 0.664] |

- Percentages computed in this pass. Per-level entries are mean ± sd over 10 training seeds; the Mean row's ± is the sd across levels (Table 4 caption).
- Baseline rows are third-party reruns of [[Domain Randomisation (Matthews 2024)]], [[PLR (Matthews 2024)]], [[Robust PLR (Matthews 2024)]] and [[ACCEL (Matthews 2024)]] under the same PPO student and levels; added to those notes. Note the ranking differs from the Craftax paper's (here DR is above PLR).
- Training curve: Fig. 6 (shaded s.e.), figure only.

## Problems and limitations

- The main text (§1, §4.2, Fig. 8) reports IQM 0.694 [0.532, 0.850] and optimality gap 0.341, which contradicts Table 5.
- Hyperparameters were selected on the same 20 levels used for evaluation (App. C.3).
- Reward only, no achievement breakdown; not stated: student architecture, number of evaluation episodes, greedy or sampled actions.
- SFL, PAIRED and other recent UED methods not compared; no limitations section.
- The OpenReview rebuttal mentions 5 more Craftax seeds in anonymous tables (anonymous.4open.science/r/Table4-AEAC/), not read in this pass.

## Code

- Not found; the paper says it builds on JaxUED (https://github.com/DramaCow/jaxued, Apache-2.0).
