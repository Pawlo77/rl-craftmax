---
title: "DLSD (Rajaraman 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Arvind Rajaraman
  - Vivek Myers
  - Anca Dragan
year: 2025
venue: "preprint, \"Under review\" (PDF dated 2025-11-04; no arXiv id found)"
paper: "https://www.arvindrajaraman.com/data/dlsd.pdf"
code: "https://github.com/arvindrajaraman/language-skills-diayn"
envs:
  - Craftax-Classic-Symbolic-v1
tags:
  - implementation
  - craftax-classic
  - skill-discovery
  - unsupervised-rl
  - dqn
  - language
created: 2026-10-20
---
# DLSD (Rajaraman 2025)

## TL;DR

- "Discovering Skills with Language": DIAYN-style skill discovery where the discriminator reads sentence-embedding captions of observations instead of raw observations, run on Craftax-Classic's symbolic observation ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) §3-4, App. C).
- Craftax-Classic, 2M steps, episodes capped at 1K steps: Crafter score 2.27 without reward (DIAYN 1.55, APT 1.71, RND 1.65) and 3.72 with a hybrid reward (DQN on the reward 3.93); 5 seeds ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) Table 1).
- Very short budget and a 1K-step cap: not comparable with any block of [[Existing Implementations]].

## Method

- Skill discovery (DIAYN) with 5 skills; the discriminator predicts the skill from a language embedding (sentence-transformers/all-mpnet-base-v2) of a caption of the observation ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) Algorithm 1, App. B-C).
- Policy learning: DQN with a Polyak-averaged target network (τ 0.001), ε-greedy exploration, warm-up and a replay buffer ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) Algorithm 1, App. C).
- Hybrid variant: R_h = R_gt + λ (log D(z | o') − log p(z)) with λ = 0.2 ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) §4, App. C).
- Network: two conv layers (32, 64 channels, 3 x 3) on the 21 one-hot grids, then dense 768-768 layers to 17 actions (App. C).
- Observation: the symbolic Craftax-Classic vector of size 1345 (App. C).

## Training setup and cost

| Hyperparameter | Value ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) App. C) |
|---|---|
| Steps | 2M (including 10K warm-up) |
| Maximum episode length | 1K steps |
| γ | 0.99 |
| ε | 1 → 0.1 by 75% of training, then 0.1 |
| Learning rate (policy, discriminator) | 1e-3 |
| Update frequency | every 20 environment steps |
| Replay buffer / batch | 50K / 512 |
| Parallel environments | 100 |
| Skills | 5 |

- Hardware: NVIDIA RTX A6000 GPUs (a cluster of 8); about 5-10 hours per seed to train and evaluate ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) App. B, C).
- Parameters: not reported.

## Results

| Method | Reward used | Crafter score (geometric mean of 22 success rates) | Seeds | Source |
|---|---|---|---|---|
| DLSD | none | 2.27 | 5 | [paper](https://www.arvindrajaraman.com/data/dlsd.pdf) Table 1 |
| DIAYN | none | 1.55 | 5 | Table 1 |
| APT | none | 1.71 | 5 | Table 1 |
| RND (reward-free) | none | 1.65 | 5 | Table 1 |
| DLSD hybrid | ground truth + skill reward | 3.72 | 5 | Table 1 |
| DQN | ground truth | 3.93 | 5 | Table 1 |

- Score formula: exp(mean over 22 of ln(100 A_i + 1)) − 1 ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) Eq. 1); achievement rates are the percentage of training episodes with the achievement (§5); no spread given.
- Per-achievement rates for all methods: Table 1; no method collects iron, diamonds or eats a plant reliably.
- The paper calls the environment "Crafter" throughout and uses "a symbolic representation of Crafter (provided by the Craftax implementation)" (App. C).

## Problems and limitations

- Long training time and reliance on a large language model, which slows discriminator learning ([paper](https://www.arvindrajaraman.com/data/dlsd.pdf) §6, "Limitations and Societal Impact").
- 1K-step episode cap and 2M steps make the absolute numbers tiny next to standard Craftax-Classic results.
- No spread, no venue yet.

## Code

- Repository: https://github.com/arvindrajaraman/language-skills-diayn (`diayn.py`); commit read in this pass: `5967778` (2025-11-15); no issues.
- Framework: JAX.
- Licence: none in the repository.
