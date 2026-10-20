---
title: "Moon PPO Baseline (Moon 2023)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Seungyong Moon
  - Junyoung Yeom
  - Bumsoo Park
  - Hyun Oh Song
year: 2023
venue: "NeurIPS 2023"
paper: "https://arxiv.org/abs/2307.03486"
code: "https://github.com/snu-mllab/Achievement-Distillation"
envs:
  - Craftax-Classic-Pixels-v1
tags:
  - implementation
  - craftax-classic
  - ppo
  - model-free
created: 2026-10-20
---
# Moon PPO Baseline (Moon 2023)

## TL;DR

- The PPO baseline of "Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning" (the paper's main method, achievement distillation, uses knowledge of Crafter's achievement hierarchy) ([paper](https://arxiv.org/abs/2307.03486) abstract).
- Moon et al. report only the original Crafter. On Craftax-Classic, the only numbers are Dedieu et al.'s reimplementation: reward 47.40 ± 0.58%, score 10.71 ± 0.29%, 4.0M parameters, 26 min for 1M steps (Pixels) ([Dedieu paper v3](https://arxiv.org/abs/2502.01591v3) Table 1).
- It is the starting point of the model-free agent in [[Efficient MBRL (Dedieu 2025)]]; comparable with block F of [[Existing Implementations]].

## Method

- Stateless CNN policy (a modified Impala ResNet) without frame stacking, trained with PPO, with an entropy bonus for exploration ([Dedieu paper](https://arxiv.org/abs/2502.01591) §3.1).
- Dedieu et al. use this baseline instead of achievement distillation because the latter "uses external knowledge about the achievement hierarchy of Crafter" ([Dedieu paper](https://arxiv.org/abs/2502.01591) §3.1, footnote 3).

## Training setup and cost

- Parameters: 4.0M ([Dedieu paper](https://arxiv.org/abs/2502.01591) Table 1).
- Wall-clock: 26 min for 1M environment steps in Dedieu's implementation; hardware for this row not stated ([Dedieu paper](https://arxiv.org/abs/2502.01591) Table 1).
- Hyperparameters of the reimplementation: not listed separately in the Dedieu paper; Moon et al.'s Crafter settings are in their paper and repository (not re-read in this pass).

## Results

| Source | Benchmark | Reward % (mean ± s.e.) | Score % (mean ± s.e.) | Seeds | Location |
|---|---|---|---|---|---|
| Dedieu et al. reimplementation | Craftax-Classic 1M, Pixels | 47.40 ± 0.58 | 10.71 ± 0.29 | not stated for this row (paper: 10 seeds) | [Dedieu paper v3](https://arxiv.org/abs/2502.01591v3) Table 1 |
| Moon et al. (quoted) | original Crafter 1M | 46.91 ± 2.41 (Crafter, not Craftax) | 15.60 ± 1.66 | | Dedieu Table 1, marked * |

- Making this model larger without an RNN, or using a smaller model, lowers performance in Dedieu's ablation ([Dedieu paper](https://arxiv.org/abs/2502.01591) §4.3, Table 4).

## Problems and limitations

- No Craftax result from the original authors; the Craftax-Classic score (10.71%) is well below their Crafter score (15.60%), and Dedieu et al. do not explain the gap.
- Stateless policy: no memory, which Dedieu et al. add to reach 55.49% ([Dedieu paper](https://arxiv.org/abs/2502.01591) §3.2).

## Code

- Original (Crafter): https://github.com/snu-mllab/Achievement-Distillation (PyTorch, MIT); no Craftax code.
- Dedieu et al.'s reimplementation: not released (see [[Efficient MBRL (Dedieu 2025)]]).
