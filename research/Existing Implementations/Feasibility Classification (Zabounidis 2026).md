---
title: "Feasibility Classification (Zabounidis 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Renos Zabounidis
  - Roy Siegelmann
  - Mohamad Qadri
  - Woojun Kim
  - Simon Stepputtis
  - Katia P. Sycara
year: 2026
venue: "Reinforcement Learning Journal vol. 7 / RLC 2026"
paper: "https://arxiv.org/abs/2603.09090"
code: "not found (paper names praxis/action_masking/craftax/masks.py; no public repository)"
envs:
  - Craftax-Symbolic-v1
  - Craftax-Classic-Symbolic-v1
tags:
  - implementation
  - craftax
  - craftax-classic
  - ppo
  - action-masking
  - recurrent
  - transformer
created: 2026-10-20
---
# Feasibility Classification (Zabounidis 2026)

## TL;DR

- Paper "Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms": gradients at visited states suppress valid actions at unvisited states through shared parameters; an auxiliary head that predicts action validity (feasibility classification) fixes the features ([paper](https://arxiv.org/abs/2603.09090) §4, §5).
- Craftax 1B (Symbolic), PPO-Hybrid (Transformer-XL + S5): 48.8 ± 0.7 raw return (21.6% of 226) with ground-truth masks at evaluation, 43.2 ± 1.1 (19.1%) with learned masks; 4 seeds ([paper](https://arxiv.org/abs/2603.09090) Table 1).
- Uses oracle validity labels and masks during training (privileged); the learned-mask column is the deployable one. Evaluation is 10,240 stochastic episodes per seed, unlike the training-episode returns of block A in [[Existing Implementations]].
- Same table gives standard unmasked PPO, PPO-RNN and PPO-Hybrid reruns on both Craftax and Craftax-Classic.

## Method

- Theory: with shared parameters, policy-gradient updates at visited states push down the logits of valid actions at unvisited states, with exponential decay (Theorem 1, [paper](https://arxiv.org/abs/2603.09090) §4).
- Feasibility classification: an auxiliary head trained to predict each action's validity from a ground-truth validity function, with a focal or KL-balanced loss, weight λ = 10 ([paper](https://arxiv.org/abs/2603.09090) §5, Table 5).
- Training conditions (Table 1): C1 masked; C2 unmasked (standard PPO); C3 masked + focal; C4 masked + KL-balanced. Deployment with ground-truth masks (GT) or masks predicted by the head (Pred).
- Architectures: PPO-MLP (3 x 512, tanh), PPO-RNN (GRU 512), PPO-Hybrid (1 Transformer-XL layer + 5 S5 layers; d_model 256, 8 heads, memory 128) ([paper](https://arxiv.org/abs/2603.09090) Table 5).
- Validity functions for all 43 Craftax and 17 Craftax-Classic actions: App. C.

## Training setup and cost

| Hyperparameter | Hybrid | RNN, MLP | Source |
|---|---|---|---|
| Steps | 1e9 | 1e9 | [paper](https://arxiv.org/abs/2603.09090) App. D.3 |
| Environments | 1024 | 1024 | Table 5 |
| Rollout length / batch | 128 / 131,072 | 64 / 65,536 | Table 5 |
| Minibatches / epochs | 8 / 4 | 8 / 4 | Table 5 |
| Learning rate | 2e-4, linear | 2e-4, linear | Table 5 |
| γ / GAE λ / clip ε | 0.999 / 0.8 / 0.2 | 0.99 / 0.8 / 0.2 | Table 5 |
| Entropy coefficient | 0.002 | 0.01 | Table 5 |
| Max gradient norm | 1.0 | 1.0 | Table 5 |
| Classification λ / focal γ | 10 / 2 | 10 / 2 | Table 5 |

- Hardware: "a single accelerator" (type not given); wall-clock and parameter counts: not reported.

## Results

Raw return (mean ± sd over 4 seeds), 1B steps, 10,240 evaluation episodes per seed, "stochastic policy without exploration bonuses" ([paper](https://arxiv.org/abs/2603.09090) Table 1, §6). Masked + focal rows are in Table 1 too (e.g. Hybrid 45.0 ± 2.3 GT, 21.2 ± 1.7 Pred on Craftax).

| Condition | Craftax, GT masks | Craftax, Pred masks | Classic, GT | Classic, Pred |
|---|---|---|---|---|
| Hybrid, masked + KL | 48.8 ± 0.7 | 43.2 ± 1.1 | 19.0 ± 0.1 | 18.4 ± 0.1 |
| RNN, masked + KL | 38.1 ± 0.1 | 36.4 ± 0.2 | 20.5 ± 0.1 | 20.4 ± 0.1 |
| MLP, masked + KL | 28.7 ± 3.2 | 28.6 ± 3.2 | 19.5 ± 0.0 | 19.3 ± 0.1 |
| Hybrid, masked only | 45.6 ± 1.3 | -0.9 ± 0.0 | 18.9 ± 0.0 | -0.9 ± 0.0 |
| RNN, masked only | 38.3 ± 0.2 | -0.9 ± 0.0 | 20.4 ± 0.0 | -0.9 ± 0.0 |
| MLP, masked only | 30.4 ± 0.2 | -0.9 ± 0.0 | 19.3 ± 0.0 | -0.8 ± 0.0 |

Unmasked baselines (C2, standard interface, Table 1):

| Architecture | Craftax raw return | % of 226 | Classic raw return | % of 22 |
|---|---|---|---|---|
| PPO-Hybrid (TrXL + S5) | 43.9 ± 0.4 | 19.4 | 19.2 ± 0.1 | 87.3 |
| PPO-RNN (GRU 512) | 36.7 ± 0.1 | 16.2 | 19.8 ± 0.0 | 90.0 |
| PPO (MLP) | 26.9 ± 0.5 | 11.9 | 19.2 ± 0.0 | 87.3 |

- Percentages computed in this pass from the raw returns.
- Classic: return 19.8 against 20.5 corresponds to diamonds in about 20% against about 60% of episodes ([paper](https://arxiv.org/abs/2603.09090) §6.3).
- The PPO-RNN and PPO rows are added to [[PPO-RNN (Matthews 2024)]] and [[PPO (Matthews 2024)]] as third-party reruns.

## Problems and limitations

- Theory assumes fixed features; predicted-mask deployment recovers but does not exceed unmasked performance because of the train-deploy distribution shift ([paper](https://arxiv.org/abs/2603.09090) §7).
- GT columns and all masked training use a privileged ground-truth validity function; only the Pred column and C2 use the standard interface.
- 4 seeds; no code release.

## Code

- Not found: the paper names `praxis/action_masking/craftax/masks.py` (App. B.1), but no public repository was found on GitHub or the RLJ page (https://rlj.cs.umass.edu/2026/papers/Paper1.html).
- Same first author: [[SCALAR (Zabounidis 2026)]].
