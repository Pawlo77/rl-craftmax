---
title: "RePPO (Nishimori 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Soichiro Nishimori
  - Paavo Parmas
  - Sotetsu Koyamada
  - Tadashi Kozuno
  - Toshinori Kitamura
  - Shin Ishii
  - Yutaka Matsuo
year: 2026
venue: "ICML 2026"
paper: "https://arxiv.org/abs/2606.00151"
code: "https://github.com/nissymori/remax-rl"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - exploration
  - model-free
created: 2026-10-20
---
# RePPO (Nishimori 2026)

## TL;DR

- PPO for the ReMax objective (expected maximum return over M retries): a Q-critic and an expected-improvement advantage computed in closed form by sorting Q-values; exploration emerges without entropy or intrinsic bonuses ([paper](https://arxiv.org/abs/2606.00151) §3, §4).
- Craftax-1B (Symbolic, feed-forward): 11.87% (sd 0.16) of max reward with m = 1.2, against 11.66% (0.33) for PPO with entropy and 11.68% (0.30) for RND with entropy, 5 seeds each ([paper](https://arxiv.org/abs/2606.00151) §5.2, Table 1).
- Same code base and hyperparameters as [[PPO (Matthews 2024)]], but evaluated on 100 episodes per seed, so close to, not identical with, block A of [[Existing Implementations]].

## Method

- ReMax: maximise the expected maximum of M sampled returns, which accounts for return uncertainty ([paper](https://arxiv.org/abs/2606.00151) §3).
- Policy gradient via expected improvement (EI), computed in closed form from sorted Q-values; M relaxed to a continuous m > 0 ([paper](https://arxiv.org/abs/2606.00151) §4.1-4.2).
- RePPO: PPO with a Q-critic and the EI-based advantage; Q(s, a_taken) is replaced by the observed return when computing EI ([paper](https://arxiv.org/abs/2606.00151) §4.3).
- No entropy or intrinsic bonus in RePPO.
- Network: feed-forward `ActorCriticConvSymbolicCraftax` (`agents/craftax/network.py`), no recurrence.

## Training setup and cost

| Hyperparameter | Value ([paper](https://arxiv.org/abs/2606.00151) Table 7) |
|---|---|
| Steps | 1e9 |
| Environments / rollout length | 1024 / 64 |
| Epochs / minibatch size | 4 / 8192 |
| Learning rate | 2e-4 |
| γ / GAE λ | 0.99 / 0.8 |
| Clip ε | 0.2 |
| Entropy coefficient | 0.0 or 0.01 (baselines) |
| Value coefficient / max gradient norm | 0.5 / 1.0 |
| m | 1.2 or 1.4 |
| RND learning rate / bonus | 3e-4 / 1.0 |

- Hardware and wall-clock: not reported; App. H (Fig. 15b) says RePPO runs at a speed comparable to PPO and faster than PPO + RND (figure only).
- Parameters: not reported.

## Results

| Method | Craftax-1B, Symbolic: % of max reward (sd across seeds) | Seeds | Source |
|---|---|---|---|
| RePPO (m = 1.2) | 11.87 (0.16) | 5 | [paper](https://arxiv.org/abs/2606.00151) Table 1 |
| RePPO (m = 1.4) | 10.79 (0.19) | 5 | Table 1 |
| PPO-V (no entropy) | 9.31 (1.00) | 5 | Table 1 |
| PPO-V + entropy 0.01 | 11.66 (0.33) | 5 | Table 1 |
| RND (no entropy) | 9.62 (1.54) | 5 | Table 1 |
| RND + entropy | 11.68 (0.30) | 5 | Table 1 |
| PPO-Q (no entropy) | 9.31 (0.93) | 5 | Table 1 |
| PPO-Q + entropy | 11.85 (0.26) | 5 | Table 1 |

- Evaluation: 100 episodes per seed ([paper](https://arxiv.org/abs/2606.00151) §5.2); greedy or sampled actions not stated for Craftax.
- PPO-V + entropy and RND + entropy are third-party reruns of [[PPO (Matthews 2024)]] and [[RND (Matthews 2024)]] with the Craftax_Baselines settings; added to those notes.

## Problems and limitations

- Needs the full Q-vector and sorting, so only discrete, small action spaces; no regret theory for deep RL; in deep RL the behaviour resembles stochastic, entropy-like exploration rather than deep exploration ([paper](https://arxiv.org/abs/2606.00151) §6.1).
- `sh/run_craftax.sh` loops seeds 0-2 (the paper uses 5), calls `ppo_v_rnd.py`, which is not in `agents/craftax/`, and its `for seed` loop lacks `do`.

## Code

- Repository: https://github.com/nissymori/remax-rl (Craftax: `agents/craftax/reppo.py`, `ppo_v.py`, `ppo_q.py`); commit read in this pass: `1d9c319` (2026-10-17); no issues.
- Framework: JAX, single-file agents.
- Licence: none in the repository.
