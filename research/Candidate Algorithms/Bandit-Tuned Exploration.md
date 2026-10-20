---
title: "Bandit-Tuned Exploration"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - exploration
  - bandits
  - intrinsic-reward
origin: "Badia et al. Agent57: Outperforming the Atari Human Benchmark. ICML 2020. https://arxiv.org/abs/2003.13350; Badia et al. Never Give Up. ICLR 2020. https://arxiv.org/abs/2002.06038"
created: 2026-10-20
---
# Bandit-Tuned Exploration

## Idea in plain words

- Exploration bonuses help or hurt depending on one number: the bonus weight β.
- Instead of guessing β, train one policy that knows several values of β (it gets β as an input).
- A multi-armed bandit picks which β each environment uses for its next episode.
- The bandit's payoff is the real game return of that episode. Values of β that lead to better play get picked more often.
- The choice can change during training, for example more exploration early, less later.

## Why it fits Craftax

- On Craftax the bonus weight decides success or failure. The Craftax paper's bonuses lowered reward at their tuned β ([[E3B (Matthews 2024)]], [[RND (Matthews 2024)]]); a later sweep found RND helps only in a narrow β range ([auto-craftax report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.5).
- Different phases of the game need different amounts of exploration: the overworld is learned fast, deeper floors need much more ([[Craftax floors and progression]]).
- Thousands of parallel environments give the bandit many episodes per update ([[Craftax performance]]).

## How it works

Builds on upper-confidence-bound action selection and nonstationary bandits ([[02 Multi-armed bandits]]).

1. Arms: $J$ values of β, for example {0, 0.003, 0.01, 0.03} for an RND or count bonus ([[Count-Based Progress Bonus]]).
2. The policy and value networks get the arm id as an extra one-hot input; the intrinsic value head is trained for every arm.
3. When environment $i$ resets, the bandit picks arm $j$ by sliding-window UCB: $\bar{G}_j + c\sqrt{\ln n / n_j}$, where $\bar{G}_j$ is the mean extrinsic return of the last $W$ episodes with arm $j$.
4. The episode uses reward $r + \beta_j r^+$.
5. When it ends, add its extrinsic return to arm $j$'s window.
6. Evaluate with the arm the bandit prefers at the end, or with β = 0.

## How to implement here

- Base: `ppo_rnd.py` or PPO-RNN plus [[Count-Based Progress Bonus]] ([[Craftax Baselines]]).
- Arm per environment kept in the env state; arm one-hot concatenated to the observation; reward mixed per environment. About 80-120 lines.
- Bandit statistics as small JAX arrays (sum and count per arm over a window), updated from finished episodes in each rollout.
- Extra hyperparameters: arm values, UCB constant $c$, window $W$, whether arms also set γ (Agent57 pairs β with γ).

## Cost and risk

- Compute: almost none beyond the bonus itself. Fits a Mac at Craftax-1M scale.
- Risk: a shared network may not learn very different behaviours for each arm; few arms (3-4) help.
- Risk: the payoff is noisy and delayed (long episodes); short windows chase noise, long windows react slowly.
- Risk: the bandit can settle on β = 0 early, before the bonus has had time to pay off. Force a minimum share per arm.

## Novelty check

- Searched: web "Craftax bandit exploration", "Craftax hyperparameter bandit", Semantic Scholar citations of arXiv 2402.16801, GitHub repositories named "craftax".
- Bandit hyperparameter tuning in RL exists (ULTHO, [arXiv 2503.06101](https://arxiv.org/abs/2503.06101), tested on ALE, Procgen, MiniGrid, PyBullet, not Craftax).
- On Craftax: β was fixed by grid search in every published run.
- No Agent57-style meta-controller on Craftax found.
- New here: a per-environment UCB choice of the bonus weight, with β-conditioned PPO.

## How to evaluate

- Reward % of max on Craftax-1M and Craftax-1B against blocks D and A of the [[Existing Implementations]] table, especially against [[RND (Matthews 2024)]] and [[E3B (Matthews 2024)]].
- Compare with the best fixed β from a small sweep at the same budget.
- Plot arm choice over training, and success rates of floor-entry achievements.

## References

- Badia, A. P. et al. "Agent57: Outperforming the Atari Human Benchmark." ICML 2020. https://arxiv.org/abs/2003.13350
- Badia, A. P. et al. "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020. https://arxiv.org/abs/2002.06038
- Sutton, R. S., Barto, A. G. "Reinforcement Learning: An Introduction", chapter 2 (UCB).
