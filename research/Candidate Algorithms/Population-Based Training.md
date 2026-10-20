---
title: "Population-Based Training"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - hyperparameters
  - population
  - jax
origin: "Jaderberg et al. Population Based Training of Neural Networks. 2017. https://arxiv.org/abs/1711.09846"
suggested-by:
  - "[[PQN-RNN (Gallici 2025)]]"
  - "[[PPO-GTrXL (Hamon 2024)]]"
created: 2026-10-20
---
# Population-Based Training

## Idea in plain words

- Train several agents (a population) at the same time, each with different hyperparameters: learning rate, entropy bonus, discount γ, GAE λ.
- Every so often, compare them on recent episode return.
- The worst agents copy the weights of the best ones (exploit), then change their hyperparameters a little (explore).
- The population finds a good hyperparameter schedule during one training run, instead of many separate runs.

## Why it fits Craftax

- Craftax results depend strongly on hyperparameters. An autonomous tuning campaign raised PPO from 26.83 to 40.50 return at 1B through rollout length (128 steps), γ = 0.999, entropy annealing, a GRU and a CNN encoder ([auto-craftax report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §5).
- The best settings change over training: early the agent needs a large entropy bonus and a short horizon; later, reaching floors needs a long horizon ([[Craftax floors and progression]]). A schedule found by PBT can follow this.
- Craftax runs in JAX, so a population is just one more `vmap` axis on one GPU ([[Craftax Baselines compute and wall-clock]]: `--num_repeats` already vmaps seeds).
- The Craftax-1B baselines were tuned one hyperparameter at a time with one seed ([[PPO (Matthews 2024)]]).

## How it works

Builds on PPO as an actor-critic ([[11 Policy approximation (actor-critic)]]); the exploit/explore loop is the bandit trade-off of [[02 Multi-armed bandits]] applied to whole training runs.

1. Start $N$ members (for example 8) with random hyperparameters from a range.
2. Train all members in parallel for $T$ updates.
3. Rank members by mean return of their last finished episodes.
4. Bottom 25%: copy weights, optimizer state and hyperparameters from a random top-25% member.
5. Perturb the copied hyperparameters: multiply each by 0.8 or 1.2.
6. Repeat until the step budget is used. Report the best member and the hyperparameter schedule.

```
for round in rounds:
    train_all(members, T)                  # vmap over members
    order = argsort(recent_return)
    for loser in bottom_quarter(order):
        winner = random(top_quarter(order))
        loser.params, loser.opt = copy(winner)
        loser.hparams = winner.hparams * choice([0.8, 1.2])
```

## How to implement here

- Base: `ppo_rnn.py`, or PQN-RNN from purejaxql ([[PQN-RNN (Gallici 2025)]]).
- Turn hyperparameters from Python constants into traced arrays (learning rate via `optax.inject_hyperparams`, entropy coefficient, γ, λ as per-member values).
- `vmap` the update over members; do the exploit/explore step between jitted chunks. About 100-150 lines.
- Extra hyperparameters: population size $N$, interval $T$, perturbation factors, ranking window.
- Fair budget: the benchmark counts all environment steps, so with $N$ members each member gets $1/N$ of the steps. Report both "per member" and "total" steps.

## Cost and risk

- Compute: $N$ times the cost of one run at the same per-member steps. On a Mac: Craftax-Classic or Craftax-1M scale with 4-8 members of 64-128 environments. Craftax-1B needs a GPU.
- Risk: greedy selection. Craftax rewards short-term overworld play before deep play; members that explore deeper may look worse early and be killed. Use a long interval $T$.
- Risk: noisy ranking from few finished episodes; average over a window.
- Risk: under an equal total budget, each member trains less; PBT may lose on the 1B challenge but win at equal per-member steps.

## Novelty check

- Searched: web "population based training Craftax", Semantic Scholar citations of arXiv 2402.16801, GitHub repositories named "craftax".
- Done on Craftax: evolution strategies (OpenES and EGGROLL on Craftax-Classic and Craftax-Symbolic, [arXiv 2511.16652](https://arxiv.org/abs/2511.16652)); grid search (auto-craftax); learned optimizers on Craftax-Classic ([arXiv 2407.07082](https://arxiv.org/abs/2407.07082)).
- No PBT result on Craftax found.
- New here: PBT schedules for PPO-RNN or PQN-RNN on Craftax, with the step-budget accounting stated.

## Suggested by existing work

- purejaxql README, Craftax section: PQN is "an intriguing starting point for population-based training in Craftax" ([[PQN-RNN (Gallici 2025)]]).
- transformerXL_PPO_JAX README, Results on Craftax: the authors note they did little hyperparameter tuning ([[PPO-GTrXL (Hamon 2024)]]).

## How to evaluate

- Best member's reward % of max against blocks A and D of the [[Existing Implementations]] table, with the budget rule stated.
- Compare with the default PPO-RNN hyperparameters at the same per-member steps.
- Plot the learned schedules (entropy coefficient, γ, learning rate over time); they are a result in themselves.

## References

- Jaderberg, M. et al. "Population Based Training of Neural Networks." 2017. https://arxiv.org/abs/1711.09846
- Gallici, M. et al. "Simplifying Deep Temporal Difference Learning." ICLR 2025. https://arxiv.org/abs/2407.04811; code https://github.com/mttga/purejaxql
- "Evolution Strategies at the Hyperscale." 2025. https://arxiv.org/abs/2511.16652
- auto-craftax research report. https://github.com/roger-creus/auto-craftax
