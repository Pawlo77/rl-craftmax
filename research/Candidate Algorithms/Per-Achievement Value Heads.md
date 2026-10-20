---
title: "Per-Achievement Value Heads"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - value-decomposition
  - credit-assignment
  - critic
origin: "van Seijen, Fatemi, Romoff, Laroche, Barnes, Tsang. Hybrid Reward Architecture for Reinforcement Learning. NeurIPS 2017. https://arxiv.org/abs/1706.04208"
created: 2026-10-20
updated: 2026-10-20
---
# Per-Achievement Value Heads

## Idea in plain words

- The Craftax reward is a sum of small parts: one part per achievement, plus a health part.
- Instead of one critic that predicts the total, train one small value head per part (or per tier of parts).
- Each head answers an easier question, for example "will I get the iron pickaxe in this episode?".
- Compute an advantage for each head and add them up for the PPO policy update.
- This is the Hybrid Reward Architecture (HRA) idea, used here inside PPO.

## Why it fits Craftax

- The reward really is a sum: 67 one-time achievement rewards (weights 1, 3, 5, 8) plus 0.1 per health point gained or lost ([[Craftax scoring and evaluation]], [[Craftax achievements]]).
- One critic must mix very different signals: frequent small health changes from combat and hunger ([[Craftax survival stats]]) and rare big achievement rewards. Health noise can drown the rare signal.
- Once an achievement is unlocked, its future reward in that episode is exactly zero. A dedicated head can learn this quickly, if it knows which achievements are done.
- Per-head values are readable: you can plot which achievements the agent expects to get, useful for debugging and for a report.

## How it works

Builds on value-function approximation ([[09a Function approximation and linear methods]]), n-step and λ-returns ([[07a n-step TD and TD(λ)]]) and the actor-critic ([[11 Policy approximation (actor-critic)]]).

1. Split the reward: $r_t = \sum_{k=1}^{K} r^{(k)}_t$. Simple split: $K = 5$ (Basic, Intermediate, Advanced, Very Advanced, health). Full split: $K = 68$.
2. Critic outputs $V^{(1)}(s), \dots, V^{(K)}(s)$ from a shared trunk.
3. For each head, compute GAE with its own reward $r^{(k)}$.
4. Policy advantage: $A_t = \sum_k A^{(k)}_t$, then normalise as usual.
5. Value loss: sum of squared errors over heads.
6. Optional: append the 67-bit "already unlocked this episode" vector to the observation, so heads can drop to zero after an unlock.

## How to implement here

- Base: `ppo_rnn.py` ([[Craftax Baselines]]).
- Reward split: from the change of `env_state.achievements` between steps, times the tier weights; health part = native reward minus achievement part. About 30 lines in a wrapper.
- Critic head: `Dense(K)` instead of `Dense(1)`; GAE via `jax.vmap` over the head axis. About 40-60 lines.
- Extra hyperparameters: $K$ (5 or 68), whether to add the unlocked-achievements vector, value-loss weight per head (start equal).

## Risks

- Runtime: a slightly wider output layer; under 5% slower.
- Risk: the 68-head version may learn slowly for heads that never see reward; start with 5 tier heads.
- Risk: adding the unlocked-achievements vector changes the observation; report it as a separate variant, since the baselines do not see it.
- Risk: summed advantages have higher variance; keep advantage normalisation.

## Novelty check

- Searched: web "Craftax hybrid reward architecture", "per-achievement value heads", Semantic Scholar citations of arXiv 2402.16801, GitHub repositories named "craftax".
- No reward-decomposed critic on Craftax found.
- Closest: LEO outputs values for every goal at once in goal-conditioned Craftax (CraftaxGC), but trains goal-reaching policies, not the native reward ([arXiv 2605.23551](https://arxiv.org/abs/2605.23551)).
- Anchor-and-Harvest Curriculum conditions its policy on a 67-dimensional achievement indicator, but keeps one value head ([arXiv 2610.15328](https://arxiv.org/abs/2610.15328)).
- New here: HRA-style decomposition of the native Craftax reward inside PPO.

## How to evaluate

- Reward % of max on Craftax-1M and Craftax-1B against blocks D and A of the [[Existing Implementations]] table.
- Explained variance of the value function, total and per head, against plain PPO-RNN.
- Achievement success rates for Intermediate and Advanced tiers, where the rare signal matters most.
- Ablations: 5 heads versus 68 heads; with and without the unlocked-achievements vector.

## References

- van Seijen, H. et al. "Hybrid Reward Architecture for Reinforcement Learning." NeurIPS 2017. https://arxiv.org/abs/1706.04208
- "Goal-Conditioned Agents that Learn Everything All at Once." 2026. https://arxiv.org/abs/2605.23551
- Matthews, M. et al. "Craftax." ICML 2024. https://arxiv.org/abs/2402.16801
