---
title: "REINFORCE with Baseline"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - policy-gradient
  - monte-carlo
origin: "Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8, 1992"
created: 2026-10-20
updated: 2026-10-20
---
# REINFORCE with Baseline

## Idea in plain words

- Play a whole episode, then push up the log-probability of every action in proportion to the return that followed it, minus a baseline $V(s)$.
- Monte Carlo policy gradient: no bootstrapping, no critic in the target.

## Verdict

- Rejected: full-episode returns over up to 100,000 steps ([[Craftax episode and time]]) give policy gradients with extreme variance, and updates wait for episode ends.
- The book's fix for this is the actor-critic (bootstrapped target), which PPO and [[Actor-Critic with Eligibility Traces]] already are.
- Already used on Craftax where episodes are short: [[Simulus (Cohen 2025)]] trains its controller with REINFORCE and a value baseline on short imagined rollouts.

## Why it fits Craftax

- It does not: the benchmark's long episodes are the worst case for Monte Carlo policy gradient.

## How it works

From [[11 Policy approximation (actor-critic)]] (actor-critic as the bootstrapped version) and [[05a Monte Carlo prediction and control]] (Monte Carlo returns).

1. Play an episode with $\pi_\theta$.
2. $G_t$ by a backward pass; advantage $G_t - V(s_t)$.
3. Gradient step on $-\log\pi_\theta(a_t \mid s_t)(G_t - V(s_t))$ and on the value loss.

## How to implement here

- PPO code with GAE $\lambda = 1$, full-episode rollouts, one epoch, no clipping.

## Risks

- Very slow and noisy learning; memory for whole episodes.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax REINFORCE".
- Only inside Simulus's imagination (short rollouts); not as a stand-alone agent on Craftax.

## How to evaluate

- Not planned.

## References

- Williams, R. J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning 8, 1992.
