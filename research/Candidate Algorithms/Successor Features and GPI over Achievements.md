---
title: "Successor Features and GPI over Achievements"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - successor-features
  - transfer
origin: "Barreto et al. Successor Features for Transfer in Reinforcement Learning. NeurIPS 2017. https://arxiv.org/abs/1606.05312"
created: 2026-10-20
updated: 2026-10-20
---
# Successor Features and GPI over Achievements

## Idea in plain words

- Write the reward as a weighted sum of features: $r = \phi(s, a, s')^\top w$. In Craftax, $\phi$ can be "which achievement was just unlocked" and $w$ the tier rewards.
- Successor features $\psi(s, a)$: the expected discounted sum of future $\phi$, one number per achievement. Then $Q(s, a) = \psi(s, a)^\top w$.
- GPI (generalized policy improvement): keep several policies, each good at one achievement, and act with the one whose $\psi^\top w$ is highest in the current state.

## Verdict

- Rejected: not in the textbook, and on Craftax it splits into two pieces that are already covered.
- With Craftax's reward (a sum of one-time achievement indicators), successor features are per-achievement value heads, already [[Per-Achievement Value Heads]].
- GPI over one policy per achievement is a hierarchy of achievement skills, covered by [[SMDP Q-Learning over Hand-Defined Skills]] in book-level form.
- Goal-conditioned "all goals at once" value learning on Craftax is done by [[Multitask Preplay (Carvalho 2025)]] and LEO ([arXiv 2605.23551](https://arxiv.org/abs/2605.23551)).

## Why it fits Craftax

- The reward really is linear in achievement indicators, the textbook case for successor features.

## How it works

From value functions and optimality in [[03b MDPs, value functions, optimality]] (the decomposition) and TD learning of each $\psi$ component ([[06 Temporal-difference learning]]).

1. Learn $\psi^{\pi_j}(s, a)$ for each policy $\pi_j$ by TD on $\phi$.
2. Act: $a = \arg\max_a \max_j \psi^{\pi_j}(s, a)^\top w$.

## How to implement here

- 67 output heads per policy and one policy per achievement (or tier); large and slow to get right.

## Risks

- Many policies to train; GPI is only as good as its best policy at each state.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers; one skill-learning analysis mentions successor features, no Craftax agent), web "Craftax successor features generalized policy improvement".
- No successor-feature agent on Craftax found; the neighbouring ideas above are done or planned.

## How to evaluate

- Not planned.

## References

- Barreto, A. et al. "Successor Features for Transfer in Reinforcement Learning." NeurIPS 2017. https://arxiv.org/abs/1606.05312
