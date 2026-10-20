---
title: "Average-Reward Differential Q-Learning"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - average-reward
  - value-based
origin: "Schwartz. A Reinforcement Learning Method for Maximizing Undiscounted Rewards (R-learning). ICML 1993; Wan, Naik, Sutton. Learning and Planning in Average-Reward MDPs. ICML 2021. https://arxiv.org/abs/2006.16318"
created: 2026-10-20
updated: 2026-10-20
---
# Average-Reward Differential Q-Learning

## Idea in plain words

- Drop the discount $\gamma$. Measure the policy by its long-run reward per step $\bar R$.
- Learn differential values: how much better than average a state-action pair is, $Q(s, a)$ with target $r - \bar R + \max_{a'} Q(s', a')$.
- Update $\bar R$ from the same TD error (R-learning, differential Q-learning).

## Verdict

- Rejected: it changes the objective in a way Craftax's reward rule turns against the benchmark.
- Average reward needs one continuing task, so episodes would be chained: death leads into a fresh world.
- Each achievement pays only once per episode ([[Craftax achievements]]), so the reward per step is maximised by unlocking the easy achievements, dying, and starting again; dying costs only the health part of the reward (0.1 x health change, [[Craftax scoring and evaluation]]).
- The benchmark scores reward per episode, so the learned policy would optimise the wrong thing.
- The long-horizon motive (discount too short) is easier to test with $\gamma = 0.999$, which helped in the auto-craftax campaign ([[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]]).

## Why it fits Craftax

- Long episodes without a natural discount fit the average-reward view, but only for continuing tasks, which Craftax is not.

## How it works

From R-learning in [[11 Policy approximation (actor-critic)]].

1. $\delta = r - \bar R + \max_{a'} Q(s', a') - Q(s, a)$.
2. $Q(s, a) \mathrel{+}= \alpha\delta$; $\bar R \mathrel{+}= \eta\alpha\delta$.

## How to implement here

- A change of target in purejaxql PQN plus one scalar $\bar R$ in the train state.

## Risks

- The reset-farming behaviour above; plus known instability of $\bar R$ with function approximation.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax average reward".
- No average-reward agent on Craftax found.

## How to evaluate

- Not planned.

## References

- Schwartz, A. "A Reinforcement Learning Method for Maximizing Undiscounted Rewards." ICML 1993.
- Wan, Y., Naik, A., Sutton, R. S. "Learning and Planning in Average-Reward Markov Decision Processes." ICML 2021. https://arxiv.org/abs/2006.16318
