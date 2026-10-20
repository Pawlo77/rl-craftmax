---
title: "Sarsa(λ) with Function Approximation"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - sarsa
  - value-based
origin: "Rummery, Niranjan. On-line Q-learning using connectionist systems. 1994; Sutton & Barto, Reinforcement Learning: An Introduction, Ch. 7 and 9 (Sarsa(λ), linear and nonlinear Sarsa)"
created: 2026-10-20
updated: 2026-10-20
---
# Sarsa(λ) with Function Approximation

## Idea in plain words

- Sarsa learns $Q(s, a)$ for the policy it actually follows: the target uses the next action it really takes, $r + \gamma Q(s', a')$.
- Q-learning uses the best next action instead, $r + \gamma \max_{a'} Q(s', a')$.
- Sarsa($\lambda$) mixes n-step targets with weight $\lambda$; Expected Sarsa averages over the policy's next actions.
- With a neural Q-network and many parallel envs it is an on-policy value-based learner.

## Verdict

- Rejected as a separate method: already covered by the value-based family on Craftax.
- [[PQN-RNN (Gallici 2025)]] is Q($\lambda$) with a neural network, parallel envs and no replay, already at 16.0% on Craftax-1B; Sarsa($\lambda$) changes only the bootstrap target (taken action instead of max), a one-line ablation of PQN, not a new update family.
- The online, backward-view form (traces) is covered by the Q($\lambda$) twin in [[Actor-Critic with Eligibility Traces]].
- Worth one ablation run inside PQN (target: max, taken action, expected), not a project direction.

## Why it fits Craftax

- On-policy targets are safer near danger (lava, mobs): Sarsa learns the value of the ε-greedy policy, so it avoids cliffs that random actions could fall into (the book's cliff-walking example).

## How it works

From [[07b Eligibility-trace control and off-policy]] and [[09b Control with approximation, bootstrapping]].

1. Act ε-greedily on $Q$.
2. Target $G^\lambda_t$ built from $r_t + \gamma Q(s_{t+1}, a_{t+1})$ terms.
3. Regress $Q(s_t, a_t)$ toward $G^\lambda_t$.

## How to implement here

- In purejaxql `pqn_rnn_craftax.py`, replace the max in the λ-return with the value of the stored next action (Sarsa) or the ε-greedy expectation (Expected Sarsa).

## Risks

- None beyond PQN's; the expected gain is small.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax Sarsa", GitHub repository search "craftax sarsa" (no hits).
- No Sarsa agent on Craftax found, but the family (λ-return value learning with a network over parallel envs) is done by PQN-RNN.

## How to evaluate

- As a PQN ablation: Craftax-1B against PQN-RNN's 16.0% (block A of [[Existing Implementations]]), same seeds.

## References

- Rummery, G. A., Niranjan, M. "On-line Q-learning using connectionist systems." Technical report, Cambridge, 1994.
- Gallici, M. et al. "Simplifying Deep Temporal Difference Learning." ICLR 2025. https://arxiv.org/abs/2407.04811
