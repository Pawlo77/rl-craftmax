---
title: "Monte Carlo Control"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - monte-carlo
  - value-based
origin: "Sutton & Barto, Reinforcement Learning: An Introduction, Ch. 5 (on-policy Monte Carlo control with ε-soft policies)"
created: 2026-10-20
updated: 2026-10-20
---
# Monte Carlo Control

## Idea in plain words

- Play whole episodes with an ε-greedy policy, then move $Q(s, a)$ toward the full return that followed each visit.
- No bootstrapping: the target is the real sum of rewards until the episode ends, not a guess from the next state.
- Control: act greedily with respect to the new $Q$ (generalized policy iteration).

## Verdict

- Rejected at the level of single game steps.
- Craftax episodes last up to 100,000 steps and achievements are rare ([[Craftax episode and time]], [[Craftax achievements]]); a full return mixes the luck of thousands of later steps, so its variance is huge, and the first update arrives only after the episode ends.
- Monte Carlo is the $\lambda = 1$ end of TD($\lambda$); [[PQN-RNN (Gallici 2025)]] (Q($\lambda$)) and PPO's GAE already expose $\lambda$ as a knob, so sweeping $\lambda$ toward 1 answers the same question without a new algorithm.
- Kept where it makes sense: as a top-level ablation in [[SMDP Q-Learning over Hand-Defined Skills]] (skill-level episodes are only tens of decisions long) and as the return target of [[Self-Imitation Learning]].

## Why it fits Craftax

- Only partly: returns are exact, so no bias from a wrong critic; but the long, stochastic episodes undo that benefit.

## How it works

From [[05a Monte Carlo prediction and control]].

1. Play an episode with ε-greedy actions on $Q$.
2. For each step, compute $G_t = \sum_{i \ge 0} \gamma^i r_{t+i}$ by a backward pass.
3. Regress $Q(s_t, a_t)$ toward $G_t$ (squared error).

## How to implement here

- One flag in PQN: set $\lambda = 1$ and compute returns over full episodes instead of 128-step rollouts.

## Risks

- Very high variance, updates only at episode end, and storing whole 100,000-step episodes.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), arXiv API `all:craftax`, web "Craftax Monte Carlo control".
- No Monte Carlo control agent on Craftax found; Monte Carlo returns appear only inside other methods (SIL-style buffers, Simulus's REINFORCE baseline in imagination).
- Not new enough to matter: it is a $\lambda$ setting of existing agents.

## How to evaluate

- If run anyway: a $\lambda \in \{0.5, 0.9, 1.0\}$ sweep of PQN-RNN on Craftax-1M (block D of [[Existing Implementations]]) and Craftax-1B (block A, PQN-RNN 16.0%).

## References

- Sutton, R. S., Barto, A. G. Reinforcement Learning: An Introduction, Chapter 5.
