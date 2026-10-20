---
title: "Least-Squares Policy Iteration"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - least-squares
  - linear-function-approximation
origin: "Bradtke, Barto. Linear Least-Squares Algorithms for Temporal Difference Learning (LSTD). Machine Learning 22, 1996; Lagoudakis, Parr. Least-Squares Policy Iteration. JMLR 4, 2003"
created: 2026-10-20
updated: 2026-10-20
---
# Least-Squares Policy Iteration

## Idea in plain words

- LSTD: for a linear value function $w^\top x(s)$, solve for the TD fixed point directly: $w = A^{-1} b$ with $A = \sum x(s)(x(s) - \gamma x(s'))^\top$ and $b = \sum x(s) r$.
- LSPI: do this for $Q$ (LSTDQ), make the policy greedy, repeat (policy iteration).
- No step size, uses every sample fully.

## Verdict

- Rejected: same representation problem as [[Linear Sarsa-Lambda with Tile Coding]] (linear in the observation cannot navigate the map), plus a $d \times d$ matrix that has to be re-solved each iteration.
- On a small progress key with one feature per key, LSPI is tabular policy iteration on the empirical model, which is the value-iteration variant of [[Dyna-Q over a Skill-Level Progress Model]].

## Why it fits Craftax

- Data-efficient, but only with good features, which Craftax's raw observation does not give.

## How it works

From [[09a Function approximation and linear methods]] and policy iteration in [[04 Dynamic programming]].

1. Collect samples $(s, a, r, s')$.
2. LSTDQ: $A = \sum x(s, a)\big(x(s, a) - \gamma x(s', \pi(s'))\big)^\top$, $b = \sum x(s, a) r$, $w = A^{-1} b$.
3. $\pi(s) = \arg\max_a w^\top x(s, a)$; repeat from step 2.

## How to implement here

- `jnp.linalg.solve` on accumulated $A$, $b$; feature design is the difficulty.

## Risks

- Poor features give a poor policy; singular $A$ needs regularisation.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax LSPI least squares".
- No least-squares agent on Craftax found; none expected to be competitive.

## How to evaluate

- Not planned.

## References

- Bradtke, S. J., Barto, A. G. "Linear Least-Squares Algorithms for Temporal Difference Learning." Machine Learning 22, 1996.
- Lagoudakis, M. G., Parr, R. "Least-Squares Policy Iteration." JMLR 4, 2003.
