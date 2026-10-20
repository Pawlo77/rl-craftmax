---
title: "Linear Sarsa(λ) with Tile Coding"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - linear-function-approximation
  - tile-coding
origin: "Sutton & Barto, Reinforcement Learning: An Introduction, Ch. 9 (coarse coding, tile coding, linear Sarsa(λ))"
created: 2026-10-20
updated: 2026-10-20
---
# Linear Sarsa(λ) with Tile Coding

## Idea in plain words

- Represent the state by many binary features: overlapping tiles over a few chosen variables (tile coding).
- $Q(s, a)$ is a weighted sum of the active features; learn the weights with Sarsa($\lambda$) and traces.
- Linear methods are fast, stable and well understood.

## Verdict

- Rejected: a linear function of the observation cannot express the core Craftax skills.
- "Walk to the nearest tree and hit it" depends on where the tree is relative to the player in a 9 x 11 x 83 view ([[Craftax observations and actions]]); its value is a nonlinear function of thousands of inputs, and tile coding grows exponentially with the number of tiled variables.
- Making it work needs hand-written features ("distance to nearest tree"), which turns the agent into a scripted one.
- The part that works, a table over a small progress key, is used as the top level of [[SMDP Q-Learning over Hand-Defined Skills]].

## Why it fits Craftax

- Only for the low-dimensional parts: floor, tools, inventory and survival stats.

## How it works

From [[09a Function approximation and linear methods]] and [[09b Control with approximation, bootstrapping]].

1. Features $x(s, a)$ from tilings.
2. $\delta = r + \gamma w^\top x(s', a') - w^\top x(s, a)$; trace $z \leftarrow \gamma\lambda z + x(s, a)$; $w \mathrel{+}= \alpha\delta z$.

## How to implement here

- A few dozen lines in JAX; the feature design is the whole difficulty.

## Risks

- Near-random play on everything that needs navigation.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax tile coding linear".
- No linear agent on Craftax found; none expected to be competitive.

## How to evaluate

- Not planned.

## References

- Sutton, R. S., Barto, A. G. Reinforcement Learning: An Introduction, Chapter 9.
