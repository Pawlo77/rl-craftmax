---
title: "Distributional Q-Learning"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - distributional-rl
  - value-based
origin: "Bellemare, Dabney, Munos. A Distributional Perspective on Reinforcement Learning (C51). ICML 2017. https://arxiv.org/abs/1707.06887; Dabney et al. Distributional RL with Quantile Regression. AAAI 2018. https://arxiv.org/abs/1710.10044"
created: 2026-10-20
updated: 2026-10-20
---
# Distributional Q-Learning

## Idea in plain words

- Instead of one number $Q(s, a)$ (the average return), learn the whole spread of possible returns: a histogram (C51) or a set of quantiles (QR-DQN).
- Act greedily on the mean of that distribution.
- The extra targets often make the network learn better features.

## Verdict

- Rejected: not in the textbook (the book learns expected values only), and the Craftax-adjacent versions exist.
- Exploration via Distributional Ensemble (QR-DQN ensembles with uncertainty-driven exploration, Jiang et al.) was state of the art on Crafter, which has Craftax-Classic's rules ([arXiv 2306.05483](https://arxiv.org/abs/2306.05483)).
- On Craftax, [[Simulus (Cohen 2025)]] already predicts values and rewards as distributions (HL-Gauss classification over 128 bins).
- On top of PQN it is a critic-loss change, not a new learning family.

## Why it fits Craftax

- Returns are multi-modal (die early or survive long), which a distribution can represent.

## How it works

1. Network outputs $N$ quantiles per action.
2. Target quantiles $r + \gamma\,\theta_j(s', a^*)$; quantile-regression (Huber) loss.

## How to implement here

- Quantile head and loss in purejaxql PQN.

## Risks

- More hyperparameters (number of quantiles, Huber $\kappa$); interaction with PQN's λ-returns is not standard.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax distributional quantile", arXiv API `crafter AND distributional`.
- Done nearby: distributional ensembles on Crafter (Jiang et al.); HL-Gauss values in Simulus on Craftax; no distributional PQN on Craftax found.

## How to evaluate

- Not planned.

## References

- Bellemare, M. G., Dabney, W., Munos, R. "A Distributional Perspective on Reinforcement Learning." ICML 2017.
- Jiang, Y., Kolter, J. Z., Raileanu, R. "On the Importance of Exploration for Generalization in Reinforcement Learning." NeurIPS 2023. https://arxiv.org/abs/2306.05483
