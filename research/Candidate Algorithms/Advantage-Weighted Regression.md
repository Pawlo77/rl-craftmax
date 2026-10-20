---
title: "Advantage-Weighted Regression"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - policy-improvement
  - off-policy
origin: "Peng, Kumar, Zhang, Levine. Advantage-Weighted Regression: Simple and Scalable Off-Policy RL. 2019. https://arxiv.org/abs/1910.00177"
created: 2026-10-20
updated: 2026-10-20
---
# Advantage-Weighted Regression

## Idea in plain words

- Keep a replay buffer of past steps.
- Fit the critic $V(s)$ to returns from the buffer.
- Fit the policy by supervised learning: copy the buffer's actions, each weighted by $\exp(A / \beta)$, so better-than-expected actions count more.

## Verdict

- Rejected: not in the textbook, and on Craftax it lands between two things already planned.
- Without replay it is a policy-improvement step close to PPO's; with replay it is a softer form of [[Self-Imitation Learning]] (which imitates only actions with positive advantage), already a Top 5 add-on.
- No separate research question for a student project.

## Why it fits Craftax

- Reusing rare successful steps fits sparse achievements, but SIL covers that directly.

## How it works

1. Buffer of $(s, a, G)$.
2. Critic: regress $V(s)$ to $G$.
3. Policy: maximise $\sum \log\pi(a \mid s)\exp\big((G - V(s))/\beta\big)$.

## How to implement here

- Shares the buffer of [[Self-Imitation Learning]]; one weighted cross-entropy loss.

## Risks

- Weight explosion for large advantages (needs clipping); a large buffer of 8268-float observations ([[Craftax observations and actions]]).

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax advantage weighted regression".
- No AWR result on Craftax found.

## How to evaluate

- Not planned; if run, as a SIL ablation (exponential weights against positive-part weights).

## References

- Peng, X. B. et al. "Advantage-Weighted Regression." 2019. https://arxiv.org/abs/1910.00177
