---
title: "Synchronous Advantage Actor-Critic (A2C)"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - actor-critic
  - policy-gradient
origin: "Mnih et al. Asynchronous Methods for Deep Reinforcement Learning. ICML 2016. https://arxiv.org/abs/1602.01783"
created: 2026-10-20
updated: 2026-10-20
---
# Synchronous Advantage Actor-Critic (A2C)

## Idea in plain words

- Run many envs for a few steps, compute n-step advantages $A_t = G_{t:t+n} - V(s_t)$, take one gradient step on $-\log\pi(a_t \mid s_t) A_t$ plus a value loss, and throw the data away.
- A3C is the asynchronous version with many CPU workers; A2C waits for all envs and updates once.

## Verdict

- Rejected: same family as PPO, and a weaker special case of it.
- PPO is A2C plus a clipped probability ratio and several epochs over the same batch; with one epoch and no clipping, PPO's code is A2C. It answers no question that PPO-based runs cannot.
- The actor-critic family with a truly different update rule is covered by [[Actor-Critic with Eligibility Traces]] (online, per-step, traces).

## Why it fits Craftax

- Parallel JAX envs are the setting A2C was designed for, but PPO already uses them the same way ([[PPO (Matthews 2024)]]).

## How it works

From [[11 Policy approximation (actor-critic)]] and n-step returns in [[07a n-step TD and TD(λ)]].

1. Collect $n$ steps in every env.
2. Compute n-step returns and advantages.
3. One gradient step on the policy, value and entropy losses.

## How to implement here

- `src/wut_murl_craftax/training/ppo.py` with one epoch, one minibatch and the clip range set very large.

## Risks

- Expected to score at or below PPO.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax A2C", GitHub repository search "craftax a2c" (no hits).
- No A2C result on Craftax found; the only citing paper that runs A2C (Explanatory World Models, [arXiv 2610.13504](https://arxiv.org/abs/2610.13504)) runs it on a recommender benchmark, not Craftax.
- Not found, but not worth a slot.

## How to evaluate

- If run anyway: as a PPO ablation (epochs 1, no clip) on Craftax-1B against PPO's 11.9% (block A of [[Existing Implementations]]).

## References

- Mnih, V. et al. "Asynchronous Methods for Deep Reinforcement Learning." ICML 2016. https://arxiv.org/abs/1602.01783
- Schulman, J. et al. "Proximal Policy Optimization Algorithms." 2017. https://arxiv.org/abs/1707.06347
