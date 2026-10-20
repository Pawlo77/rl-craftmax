---
title: "V-trace Off-Policy Actor-Critic"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - actor-critic
  - off-policy
  - importance-sampling
origin: "Espeholt et al. IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. ICML 2018. https://arxiv.org/abs/1802.01561"
created: 2026-10-20
updated: 2026-10-20
---
# V-trace Off-Policy Actor-Critic

## Idea in plain words

- IMPALA: many actors play with slightly old copies of the policy and send their data to one learner.
- V-trace corrects for the difference between the old policy (behaviour) and the current one (target) with clipped importance weights $\min(\bar c, \pi/\mu)$.
- Clipped weights keep the variance low, at the price of a small bias.

## Verdict

- Rejected: it fixes a problem Craftax training does not have.
- In the JAX setup, envs and learner run in lockstep on the same device ([[Craftax performance]]); data is always from the current policy, so all importance weights are 1 and V-trace reduces to an on-policy n-step actor-critic (A2C, [[Synchronous Advantage Actor-Critic (A2C)]]).
- Its other use, learning from replayed old data, overlaps [[Self-Imitation Learning]] and needs truncated importance sampling beyond the book's plain version.

## Why it fits Craftax

- Only if old data is replayed; the default Craftax pipeline has no actor lag.

## How it works

From importance sampling in [[05b Off-policy Monte Carlo]] and off-policy traces in [[07b Eligibility-trace control and off-policy]].

1. Collect data with behaviour policy $\mu$.
2. V-trace target: $v_s = V(s) + \sum_t \gamma^{t-s} (\prod_{i<t} c_i)\, \rho_t \delta_t$ with clipped ratios $\rho_t, c_i$.
3. Policy gradient weighted by $\rho_t (r_t + \gamma v_{t+1} - V(s_t))$.

## How to implement here

- `rlax.vtrace` on top of the A2C form of `training/ppo.py`.

## Risks

- No gain without lag or replay.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (106 papers), web "Craftax IMPALA V-trace".
- No IMPALA result on Craftax found; IMPALA was used on Crafter by SEA ([arXiv 2305.00508](https://arxiv.org/abs/2305.00508)) as a sub-policy learner and baseline.

## How to evaluate

- Not planned.

## References

- Espeholt, L. et al. "IMPALA." ICML 2018. https://arxiv.org/abs/1802.01561
