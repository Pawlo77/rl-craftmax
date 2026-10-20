---
title: "Off-Policy Dyna with Imagined Data"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - model-based
  - dyna
  - off-policy
origin: "Dedieu et al. Improving Transformer World Models for Data-Efficient RL. 2025. https://arxiv.org/abs/2502.01591; Sutton. Dyna. 1991."
suggested-by:
  - "[[Efficient MBRL (Dedieu 2025)]]"
created: 2026-10-20
---
# Off-Policy Dyna with Imagined Data

## Idea in plain words

- Dyna: learn a model of the game, then train the policy on both real games and games imagined by the model.
- The Efficient MBRL agent does this with PPO, which can only use fresh on-policy data, so it updates on real and imagined data in separate steps.
- Their suggestion: use an off-policy learner (for example Q-learning) that can mix real and imagined data in one replay batch.
- They also suggest prioritised replay to train the world model faster.

## Why it fits Craftax

- Craftax-1M is a sample-efficiency test: only 1M real steps ([[Craftax Baselines benchmark protocol]]). Every extra imagined step is free data.
- Model-based agents lead the 1M blocks of the [[Existing Implementations]] table ([[Simulus (Cohen 2025)]], [[Efficient MBRL (Dedieu 2025)]]).
- PQN shows that off-policy Q-learning works on Craftax without a large replay buffer ([[PQN-RNN (Gallici 2025)]]).

## How it works

Builds on Dyna ([[08a Models, planning, Dyna]]), Q-learning ([[06 Temporal-difference learning]]) and off-policy learning ([[05b Off-policy Monte Carlo]]).

1. Collect real steps with the current policy; store them.
2. Train a world model on stored real data (prioritised by model loss).
3. Start imagined rollouts from real states; roll the model forward a few steps.
4. Put real and imagined transitions in one batch; update a Q-learning (or off-policy actor-critic) agent on it.
5. Warm up: use imagined data only after the model is good enough (Efficient MBRL waits 200k steps).

## How to implement here

- Not a small change: needs a world model (Efficient MBRL uses a transformer world model with a patch tokenizer) and an off-policy learner.
- Student-scale variant: Craftax-Classic symbolic, a small MLP model that predicts the next inventory and nearby tiles, PQN as the learner. Still several hundred lines.
- Extra hyperparameters: imagination horizon, ratio of imagined to real data, warm-up length, model-replay priority.

## Cost and risk

- Student level: no, as proposed. Efficient MBRL trained on 8 H100 GPUs; a faithful version is beyond a one-semester student project. The small-model variant is possible but research-grade.
- Risk: model errors compound over imagined steps; Efficient MBRL saw collapse without warm-up.
- Risk: off-policy learning on imagined data can diverge (the "deadly triad").

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801, web "Craftax Dyna off-policy".
- Prioritised world-model replay was already used on Craftax-1M by [[Simulus (Cohen 2025)]].
- An off-policy learner mixing real and imagined data on Craftax: not found.

## Suggested by existing work

- Efficient MBRL §3.4: "use an off-policy algorithm and mix real and imaginary data"; left as future work ([[Efficient MBRL (Dedieu 2025)]]).
- Efficient MBRL §5: prioritised experience replay "is a promising approach to accelerate TWM training" ([[Efficient MBRL (Dedieu 2025)]]).

## How to evaluate

- Craftax-1M reward % of max against block D (symbolic) or E/F (pixels) of the [[Existing Implementations]] table.
- Ablations: no imagination, imagination with separate updates (as in Efficient MBRL), mixed batches.

## References

- Dedieu, A. et al. "Improving Transformer World Models for Data-Efficient RL." 2025. https://arxiv.org/abs/2502.01591
- Cohen, L. et al. "Simulus." 2025. https://arxiv.org/abs/2502.11537
- Sutton, R. S. "Dyna, an integrated architecture for learning, planning, and reacting." SIGART Bulletin 2(4), 1991.
