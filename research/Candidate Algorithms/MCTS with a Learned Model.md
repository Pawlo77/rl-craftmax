---
title: "MCTS with a Learned Model"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - model-based
  - mcts
  - planning
origin: "Schrittwieser et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero). Nature 2020. https://arxiv.org/abs/1911.08265"
created: 2026-10-20
updated: 2026-10-20
---
# MCTS with a Learned Model

## Idea in plain words

- Learn a model that predicts, from a hidden state and an action, the next hidden state, the reward, a value and a policy (MuZero).
- At every decision, run Monte Carlo tree search (MCTS) inside the learned model and play the most-visited action.
- Train the model and the policy on the search results.

## Verdict

- Rejected: beyond student level at the level of game steps, and not needed at the skill level.
- Game-step level: MuZero-style agents combine a learned latent model, tree search inside the training loop, value and policy targets from search, and reanalysis; each part is research-grade, and learned models of Craftax screens already need large systems ([[Simulus (Cohen 2025)]], [[Efficient MBRL (Dedieu 2025)]]).
- Search with the real simulator is not comparable with the benchmark (already in the index's rejected ideas; best-of-attempts search in [[Sable Inference Search (Chalumeau 2025)]]).
- Skill level: the progress-key model is small enough for exact value iteration, so MCTS adds nothing over the full backups in [[Dyna-Q over a Skill-Level Progress Model]].

## Why it fits Craftax

- Planning a few moves ahead helps in combat and around lava, but the learned model is the hard part.

## How it works

From [[08b Trajectory sampling, search, MCTS]].

1. Selection by an upper-confidence rule down the tree.
2. Expansion with the learned model.
3. Evaluation with the learned value.
4. Backup of values to the root; act on visit counts.

## How to implement here

- DeepMind's `mctx` library in JAX gives the search; the learned model and training loop are the large part.

## Risks

- Model errors compound in deep trees; long-range Craftax goals are far beyond a search horizon of tens of steps.

## Novelty check

- Searched: web "MuZero OR EfficientZero OR Monte Carlo tree search Craftax", arXiv API `all:craftax`, Semantic Scholar citations of arXiv 2402.16801 (106 papers), GitHub repository search "craftax muzero", "craftax mcts" (no hits).
- No MuZero-style agent on Craftax found.

## How to evaluate

- Not planned.

## References

- Schrittwieser, J. et al. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." Nature 2020. https://arxiv.org/abs/1911.08265
- Ye, W. et al. "Mastering Atari Games with Limited Data (EfficientZero)." NeurIPS 2021. https://arxiv.org/abs/2111.00210
