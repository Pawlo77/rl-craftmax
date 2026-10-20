---
title: "Plasticity-Preserving Training"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - continual-learning
  - plasticity
  - optimisation
origin: "Ash, Adams. On Warm-Starting Neural Network Training (shrink and perturb). NeurIPS 2020. https://arxiv.org/abs/1910.08475; Nikishin et al. The Primacy Bias in Deep RL. ICML 2022. https://arxiv.org/abs/2205.07802"
suggested-by:
  - "[[PPO (Matthews 2024)]]"
  - "[[PPO-RNN (Matthews 2024)]]"
created: 2026-10-20
---
# Plasticity-Preserving Training

## Idea in plain words

- A network trained for a long time can lose its ability to learn new things ("loss of plasticity").
- Craftax keeps showing new situations late in training: a new floor, new creatures, new items.
- Simple fixes: every so often shrink the weights a little and add small noise ("shrink and perturb"), or re-initialise the last layers.
- The agent keeps what it knows (mostly) but stays able to learn the next floor.

## Why it fits Craftax

- PPO-RNN at 10B steps "barely improves" over 1B, and PPO-GTrXL gains nothing from 4B to 8B ([[PPO-RNN (Matthews 2024)]], [[PPO-GTrXL (Hamon 2024)]]). A stalled learner is one sign of lost plasticity.
- Each floor brings new creatures and rules ([[Craftax floors and progression]], [[Craftax creatures]]); the data the agent sees changes over training, as in continual learning.
- The Craftax paper says solving it needs "continual adaptation to novel situations" (abstract).

## How it works

Builds on the stability issues of bootstrapping with function approximation ([[09b Control with approximation, bootstrapping]]).

1. Train PPO-RNN as usual.
2. Every $T$ updates: $\theta \leftarrow \lambda\,\theta + \sigma\,\epsilon$, with $\epsilon$ from the initialiser's distribution ($\lambda \approx 0.8$-$0.99$, small $\sigma$).
3. Or: re-initialise the last 1-2 layers of actor and critic every $T$ updates.
4. Optional: reset Adam's statistics at the same time.
5. Monitor dormant units (units that are almost always zero) and the effective rank of the features.

## How to implement here

- Base: `ppo_rnn.py` ([[Craftax Baselines]]); a function on the parameter tree with `jax.tree_util.tree_map`, called inside the update loop with `jax.lax.cond`. About 30-50 lines.
- Extra hyperparameters: interval $T$, shrink $\lambda$, noise $\sigma$, which layers.

## Cost and risk

- Compute: negligible. The effect shows only in long runs (1B+), so a GPU is needed; Craftax-Classic runs can show it on a Mac at smaller scale.
- Student level: yes, low effort.
- Risk: each reset causes a temporary drop in return; too frequent resets prevent learning.
- Risk: plasticity loss may not be the bottleneck; exploration of the deep floors may be.

## Novelty check

- Searched: web "Craftax plasticity", Semantic Scholar citations of arXiv 2402.16801.
- Tried since the suggestion:
  - Plasticine includes Craftax as its open-ended environment and implements 13+ plasticity methods ([arXiv 2504.17490](https://arxiv.org/abs/2504.17490)).
  - Adam-Rel resets Adam's timestep after each PPO batch and was tested with PPO on Craftax-Classic at 1B steps ([arXiv 2412.17113](https://arxiv.org/abs/2412.17113)).
  - Gaussian trust-region PPO targets stalled behaviour change in non-stationary settings; it cites Craftax and reports open-world exploration results ([arXiv 2606.03382](https://arxiv.org/abs/2606.03382)).
- Status: already explored; a student could still run a clean shrink-and-perturb study at Craftax-1B with PPO-RNN.

## Suggested by existing work

- Craftax paper §6 Conclusion: research into "exploration, continual learning, generalisation, skill acquisition and long term reasoning" ([[PPO (Matthews 2024)]]).
- Craftax paper App. F.2: PPO-RNN barely improves from 1B to 10B steps ([[PPO-RNN (Matthews 2024)]]).

## How to evaluate

- Reward % of max at 1B and beyond against blocks A and C of the [[Existing Implementations]] table.
- Learning-curve slope after 500M steps; dormant-unit fraction over training.

## References

- Ash, J., Adams, R. "On Warm-Starting Neural Network Training." NeurIPS 2020. https://arxiv.org/abs/1910.08475
- Nikishin, E. et al. "The Primacy Bias in Deep Reinforcement Learning." ICML 2022. https://arxiv.org/abs/2205.07802
- "Plasticine: Accelerating Research in Plasticity-Motivated Deep Reinforcement Learning." 2025. https://arxiv.org/abs/2504.17490
