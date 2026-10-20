---
title: "In-Context Adaptation Memory"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - memory
  - meta-rl
  - potions
origin: "Duan et al. RL²: Fast Reinforcement Learning via Slow Reinforcement Learning. 2016. https://arxiv.org/abs/1611.02779; Wang et al. Learning to Reinforcement Learn. 2016. https://arxiv.org/abs/1611.05763"
suggested-by:
  - "[[PPO-RNN (Matthews 2024)]]"
created: 2026-10-20
---
# In-Context Adaptation Memory

## Idea in plain words

- Some Craftax rules change every episode. Potion colours are shuffled: red heals in one game and hurts in the next.
- The agent can only find out by trying and remembering what happened.
- Give the recurrent network the inputs it needs to learn this: the previous action, the previous reward and the change in health, mana and energy.
- With these inputs, the GRU can learn a small "experiment, then exploit" routine inside one episode (meta-reinforcement learning, RL²).

## Why it fits Craftax

- Each episode draws a random permutation of six potion effects over six colours: +8 or -3 health, mana or energy ([[Craftax magic potions and enchanting]], [[Craftax episode and time]]).
- `DRINK_POTION` is an Intermediate achievement (3 points); later floors need healing and mana potions ([[Craftax achievements]]).
- The baseline GRU sees only the current observation; it gets no direct signal of what its last action did to its stats.
- The Craftax paper names potions as a test of in-context learning and memory (§3.2).

## How it works

Builds on the Markov property and partial observability ([[03a MDP setting and returns]]) and the actor-critic ([[11 Policy approximation (actor-critic)]]).

1. At each step, build an extra input: one-hot previous action, previous reward, and the change in health, food, drink, energy and mana since the last step.
2. Concatenate it with the observation embedding before the GRU.
3. Train PPO-RNN as usual; reset the GRU at episode start, never inside an episode.
4. Optional: an auxiliary head that predicts the effect of each potion colour from the GRU state, trained on the observed stat changes after drinking.

## How to implement here

- Base: `ppo_rnn.py` ([[Craftax Baselines]]); PQN-RNN already feeds the last action ([[PQN-RNN (Gallici 2025)]]).
- Changes: carry the previous action and previous stats in the rollout state; concatenate them in `ActorCriticRNN`. About 30-60 lines; the auxiliary head adds about 40 more.
- Extra hyperparameters: which deltas to include, auxiliary loss weight.

## Cost and risk

- Compute: negligible. Fits a Mac at Craftax-1M scale; potion use matters mostly in long 1B runs, so a GPU is needed to see the full effect.
- Student level: yes, low effort.
- Risk: potions are found in chests on deeper floors ([[Craftax items and inventory]]); if the agent rarely gets there, the effect is too rare to measure.
- Risk: small overall gain; measure potion-specific metrics, not only total reward.

## Novelty check

- Searched: web "Craftax potions in-context learning", "Craftax meta-RL", Semantic Scholar citations of arXiv 2402.16801.
- Memory architectures on Craftax exist (GRU, LSTM with last action, GTrXL, complex-valued RNNs; see [[Existing Implementations]]), but no study of potion identification or of reward and stat-change inputs was found.
- Not tried on Craftax as such.

## Suggested by existing work

- Craftax paper §3.2: shuffled potions mean "further testing in-context learning and memory" ([[PPO-RNN (Matthews 2024)]]).

## How to evaluate

- Potion metrics: share of drinks that are the helpful colour after the first drink of each colour; health lost to bad potions.
- Reward % of max on Craftax-1B against block A of the [[Existing Implementations]] table.
- `DRINK_POTION` success rate against PPO-RNN.

## References

- Duan, Y. et al. "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning." 2016. https://arxiv.org/abs/1611.02779
- Wang, J. X. et al. "Learning to Reinforcement Learn." 2016. https://arxiv.org/abs/1611.05763
- Matthews, M. et al. "Craftax." ICML 2024. https://arxiv.org/abs/2402.16801
