---
title: "Count-Based Progress Bonus"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - exploration
  - intrinsic-reward
  - count-based
origin: "Tang et al. #Exploration: A Study of Count-Based Exploration for Deep RL. NeurIPS 2017. https://arxiv.org/abs/1611.04717; Bellemare et al. Unifying Count-Based Exploration and Intrinsic Motivation. NeurIPS 2016. https://arxiv.org/abs/1606.01868"
suggested-by:
  - "[[PQN-RNN (Gallici 2025)]]"
  - "[[PPO (Matthews 2024)]]"
created: 2026-10-20
---
# Count-Based Progress Bonus

## Idea in plain words

- Give a small extra reward for reaching a situation the agent has rarely been in.
- "Situation" is not the raw screen. It is a short summary of game progress: floor, tool levels, armour, bow, spells and rounded inventory counts.
- Count how often each summary was seen over all of training. Bonus = $\beta / \sqrt{N}$, so it shrinks as the situation becomes common.
- The agent is pushed toward new progress (first iron, first diamond pair, a new floor), not toward new grass.

## Why it fits Craftax

- The Craftax paper's learned bonuses (RND, ICM, E3B) did not help at 1B; the authors suspect the bonus distracts the agent ([[RND (Matthews 2024)]], [[E3B (Matthews 2024)]]). These bonuses look at the whole 8268-float observation, which changes with terrain, mobs and daylight ([[Craftax observations and actions]]).
- A progress summary ignores scenery. Only crafting, mining and descending change it ([[Craftax crafting and placing]], [[Craftax tools weapons and armour]], [[Craftax floors and progression]]).
- The game already pays once per achievement, but many useful states are not achievements: holding iron and coal at the same time, 3 diamonds before a diamond pickaxe, 8 kills on a floor (ladder open). The summary can include these.
- The symbolic observation already holds the needed fields as numbers (inventory, tool levels, floor, ladder-open flag), so no learned network is needed.

## How it works

Builds on optimism and upper-confidence bonuses ([[02 Multi-armed bandits]]) and the Dyna-Q+ exploration bonus ([[08a Models, planning, Dyna]]).

1. Define a progress key $\phi(s)$ from the inventory part of the observation: floor, pickaxe and sword level, armour levels, bow, learned spells, ladder-open flag, and $\lfloor \log_2(1+n) \rfloor$ for each material count $n$.
2. Hash the key to an integer in $[0, K)$ (for example $K = 2^{20}$).
3. Keep a global count table $N[\cdot]$. After each rollout, add 1 for every visited key.
4. Intrinsic reward: $r^+_t = \beta / \sqrt{N[\phi(s_{t+1})]}$.
5. Train PPO on $r_t + r^+_t$, or with a second value head for $r^+$ as in RND.

```
key  = hash(floor, pick, sword, armour, bow, spells, ladder_open, log2_bucket(inventory))
N[key] += 1                      # global, never reset
bonus = beta / sqrt(N[key])
```

## How to implement here

- Base: `ppo_rnn.py` ([[Craftax Baselines]]); the RND script `ppo_rnd.py` already shows how to add an intrinsic reward with its own value head.
- New code, about 60-100 lines: a key function on the observation slice `[8217:8268]`, a hash (multiply-add of the integer fields mod $K$), the count table (`jnp.zeros(K, jnp.int32)`) carried in the train state, `counts.at[keys].add(1)` after each rollout.
- No new network and no new optimizer.
- Extra hyperparameters: β (try 0.01, 0.03, 0.1), table size $K$, bucket rule, extrinsic/intrinsic value heads on or off.
- Variant for PQN: replace ε-greedy exploration in PQN with this bonus (PQN's authors call ε-greedy too simple).

## Cost and risk

- Compute: negligible (one hash and one scatter-add per step). Craftax-1M runs on a Mac CPU in minutes to hours; Craftax-1B needs a GPU ([[Craftax Baselines compute and wall-clock]]).
- Risk: a bad key. Too fine (exact counts) and every state is new; too coarse and nothing is new. Start coarse and log how many distinct keys appear.
- Risk: the bonus rewards hoarding (99 wood is a new bucket). Cap buckets at the amounts recipes need.
- Risk: hash collisions with small $K$; check the fill rate.
- Risk: β too large hurts, as seen for RND (an autonomous sweep found 0.003-0.015 sensitive; [auto-craftax report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.5).

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (105 papers), web searches "Craftax count-based", "Crafter count-based exploration", GitHub repositories named "craftax".
- Done on Craftax: learned novelty bonuses only (RND, ICM, E3B in the Craftax paper; NovelD in the auto-craftax campaign, -4.7%; temporal-contrastive C-TeC on Craftax-Classic, [arXiv 2603.02008](https://arxiv.org/abs/2603.02008)).
- No count-based or hash-based bonus on Craftax found.
- New here: counts over a hand-chosen progress summary that ignores terrain, plus recipe-aware buckets.

## Suggested by existing work

- PQN paper §5.2: ε-greedy "is too simple to solve ALE", "a clear research direction" (said for Atari; PQN-RNN on Craftax uses the same ε-greedy) ([[PQN-RNN (Gallici 2025)]]).
- Craftax paper §6 Conclusion: hopes for research into "exploration, continual learning, generalisation, skill acquisition and long term reasoning" ([[PPO (Matthews 2024)]]).

## How to evaluate

- Craftax-1M against block D of the [[Existing Implementations]] table: PPO-RNN 2.3%, RND 2.2%, E3B 2.2%.
- Craftax-1B against block A: PPO-RNN 15.3%, RND 12.0%, E3B 11.0%.
- Same seeds, our own PPO-RNN rerun as the control; 5+ seeds.
- Achievement success rates for the first achievements of each tier, and the share of episodes that enter floors 1 and 2.
- Log the number of distinct keys seen over training; it should grow faster than for PPO-RNN.

## References

- Tang, H. et al. "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning." NeurIPS 2017. https://arxiv.org/abs/1611.04717
- Bellemare, M. et al. "Unifying Count-Based Exploration and Intrinsic Motivation." NeurIPS 2016. https://arxiv.org/abs/1606.01868
- Gallici, M. et al. "Simplifying Deep Temporal Difference Learning." ICLR 2025. https://arxiv.org/abs/2407.04811
- Matthews, M. et al. "Craftax." ICML 2024. https://arxiv.org/abs/2402.16801
