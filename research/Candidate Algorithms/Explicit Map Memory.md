---
title: "Explicit Map Memory"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - memory
  - partial-observability
  - observation
origin: "Parisotto, Salakhutdinov. Neural Map: Structured Memory for Deep Reinforcement Learning. ICLR 2018. https://arxiv.org/abs/1702.08360"
created: 2026-10-20
---
# Explicit Map Memory

## Idea in plain words

- The agent sees only a 9 x 11 window of a 48 x 48 floor.
- A recurrent network (GRU) must remember where the water, the table and the ladder were. That is hard for it.
- Give the agent a written map instead: every tile it has seen is stored in a 48 x 48 grid per floor.
- Each step, feed the policy a crop of this map around the player plus a coarse summary of the whole floor.
- The map is not learned. It is filled in by a simple rule, like drawing a map on paper.

## Why it fits Craftax

- The symbolic view is 9 x 11 tiles; floors are 48 x 48 ([[Craftax observations and actions]], [[Craftax floors and progression]]).
- Floors 2, 5, 7 and 8 are dark; unlit tiles are blanked ([[Craftax floors and progression]]). Remembering lit tiles and placed torches matters there.
- The ladder may be found before 8 kills open it; the agent has to come back to it later ([[Craftax floors and progression]]).
- Water, cows and the crafting table are needed again and again for drink, food and crafting ([[Craftax survival stats]], [[Craftax crafting and placing]]).
- Memory already helps: PPO-RNN beats PPO, and PPO-GTrXL beats PPO-RNN on Craftax-1B ([[Existing Implementations]]). A map gives long spatial memory without a transformer.

## How it works

Builds on the Markov property and state construction ([[03a MDP setting and returns]]) and feature construction for function approximation ([[09a Function approximation and linear methods]]).

1. Keep `memory[floor, x, y]` = last seen block type (an integer), plus `seen[floor, x, y]` and `last_seen_time`.
2. Each step, write the visible tiles of the 9 x 11 view into the memory at the player's position.
3. Build extra inputs: a 15 x 15 crop of the memory centred on the player, and the whole floor pooled to 12 x 12 (fraction of each important block type per cell).
4. Encode them with a small CNN and concatenate with the usual observation embedding before the GRU.
5. Clear the memory at episode end. Keep one map per floor, so going back up still has the old map.

## How to implement here

- A wrapper that keeps the memory in the env state, about 100-150 lines; a small CNN in the network, about 30 lines.
- Position: Craftax's symbolic observation has no absolute position. Two options:
  - read `env_state.player_position` (simple, but privileged information; report it as such);
  - dead reckoning: track position from movement actions and whether the view shifted (harder, no privileged information).
- Store block ids as `int8`: 9 floors x 48 x 48 = about 21 KB per environment, about 21 MB for 1024 environments.
- Extra hyperparameters: crop size, pooled map size, which block types to keep (all 37 or a short list such as water, table, furnace, ladder, ores, lava).

## Cost and risk

- Compute: the CNN on a 15 x 15 crop is cheap; 10-30% slower overall. Fits a Mac for Craftax-1M.
- Risk: privileged position. Results with `player_position` are not strictly comparable with the baselines; run the dead-reckoning variant for a fair comparison.
- Risk: stale map. Mobs move and blocks change (mined, placed); store only static block types and the time since last seen.
- Risk: more input dimensions slow early learning on Craftax-1M.

## Novelty check

- Searched: web "Craftax allocentric map memory", "Craftax episodic memory", Semantic Scholar citations of arXiv 2402.16801, GitHub repositories named "craftax".
- Memory on Craftax so far is sequence memory: GRU ([[PPO-RNN (Matthews 2024)]]), LSTM ([[PQN-RNN (Gallici 2025)]]), Transformer-XL ([[PPO-GTrXL (Hamon 2024)]]), complex-valued RNNs ([arXiv 2609.38598](https://arxiv.org/abs/2609.38598)), RetNet in a world model ([[Simulus (Cohen 2025)]]).
- A partial-observability benchmark masks Craftax's inventory to test memory ([arXiv 2508.00046](https://arxiv.org/abs/2508.00046)); it adds no map.
- No explicit spatial map memory on Craftax found.
- New here: a hand-written per-floor map, the simplest form of the Neural Map idea, fitted to Craftax's grid.

## How to evaluate

- Reward % of max on Craftax-1M and Craftax-1B against blocks D and A of the [[Existing Implementations]] table, with privileged and non-privileged position reported separately.
- Success rates of `ENTER_DUNGEON`, `ENTER_GNOMISH_MINES`, `COLLECT_DRINK` timing, and deaths from thirst.
- Ablation: crop only, pooled map only, both.

## References

- Parisotto, E., Salakhutdinov, R. "Neural Map: Structured Memory for Deep Reinforcement Learning." ICLR 2018. https://arxiv.org/abs/1702.08360
- "Benchmarking Partial Observability in Reinforcement Learning with a Suite of Memory-Improvable Domains." 2025. https://arxiv.org/abs/2508.00046
- "Reinforcement Learning with Complex (valued) Memories." 2026. https://arxiv.org/abs/2609.38598
