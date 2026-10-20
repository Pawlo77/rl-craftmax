---
title: "Go-Explore with Achievement Cells"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - exploration
  - go-explore
  - start-states
origin: "Ecoffet, Huizinga, Lehman, Stanley, Clune. First return, then explore. Nature 2021. https://arxiv.org/abs/2004.12919"
suggested-by:
  - "[[PPO (Matthews 2024)]]"
created: 2026-10-20
updated: 2026-10-20
---
# Go-Explore with Achievement Cells

## Idea in plain words

- Keep an archive of the most interesting game states found so far, for example "on floor 1 with an iron pickaxe".
- Group states into cells. A cell is a short progress summary (floor, tools, achievement count), not the exact screen.
- At the start of some episodes, do not start a new world. Load a saved state from the archive and play on from there ("first return, then explore").
- Pick cells that were rarely chosen, so the frontier keeps moving.
- Craftax makes the "return" step free: the whole game state is one JAX object that can be saved and loaded.

## Why it fits Craftax

- Agents hit a floor wall: PPO-RNN does not progress into the Gnomish Mines at 1B or 10B steps ([[PPO-RNN (Matthews 2024)]]). To reach floor 2 an episode must first gather tools, open the ladder with 8 kills on floor 1, and survive ([[Craftax floors and progression]]).
- Every fresh episode restarts at the overworld centre with an empty inventory ([[Craftax episode and time]]). Most training steps repeat the easy early game.
- Starting from archived floor-1 and floor-2 states gives the agent many more chances to practise deep-floor skills.
- `EnvState` is a `flax.struct.dataclass`; one batched state can be indexed, stored and written back with `jax.tree_util.tree_map` ([[Craftax API and wrappers]]).

## How it works

Builds on exploring starts in Monte Carlo control ([[05a Monte Carlo prediction and control]]): every state should be a possible start. Also related to trajectory sampling ([[08b Trajectory sampling, search, MCTS]]).

1. Cell function $c(s)$ = (floor, pickaxe level, sword level, armour count, bow, number of achievements unlocked, ladder open).
2. Archive: for each cell, one saved `EnvState` and a counter of how often it was chosen.
3. During training, after each rollout, add any state whose cell is new (or reached with a higher return) to the archive.
4. On episode reset, with probability $p$: sample cell $k$ with weight $1/\sqrt{\text{chosen}_k + 1}$ and load its state. Otherwise reset normally.
5. Train PPO on all episodes. Start the GRU from a zero hidden state.
6. Evaluate only on normal resets, with $p = 0$.

```
on reset(env i):
    if rand() < p and archive not empty:
        k = sample(archive, weight = 1/sqrt(chosen + 1))
        state_i = archive[k].state ; chosen[k] += 1
    else:
        state_i = new_world()
```

## How to implement here

- Base: `ppo_rnn.py`; the change is in the auto-reset wrapper (`AutoResetEnvWrapper` / `OptimisticResetVecEnvWrapper` in `wrappers.py`, [[Craftax Baselines]]).
- New code, about 150-250 lines: cell function, archive as a fixed-size batched `EnvState` (for example 512 slots), insertion with `jax.lax` ops, weighted sampling with `jax.random.choice`.
- Read progress from `env_state` fields (`achievements`, inventory, `player_level`), not from `info`: achievement info is non-zero only on the done step ([[Craftax Baselines benchmark protocol]]).
- The saved state keeps its own timestep, so the 100,000-step cap still applies ([[Craftax episode and time]]).
- Extra hyperparameters: $p$ (0.25-0.5), archive size, cell definition, whether returns from archived starts count toward the benchmark step budget (they should).

## Risks

- Runtime: environment steps are unchanged; the archive costs memory. One `EnvState` holds all 9 floors of 48 x 48 tiles in several arrays; budget roughly 0.1-1 MB per slot, about 50-500 MB for 512 slots.
- Risk: distribution shift. The agent sees deep states more often than in real play, and may play the early game worse. Keep $p \le 0.5$ and always evaluate on normal resets.
- Risk: archived states are fragile (low health, night). Store the best state per cell, by return or health.
- Risk: hidden-state mismatch: a GRU started at zero in the middle of a game lacks memory of what came before.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801 (105 papers), web search "Craftax Go-Explore", GitHub repositories named "craftax".
- Closest published work:
  - Anchor-and-Harvest Curriculum stores "progress signatures" and writes them into freshly generated worlds as start states for achievement tasks, with difficulty knobs; about 20.7% of max at 2B steps ([arXiv 2610.15328](https://arxiv.org/abs/2610.15328)).
  - SCALAR's Frontier Checkpointing resets to saved states at LLM-defined skill boundaries ([arXiv 2603.09036](https://arxiv.org/abs/2603.09036)).
  - The auto-craftax campaign implemented Go-Explore checkpointing, but a bug (reading achievements from `info`) made it fail; no valid result.
- No archive-based Go-Explore with count-weighted cell selection and exact state restore found on Craftax.
- New here: exact JAX state restore, cells from game progress, selection by visit counts, no task list, no LLM.

## Suggested by existing work

- Craftax paper §6 Conclusion lists "exploration" first among the research areas it hopes to enable; abstract: solving Craftax needs "deep exploration" ([[PPO (Matthews 2024)]]).

## How to evaluate

- Main metric on normal resets: Craftax-1B reward % of max against block A of the [[Existing Implementations]] table (PPO-RNN 15.3%, PPO-GTrXL 18.3%).
- Secondary: Craftax-1M against block D (PPO-RNN 2.3%), where the archive holds mostly overworld and floor-1 cells.
- Success rates of `ENTER_GNOMISH_MINES`, `ENTER_SEWERS`, `DEFEAT_GNOME_WARRIOR`: these are near zero for every 1B baseline.
- Archive growth: number of cells and deepest floor in the archive over training.
- Ablation: $p = 0$ (plain PPO-RNN) versus $p = 0.25, 0.5$.

## References

- Ecoffet, A. et al. "First return, then explore." Nature 590, 2021. https://arxiv.org/abs/2004.12919
- Zhu, J. et al. "Anchor-and-Harvest Curriculum: A Fully Unmasked Non-LLM Curriculum for Craftax." 2026. https://arxiv.org/abs/2610.15328
- "SCALAR: Learning and Composing Skills through LLM Guided Symbolic Planning and Deep RL Grounding." 2026. https://arxiv.org/abs/2603.09036
- auto-craftax research report. https://github.com/roger-creus/auto-craftax
