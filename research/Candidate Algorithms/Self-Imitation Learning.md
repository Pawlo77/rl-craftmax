---
title: "Self-Imitation Learning"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - ppo
  - self-imitation
  - sparse-reward
origin: "Oh, Guo, Singh, Lee. Self-Imitation Learning. ICML 2018. https://arxiv.org/abs/1806.05635"
created: 2026-10-20
updated: 2026-10-20
---
# Self-Imitation Learning

## Idea in plain words

- The agent sometimes gets lucky: it crafts an iron pickaxe or finds the ladder by chance.
- PPO uses that lucky episode for a few gradient steps and then throws it away.
- Self-imitation learning (SIL) keeps such good episodes in a buffer and keeps training the policy to repeat their actions.
- It only imitates an action when the return that followed it was better than the critic expected, so bad episodes are ignored.
- Craftax twist: keep the buffer per achievement, so every rare achievement keeps its own "how I did it" examples.

## Why it fits Craftax

- Rewards are one-time achievement bonuses, and the deep ones are very rare ([[Craftax achievements]]). PPO-RNN reaches the Gnomish Mines "only very rarely" at 1B steps ([[PPO-RNN (Matthews 2024)]]).
- A rare success is a long recipe: wood, table, pickaxes, furnace, coal and iron, then the ladder after 8 kills ([[Craftax crafting and placing]], [[Craftax floors and progression]]). One good episode shows the whole chain, and SIL replays it many times.
- Basic achievements are mastered early and stop giving gradient. A per-achievement buffer keeps rare skills (for example `ENTER_GNOMISH_MINES`, `FIND_BOW`) from being forgotten.
- `DEFEAT_ZOMBIE` success falls during training as agents trade overworld goals for dungeon goals ([[Craftax Baselines results]]). Replaying old successes counters such forgetting.

## How it works

Builds on Monte Carlo returns ([[05a Monte Carlo prediction and control]]) and the actor-critic ([[11 Policy approximation (actor-critic)]]).

1. Run PPO as usual.
2. When an episode ends, compute the discounted return $R_t$ for every step.
3. Store $(s_t, a_t, R_t)$ in a replay buffer. Craftax variant: store only the window of steps before a newly unlocked achievement, in a slot for that achievement.
4. After each PPO update, sample a minibatch from the buffer, preferring large $(R_t - V(s_t))_+$.
5. Add the SIL loss and take extra gradient steps:

$$L_{\text{sil}} = -\log \pi(a_t \mid s_t)\,(R_t - V(s_t))_+ + \tfrac{\beta}{2}\,\big((R_t - V(s_t))_+\big)^2$$

- $(x)_+ = \max(x, 0)$: only better-than-expected actions are imitated.
- When the critic catches up, $(R_t - V)_+$ goes to zero and the sample stops mattering.

## How to implement here

- Base: `ppo.py` (feed-forward) first; `ppo_rnn.py` later ([[Craftax Baselines]]).
- New code, about 120-180 lines:
  - a fixed-size buffer in JAX arrays (`obs`, `action`, `return`, `achievement_id`), written with `.at[idx].set`;
  - per-episode return computation from the rollout (reverse `jax.lax.scan`);
  - an extra loss and 1-2 extra optimizer steps per update.
- Read achievement unlocks from `env_state.achievements`, not from `info["Achievements/..."]`: the info value is non-zero only on the step where the episode ends ([[Craftax Baselines benchmark protocol]]).
- RNN version: store the GRU hidden state with each step, or store whole windows of 32-64 steps and replay them from a zero state.
- Memory: one symbolic observation is 8268 floats (33 KB). Store as `float16` or `uint8` (most channels are one-hot); 50k steps in `float16` is about 0.8 GB.
- Extra hyperparameters: buffer size (20k-100k steps), SIL updates per PPO update (1-2), SIL minibatch size (512), value-loss weight β (0.01), window length per achievement (256-1024 steps).

## Risks

- Runtime: one extra forward and backward pass per update, about 10-30% slower per update.
- Risk: buffer memory. An autonomous agent's SIL attempt on Craftax crashed with CUDA out-of-memory ([auto-craftax report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.5). Keep the buffer small and compressed.
- Risk: imitating lucky but unsafe behaviour (for example sleeping next to a zombie). Prefer recent entries and cap each episode's share of the buffer.
- Risk: off-policy samples inside PPO. Keep the SIL step small (its own learning rate or weight).

## Novelty check

- Searched: arXiv and Semantic Scholar (105 papers citing arXiv 2402.16801, read 2026-10-20), web search "Craftax self-imitation", GitHub repositories named "craftax".
- No published SIL result on Craftax or Craftax-Classic found.
- Anchor-and-Harvest Curriculum lists self-imitation as disabled ("Self-imitation learning is off"; [arXiv 2610.15328](https://arxiv.org/abs/2610.15328) §2).
- The auto-craftax campaign tried SIL but the runs crashed (out of memory), so no valid result exists.
- New here: per-achievement buffer slots, aimed at rare deep achievements.

## How to evaluate

- Craftax-1M: reward as % of max (226) against [[PPO-RNN (Matthews 2024)]] 2.3% and [[PPO (Matthews 2024)]] 2.2% in the [[Existing Implementations]] table, block D.
- Craftax-1B: against PPO-RNN 15.3% and [[PPO-GTrXL (Hamon 2024)]] 18.3%, block A.
- Rerun our own PPO-RNN with the same seeds and budget; report the difference with 5+ seeds.
- Achievement success rates: `ENTER_DUNGEON`, `ENTER_GNOMISH_MINES`, `MAKE_IRON_PICKAXE`, `DEFEAT_ZOMBIE` (forgetting check).

## References

- Oh, J., Guo, Y., Singh, S., Lee, H. "Self-Imitation Learning." ICML 2018. https://arxiv.org/abs/1806.05635
- Matthews, M. et al. "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning." ICML 2024. https://arxiv.org/abs/2402.16801
- Zhu, J. et al. "Anchor-and-Harvest Curriculum." 2026. https://arxiv.org/abs/2610.15328
- auto-craftax research report. https://github.com/roger-creus/auto-craftax
