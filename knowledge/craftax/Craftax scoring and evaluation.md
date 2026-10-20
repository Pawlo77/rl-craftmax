---
title: "Craftax scoring and evaluation"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, evaluation, reward]
created: 2026-10-20
updated: 2026-10-21
---
# Craftax scoring and evaluation

## Reward

Per step, exactly as the end of `craftax_step` computes it ([game_logic.py L3072-L3079](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L3072-L3079)); `init_*` are read at the start of the step ([L3006-L3008](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L3006-L3008)):

```python
achievement_reward = (
    (state.achievements.astype(int) - init_achievements.astype(int))
    * ACHIEVEMENT_REWARD_MAP  # tier weight per achievement: 1, 3, 5 or 8
).sum()
health_reward = (state.player_health - init_health) * 0.1
reward = achievement_reward + health_reward
```

- `state` is the state after `clip_inventory_and_intrinsics`, so health is already clipped to [0, 8 + strength].
- Achievements never turn false, so the first term pays each achievement once, in the step it first becomes true.
- `step_env` returns this reward unchanged: no clipping, scaling or discount ([craftax_symbolic_env.py L56](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/envs/craftax_symbolic_env.py#L56)).
- Over an episode the health terms add up to 0.1 x (final health - 9), the start health: between -0.9 (death) and +0.4 (health 13 at strength 5).
- So an episode return is the achievement total plus a term in [-0.9, +0.4]; Craftax-Classic (start and maximum health 9): [-0.9, 0].

| Variant | Achievement reward | Health term | Source |
|---|---|---|---|
| Craftax | 1 / 3 / 5 / 8 by tier (basic / intermediate / advanced / very advanced), max 226 | +-0.1 per health point | [constants.py `achievement_mapping`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L523) |
| Craftax-Classic | +1 per achievement, 22 achievements, max 22 | +-0.1 per health point | [classic game_logic.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py#L1655) |

- Each achievement pays once per episode (first unlock).
- Tiers were introduced because a flat +1 per achievement gave too little incentive to enter the dungeons ([paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3)).
- Health is clipped at 0 before the reward is computed, so death costs at most the remaining health x 0.1; in Classic this was fixed in v1.6.0 (see [[Craftax known issues]]).
- The full list of 67 achievements, tiers and triggers: [[Craftax achievements]].
- Per-achievement reward only (for example `EAT_COW`) is not built in; the maintainer suggests editing `achievement_mapping` or a reward wrapper ([issue #37](https://github.com/MichaelTMatthews/Craftax/issues/37)).

## Reported metric

- Craftax results are reported as episode reward as a percentage of the maximum 226 ([paper §4.3](https://arxiv.org/html/2402.16801v2#S4.SS3); README Scoreboard).
- Achievement "success rate" = fraction of finished episodes in which the achievement was unlocked ([paper Fig. 4, 5](https://arxiv.org/html/2402.16801v2#S4.SS3)).
- Episode return includes the health term, so % of 226 is return / 226, not achievements only.

## Benchmarks

| Benchmark | Budget | Env | Purpose | Source |
|---|---|---|---|---|
| Craftax-1B | 1e9 environment interactions | `Craftax-Symbolic-v1` | exploration, continual learning, long-term planning | [paper §3.4](https://arxiv.org/html/2402.16801v2#S3.SS4) |
| Craftax-1M | 1e6 environment interactions | `Craftax-Symbolic-v1` | sample efficiency | [paper §3.4](https://arxiv.org/html/2402.16801v2#S3.SS4) |

- Both benchmarks use symbolic observations only.
- Craftax-Classic is kept "mostly as a tool" for quick iteration, not as a benchmark.
- The authors equate 1B steps at about 5 steps per second to over 6 years of human play.
- Episodes are capped at 100,000 steps (`max_timesteps`; [paper App. D](https://arxiv.org/html/2402.16801v2#A4)); hitting the cap returns `done=True` with `discount=0`, the same as death.

## Scoreboard

Reward as % of 226, from the [README](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md); entries from outside the paper are "reported and have not been verified".

| Algorithm | Craftax-1B | Craftax-1M |
|---|---|---|
| PPO-GTrXL | 18.3 | |
| PQN-RNN | 16.0 | |
| PPO-RNN | 15.3 | 2.3 |
| RND | 12.0 | 2.2 |
| PPO | 11.9 | 2.2 |
| ICM | 11.9 | 2.2 |
| E3B | 11.0 | 2.2 |
| Simulus | | 6.6 |
| Efficient MBRL | | 5.4 |

- Paper findings on Craftax-1B (10 seeds each): PPO and PPO-RNN master most basic achievements; no method makes progress in the advanced tiers; intrinsic-reward methods do not beat PPO, E3B lowers reward ([paper §4.3](https://arxiv.org/html/2402.16801v2#S4.SS3)).
- PPO-RNN for 10B steps (4 seeds) barely improves on 1B and does not reach the gnomish mines ([paper App. F.2](https://arxiv.org/html/2402.16801v2#A6.SS2)).
- UED on Craftax-1B (DR, PLR, robust PLR, ACCEL with noise / swap / restricted-swap mutations, via JaxUED): only PLR clearly beats domain randomisation ([paper §4.5](https://arxiv.org/html/2402.16801v2#S4.SS5)).

## Baseline hyperparameters

PPO on Craftax-1B ([paper Table 7](https://arxiv.org/html/2402.16801v2#A5.SS1)); the same values are the defaults of [Craftax_Baselines ppo.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa/ppo.py).

| Hyperparameter | Craftax-1B | Craftax-1M (Table 11) |
|---|---|---|
| Environment workers | 1024 | 256 |
| Steps per rollout | 64 | 16 |
| Minibatches | 8 | 8 |
| Update epochs | 4 | 4 |
| Learning rate | 0.0002, annealed | 0.0003, annealed |
| gamma | 0.99 | 0.99 |
| GAE lambda | 0.8 | 0.8 |
| Clip epsilon | 0.2 | 0.2 |
| Entropy coefficient | 0.01 | 0.01 |
| Value coefficient | 0.5 | 0.5 |
| Network | 4-layer MLP, width 512, tanh | width 512, tanh |

- Policy and value use flattened symbolic observations; convolutions on the map view did worse than MLPs ([paper §4.1](https://arxiv.org/html/2402.16801v2#S4.SS1)).
- `ppo.py` also sets `max_grad_norm=1.0`, `layer_size=512`, `total_timesteps=1e9`.
- PPO-RNN uses a GRU with hidden size equal to the MLP width; ICM, E3B and RND settings are in paper Tables 8-10 and 12-13.

## Logging in code

- Craftax `info["Achievements/<name>"]` = 100.0 x unlocked, non-zero only on terminal steps ([envs/common.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/envs/common.py)).
- Average success rate per achievement = mean of that key over steps where `info["returned_episode"]` is true, as done in `ppo.py` ([ppo.py L449](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa/ppo.py#L449)).
- `logz/batch_logging.py` also logs `achievements` = sum over keys / 100, the mean number of achievements per episode ([batch_logging.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa/logz/batch_logging.py)).
- Craftax-Classic adds `info["score"] = exp(mean(log(1 + achievements))) - 1` over the 0/100 per-achievement values of the finished episode ([classic envs/common.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/envs/common.py), commented "Geometric mean with an offset of 1%").
- Each term of that `score` is 0 or 100, so the mean of per-episode scores is not the formula applied to averaged success rates; for a score over success rates, average the `Achievements/*` keys over episodes first.
