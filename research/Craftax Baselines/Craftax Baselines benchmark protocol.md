---
title: "Craftax Baselines benchmark protocol"
type: benchmark
parent: "[[Craftax Baselines]]"
tags:
  - craftax
  - benchmark
  - evaluation
  - protocol
source: "https://arxiv.org/pdf/2402.16801v2"
created: 2026-10-20
---
# Craftax Baselines benchmark protocol

> [!abstract] Summary
> The paper defines two benchmarks, Craftax-1B and Craftax-1M, both on `Craftax-Symbolic-v1`; the metric is episode reward as a percentage of the maximum 226. Pixels and Craftax-Classic are not part of the benchmark. Baselines use 10 seeds and report training-episode returns.

## Environments

| Environment | Observation | Actions | Achievements | Max reward | Max episode length | In benchmark |
|---|---|---|---|---|---|---|
| `Craftax-Symbolic-v1` | flat 8268 | 43 | 67 (Table 6) | 226 | 100,000 | yes |
| `Craftax-Pixels-v1` | 110x130x3 | 43 | 67 | 226 | 100,000 | no |
| `Craftax-Classic-Symbolic-v1` | flat 1345 | 17 | 22 | 22 | 10,000 | no |
| `Craftax-Classic-Pixels-v1` | 63x63x3 | 17 | 22 | 22 | 10,000 | no |

- Observation and action sizes: paper Section 3.3 ([arXiv v2](https://arxiv.org/pdf/2402.16801v2)).
- Max reward 226 (Craftax) and 22 (Classic): maintainer in [issue #1](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/1); 226 also in paper Section 4.3.
- Max episode length: paper Appendix D (100,000, then truncation); code `max_timesteps` is 100000 in [craftax_state.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/craftax/craftax/craftax_state.py) and 10000 in [craftax_classic/envs/craftax_state.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/craftax/craftax_classic/envs/craftax_state.py).
- Symbolic map view: 9x11 tiles in Craftax, 7x9 in Classic; each tile is one-hot block (37), item (5), creature (36) plus light level (Appendix D.4).
- Symbolic environments run about 10x faster than pixel ones (Section 3.1).
- Craftax-Classic is kept "mostly as a tool to help researchers iterate quickly" (Section 3.4).

## Reward

- Each achievement pays once per episode, the first time it is completed (Section 3.3).
- Health: -0.1 per point of damage, +0.1 per point recovered (Section 3.3).
- Craftax achievement weights by category: Basic 1, Intermediate 3, Advanced 5, Very Advanced 8 (Section 3.3).
- Classic: flat +1 per achievement, 22 achievements (Section 2.2, issue #1), plus the same 0.1 x health-change term ([craftax_classic/game_logic.py L1695-L1700](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/craftax/craftax_classic/game_logic.py#L1695-L1700)).
- Because of the health term, episode return can differ from the achievement score; the maximum 226 (or 22) counts achievements only.

| Category | Reward each | Count (Table 6) | Total |
|---|---|---|---|
| Basic | 1 | 25 | 25 |
| Intermediate | 3 | 18 | 54 |
| Advanced | 5 | 15 | 75 |
| Very Advanced | 8 | 9 | 72 |
| All | | 67 | 226 |

- Counts and totals derived from paper Appendix D.6, Table 6 (IDs 0-66); they sum to the stated 226.
- Section 3.3 text says "65 in Craftax"; Table 6 lists 67.
- Classic achievements are Table 6 IDs 0-21 (Appendix D.6).

## The two benchmarks

| Benchmark | Environment | Budget | Purpose (paper Section 3.4) |
|---|---|---|---|
| Craftax-1B | `Craftax-Symbolic-v1` | 1,000,000,000 env interactions | Exploration, continual learning, long-term planning, without sample-efficiency pressure |
| Craftax-1M | `Craftax-Symbolic-v1` | 1,000,000 env interactions | Sample efficiency; runs take seconds |

- Both benchmarks are restricted to symbolic observations because they are faster and pixels "do not contribute anything significant" to the target research directions (Section 3.4).
- Budget counts environment steps summed over all parallel workers (`total_timesteps` in the code).
- Derived from code: integer division of the budget by steps per update leaves 15,258 updates x 65,536 = 999,948,288 steps for 1B defaults and 244 x 4,096 = 999,424 steps for the 1M config.
- The paper notes that even at 1M, massive parallelism worked better (Section 3.4, Table 11).

## Metric

- "Reward as a percentage of the maximum achievable reward (obtained by completing every achievement) of 226" (Section 4.3).
- The Craftax README scoreboard uses the same metric: "We report reward as a % of the maximum (226)" ([README, Scoreboard](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/README.md)).
- Achievement success rate: the % of episodes in which an achievement is unlocked, shown per difficulty category (Figure 4), for selected achievements (Figure 5) and for all achievements (Figures 19-20).
- In code, `Achievements/<name>` is `achievement unlocked x done x 100` from the environment, averaged over episodes that ended in the update ([craftax/envs/common.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/craftax/craftax/envs/common.py), [ppo.py L448-L452](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L448-L452)).
- Returns are those of training episodes under the stochastic training policy; the scripts have no separate evaluation loop.
- The paper describes no separate evaluation for the PPO, PPO-RNN, ICM, E3B and RND runs; only the UED runs are evaluated on 20 fixed levels (Section 4.5).
- How the single scoreboard number is aggregated (final logged value, mean over seeds) is not stated; by visual reading the README values match the end points of Figures 3 and 6.

## Seeds and error bars

| Experiment | Seeds | Error bars | Source |
|---|---|---|---|
| Craftax-1B, 5 baselines | 10 per algorithm | 1 standard error | Figures 3-5 captions |
| Craftax-1M, 5 baselines | 10 per algorithm | 1 standard error | Figure 6 caption |
| PPO-RNN, Craftax 10B steps | 4 | 1 standard error | Figures 21-22 captions |
| PPO-RNN, Craftax-Classic 1B | 10 | 1 standard error | Figures 23-24 captions |
| UED on Craftax-1B | 10 runs, 20 evaluation levels | 1 standard error | Section 4.5, Figure 7 |
| Hyperparameter tuning, 1B | 1 per configuration | none | Appendix E.1 |

- The seed values themselves are not published; the scripts draw a random seed when `--seed` is omitted.

## Scoreboard rules

- New entries arrive as PRs to the Craftax README with "a reference to the source of the results" ([README, Scoreboard](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/README.md)).
- "All scores from outside the original Craftax paper are reported and have not been verified" (same section).
- No rule fixes the number of seeds, evaluation mode, observation type check or compute limit beyond the step budget.
- Craftax version is not part of the protocol, although the README errata list reward-affecting fixes in v1.5.0 and v1.6.0 ([[Craftax Baselines reproduction notes]]).

## Our comparison checklist

- Train on `Craftax-Symbolic-v1` with exactly 1e9 or 1e6 env steps summed over workers.
- Report reward as % of 226 alongside absolute return, with seed count and standard error.
- State whether the number is the training-episode return (as in the baselines) or a separate evaluation.
- Log per-achievement success rates with the same `Achievements/<name>` semantics to compare with Figures 4, 5, 19 and 20.
- Record the Craftax version used.
