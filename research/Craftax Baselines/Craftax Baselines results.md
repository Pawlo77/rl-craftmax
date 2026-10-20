---
title: "Craftax Baselines results"
type: benchmark
parent: "[[Craftax Baselines]]"
tags:
  - craftax
  - benchmark
  - results
  - baselines
source: "https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/README.md"
created: 2026-10-20
---
# Craftax Baselines results

> [!abstract] Summary
> The paper reports results as curves only (Figures 3-8, 19-24); the exact numbers are in the Craftax README scoreboard. Best baseline: PPO-RNN with 15.3% of max on Craftax-1B and 2.3% on Craftax-1M. The ICM numbers are known to be wrong and the RND numbers predate a GAE bug fix.

## Headline scores

Reward as % of the maximum 226 on `Craftax-Symbolic-v1`, 10 seeds per algorithm.

| Algorithm | Craftax-1B (% max) | Craftax-1M (% max) | Script | Status |
|---|---|---|---|---|
| PPO-RNN | 15.3 | 2.3 | `ppo_rnn.py` | as published |
| RND | 12.0 | 2.2 | `ppo_rnd.py` | produced before the RND GAE fix ([PR #5](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/5)) |
| PPO | 11.9 | 2.2 | `ppo.py` | as published |
| ICM | 11.9 | 2.2 | `ppo.py --train_icm` | wrong per maintainer ([issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4)) |
| E3B | 11.0 | 2.2 | `ppo.py --use_e3b` | 1M reproduced lower by a third party (1.9) |

- Source: Craftax README, "Scoreboard", Craftax-1B and Craftax-1M tables ([README at c3c2e0d](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/README.md)).
- The five baseline rows have not changed since the RND row was added on 2024-04-30 ([commit c985ff6](https://github.com/MichaelTMatthews/Craftax/commit/c985ff6)); PPO, PPO-RNN, ICM and E3B values were already on the board before that.
- Seeds and error bars: Figures 3 and 6 captions of the [paper](https://arxiv.org/pdf/2402.16801v2), 10 seeds, shaded 1 standard error.
- RND first appears in paper v2 (June 2024); v1 (February 2024) shows PPO, PPO-RNN, ICM and E3B only.

## Absolute returns (derived)

% of max x 226 / 100; computed here, not printed in the sources.

| Algorithm | Craftax-1B | Craftax-1M |
|---|---|---|
| PPO-RNN | 34.58 | 5.198 |
| RND | 27.12 | 4.972 |
| PPO | 26.89 | 4.972 |
| ICM | 26.89 | 4.972 |
| E3B | 24.86 | 4.972 |

- The 4.972 for E3B at 1M matches the "reported score" quoted in [issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4).

## Other entries on the same scoreboard

Reported by their authors, "not verified" per the README; details in [[Existing Implementations]].

| Algorithm | Craftax-1B (% max) | Craftax-1M (% max) | Code linked on the scoreboard |
|---|---|---|---|
| PPO-GTrXL | 18.3 | | [TransformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX) |
| PQN-RNN | 16.0 | | [purejaxql](https://github.com/mttga/purejaxql/) |
| Simulus | | 6.6 | [Simulus](https://github.com/leor-c/Simulus) |
| Efficient MBRL | | 5.4 | none |

- Source: same README scoreboard; Simulus was first listed as M3 (renamed 2025-05-26).

## Craftax-1B findings (paper Section 4.3)

- PPO and PPO-RNN quickly master most Basic achievements; memory pushes PPO-RNN further into Intermediate.
- None of ICM, E3B or RND improved on PPO; E3B "significantly reduced the reward".
- Authors' hypothesis: the reward is dense enough without intrinsic reward, which may distract the agent.
- No agent makes progress in the Advanced or Very Advanced categories (Figure 4).
- All methods learn `make_wood_pickaxe` quickly (Figure 5).
- PPO-RNN enters the dungeon significantly more often than the others, which drives its score.
- `defeat_zombie` success falls over training, most for PPO-RNN, as stronger agents trade overworld achievements for underground ones.
- The gnomish mines (floor 2) are reached only very rarely.
- `eat_plant` is ignored by PPO-RNN and achieved best by E3B.
- Per-achievement curves for all 5 algorithms: Figure 19 (1B), Figure 20 (1M).

## Craftax-1M findings (paper Section 4.4)

- Much less separation between algorithms than at 1B; all perform similarly (Figure 6).
- Progress only on very basic achievements, mostly those carried over from Crafter.
- Agents enter the dungeon only on individual occurrences.

## Longer budget and Craftax-Classic

| Run | Result | Source |
|---|---|---|
| PPO-RNN, Craftax, 10B steps, 4 seeds | Barely improves over 1B; still does not reach the gnomish mines | Appendix F.2, Figures 21-22 |
| PPO, Craftax-Classic, 1B steps | Averages 90% of the optimal reward, in under an hour on one GPU | Abstract; Introduction says 51 minutes |
| PPO-RNN, Craftax-Classic, 1B steps, 10 seeds | All achievements except `collect_diamond` and `eat_plant` reliably completed; diamonds somewhat reliable, `eat_plant` very rare | Appendix F.3, Figures 23-24 (curves only) |

- The faster rise at the end of 1B and 10B runs is attributed to the learning-rate schedule, not to continued learning (Section 4.3, Appendix F.2).

## UED results (paper Section 4.5, not in this repo)

- Run with [JaxUED](https://github.com/DramaCow/jaxued), not with Craftax Baselines; hyperparameters in Appendix G.3, Table 14.
- Ranking after 1B steps on 20 fixed evaluation levels: PLR best, then DR, robust PLR, ACCEL restricted swap and ACCEL noise roughly equal; ACCEL unrestricted swap worst (Figure 7).
- ACCEL unrestricted swap collects diamonds more than 30% of the time on replay levels vs 7% on DR levels (Figures 32-33).
- ACCEL restricted swap reaches the gnomish mines 1% of the time on replay levels, almost never on DR levels.
- Values are given only as curves; no exact % of max is published.

## Not available

- No exact numbers for Pixels environments; the paper ran no pixel experiments ([Craftax issue #19](https://github.com/MichaelTMatthews/Craftax/issues/19) maintainer reply).
- No exact Craftax-Classic score per algorithm beyond "90% of optimal" for PPO.
- No per-achievement success-rate tables; only figures.
- No raw logs or W&B runs to recompute numbers or standard errors.

## Caveats on the published numbers

- ICM: the maintainer found the sweep was run on an old commit where the ICM error was multiplied by `done` instead of `1 - done`, so intrinsic reward was paid only at episode termination; "The ICM results in the paper are wrong" ([issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4), 2025-01-09).
- ICM: the maintainer promised an arXiv update; as of 2026-10-20 arXiv still has only v2 (3 Jun 2024) and the README still lists ICM at 11.9 and 2.2.
- ICM: the issue author's plot of the reproduction (5 seeds per algorithm, Craftax-1M) is an image only; its numbers are not given in text.
- E3B 1M: the issue #4 author reports 4.306 (1.9% of max) over 10 seeds vs 4.972 (2.2%) reported, called "a minor discrepancy".
- RND: before [PR #5](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/5) (merged 2025-05-12) intrinsic advantages bootstrapped from the extrinsic value; maintainer: "I hope this won't have had too large an effect on results but will double check"; no updated number has been posted.
- Craftax version: the baseline numbers date from 2024, before Craftax v1.5.0 (fix for the first planted plant never ripening, affecting `eat_plant`) and v1.6.0 (Classic health-reward fixes) ([README, Errata](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/README.md)).
- Metric: training-episode returns of the stochastic policy, not a separate evaluation ([[Craftax Baselines benchmark protocol]]).
