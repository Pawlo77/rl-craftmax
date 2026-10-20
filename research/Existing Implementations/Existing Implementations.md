---
title: "Existing Implementations"
type: index
parent: "[[wut_murl_craftax]]"
tags:
  - index
  - craftax
  - benchmark
  - implementations
created: 2026-10-20
---
# Existing Implementations

Published agents with results on Craftax or Craftax-Classic, one note each, as the benchmark we compare our agents against. How to run the official baselines and the evaluation protocol: [[Craftax Baselines]].

## Comparison table

Score = reward as % of the maximum (226 for Craftax, 22 for Craftax-Classic). Only rows inside the same block are comparable. "n.r." = not reported. ± is given exactly as the source gives it.

**A. Craftax, Symbolic, 1B steps (Craftax-1B challenge): training-episode return**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PPO-GTrXL (Hamon 2024)]] | Craftax | Symbolic | 1B | 18.3 | n.r. (5 per-seed files in repo) | 1x A100 (2x V100 similar) | ~6.5 h | yes, MIT |
| [[PQN-RNN (Gallici 2025)]] | Craftax | Symbolic | 1B | 16.0 | 10 | 1x A40 | n.r. ("similar" to PPO-RNN) | yes, Apache-2.0 |
| [[PPO-RNN (Matthews 2024)]] | Craftax | Symbolic | 1B | 15.3 | 10 | RTX 4090 (paper's machine) | n.r. per run; "a run ... can finish in an hour" | yes, MIT |
| [[RND (Matthews 2024)]] | Craftax | Symbolic | 1B | 12.0 (code later fixed, PR #5) | 10 | RTX 4090 (paper's machine) | n.r. | yes, MIT |
| [[PPO (Matthews 2024)]] | Craftax | Symbolic | 1B | 11.9 | 10 | RTX 4090 (paper's machine) | n.r. per run; "a run ... can finish in an hour" | yes, MIT |
| [[ICM (Matthews 2024)]] | Craftax | Symbolic | 1B | 11.9 (author: paper's ICM results wrong) | 10 | RTX 4090 (paper's machine) | n.r. | yes, MIT |
| [[E3B (Matthews 2024)]] | Craftax | Symbolic | 1B | 11.0 | 10 | RTX 4090 (paper's machine) | n.r. | yes, MIT |

**B. Craftax, Symbolic, 1B steps, UED study: evaluation on 20 fixed levels (not comparable with block A)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PLR (Matthews 2024)]] | Craftax | Symbolic | 1B | n.r. (Fig. 7 only); best UED method | 10 | n.r. | n.r. | yes, JaxUED, Apache-2.0 |
| [[Domain Randomisation (Matthews 2024)]] | Craftax | Symbolic | 1B | n.r.; second | 10 | n.r. | n.r. | yes, JaxUED |
| [[Robust PLR (Matthews 2024)]] | Craftax | Symbolic | 1B | n.r.; slightly above DR, below PLR | 10 | n.r. | n.r. | yes, JaxUED |
| [[ACCEL (Matthews 2024)]] (RSwap, Noise) | Craftax | Symbolic | 1B | n.r.; about DR level | 10 | n.r. | n.r. | yes, JaxUED |
| [[ACCEL (Matthews 2024)]] (Swap) | Craftax | Symbolic | 1B | n.r.; worst | 10 | n.r. | n.r. | yes, JaxUED |

**C. Craftax, Symbolic, more than 1B steps (not comparable with the 1B challenge)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PPO-GTrXL (Hamon 2024)]] | Craftax | Symbolic | 4B | 20.6 | n.r. | n.r. | n.r. | yes, MIT |
| [[PPO-GTrXL (Hamon 2024)]] | Craftax | Symbolic | 8B | n.r. ("no significant improvement" over 4B) | n.r. | n.r. | n.r. | yes, MIT |
| [[PPO-RNN (Matthews 2024)]] | Craftax | Symbolic | 10B | n.r. ("barely improves" over 1B) | 4 | n.r. | n.r. | yes, MIT |

**D. Craftax, Symbolic, 1M steps (Craftax-1M challenge)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[Simulus (Cohen 2025)]] | Craftax | Symbolic | 1M | 6.59 (sd 0.49, computed from repo per-seed file); 100 test episodes | 5 | 1x RTX 4090 | ~94 h | yes, GPL-3.0 |
| [[PPO-RNN (Matthews 2024)]] | Craftax | Symbolic | 1M | 2.3 (third-party rerun: 2.13) | 10 | n.r. | n.r. | yes, MIT |
| [[RND (Matthews 2024)]] | Craftax | Symbolic | 1M | 2.2 (third-party rerun: 2.11) | 10 | n.r. | n.r. | yes, MIT |
| [[PPO (Matthews 2024)]] | Craftax | Symbolic | 1M | 2.2 | 10 | n.r. | n.r. | yes, MIT |
| [[ICM (Matthews 2024)]] | Craftax | Symbolic | 1M | 2.2 (author: wrong, buggy commit) | 10 | n.r. | n.r. | yes, MIT |
| [[E3B (Matthews 2024)]] | Craftax | Symbolic | 1M | 2.2 (third-party rerun: 1.91) | 10 | n.r. | n.r. | yes, MIT |

**E. Craftax, Pixels, 1M steps (different observation from block D)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[Efficient MBRL (Dedieu 2025)]] (MBRL, slow) | Craftax | Pixels | 1M | 7.20 ± 0.09 s.e. (v3); 5.44 ± 0.25 s.e. in v1 = scoreboard 5.4 | 10 | 8x H100 | n.r. | no |
| [[Efficient MBRL (Dedieu 2025)]] (MFRL) | Craftax | Pixels | 1M | 4.63 ± 0.20 s.e. | 10 | 8x H100 | n.r. | no |

**F. Craftax-Classic, Pixels, 1M steps (Crafter-style protocol)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[Efficient MBRL (Dedieu 2025)]] (MBRL, slow) | Craftax-Classic | Pixels | 1M | 69.66 ± 1.20 s.e. | 10 | 8x H100 | 2749 min | no |
| [[Efficient MBRL (Dedieu 2025)]] (MBRL, fast) | Craftax-Classic | Pixels | 1M | 67.42 ± 0.55 s.e. | 10 | 8x H100 | 759 min | no |
| [[Efficient MBRL (Dedieu 2025)]] (MFRL) | Craftax-Classic | Pixels | 1M | 55.49 ± 1.33 s.e. | 10 | 1x A100 (per §3.2) | 15 min | no |

**G. Craftax-Classic, Symbolic, 1B steps**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PPO-RNN (Matthews 2024)]] | Craftax-Classic | Symbolic | 1B | about 90 (text; curve in App. F.3) | 10 | single GPU | 51 min | yes, MIT |

Sources for every cell are in the linked notes.

## How to read the table

- Like with like: compare only within one block; environment, observation type, step budget and metric must all match.
- Block A is the closest target for our agents: the scoreboard values are the README's single numbers, a mean over 10 seeds for the Craftax-paper rows, with no spread given ([Craftax README](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [Craftax paper](https://arxiv.org/abs/2402.16801) Fig. 3).
- The scoreboard states that scores from outside the Craftax paper are "reported and have not been verified" (PPO-GTrXL, PQN-RNN, Simulus, Efficient MBRL).
- Metric differs between sources: Craftax-paper baselines log returns of training episodes with the stochastic policy; Simulus scores 100 test episodes at the end with sampling temperature 0.5; the UED block scores checkpoints on 20 fixed levels; PQN-RNN, PPO-GTrXL and Efficient MBRL do not state their protocol.
- Score in Efficient MBRL means two things: "reward %" (used in the table) and a separate "score %" (geometric mean of achievement success rates); the table uses reward % throughout.
- Spread: "s.e." is standard error over seeds; Simulus's sd is computed in this pass from the per-seed file in its repository; everything else has no spread.
- ICM numbers are unreliable: the author states they came from a buggy commit ([Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4)).
- RND numbers predate a fix to the intrinsic-value bootstrap ([Baselines PR #5](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/5), May 2025).
- The third-party reruns in block D are the Simulus authors' runs of the Craftax baselines ([Simulus paper](https://arxiv.org/abs/2502.11537) Fig. 1), slightly below the scoreboard values.
- Environment versions differ and are mostly not reported; Craftax errata change rewards before v1.5.0 and v1.6.0 ([Craftax README](https://github.com/MichaelTMatthews/Craftax#-errata)), and open bugs affect Craftax-Classic mob handling ([issue #57](https://github.com/MichaelTMatthews/Craftax/issues/57), [issue #60](https://github.com/MichaelTMatthews/Craftax/issues/60)).
- Wall-clock and hardware are rarely per run: the Craftax paper gives only benchmark-level statements, so "n.r." in those cells means no per-algorithm number exists.

## Sources scanned

This pass covered only implementations mentioned in the two Craftax repositories and the paper they cite.

- Craftax repository README at commit `c3c2e0d` (2026-06-20): scoreboard (Craftax-1B, Craftax-1M), Experiment, Gotchas, Errata, See also, Citation ([repo](https://github.com/MichaelTMatthews/Craftax)); `MAINTENANCE.md`, `CONTRIBUTING.md`, `tutorial.md` contain no algorithms.
- Craftax paper, arXiv:2402.16801 v2 (3 June 2024, ICML 2024): §4 baselines, App. E hyperparameters, App. F extra results, App. G UED.
- Craftax_Baselines repository at commit `7ce36fa` (2026-06-16): README, `ppo.py`, `ppo_rnn.py`, `ppo_rnd.py`, `models/`, `wrappers.py`.
- Craftax "See also" list: Crafter, NLE, PureJaxRL, JaxUED, Minimax, Gymnax, JaxMARL; none reports Craftax results except JaxUED, whose `examples/craftax/craftax_plr.py` implements the paper's UED runs.
- Primary sources of the scoreboard entries: transformerXL_PPO_JAX README and issues; PQN paper arXiv:2407.04811 v6 and purejaxql repository; Simulus paper arXiv:2502.11537 v4 and repository; Efficient MBRL paper arXiv:2502.01591 v1, v2, v3.
- GitHub issues read: all issues of Craftax_Baselines (#1, #2, #4, #7, #11); Craftax #19, #34, #37, #50, #53, #57, #60; transformerXL_PPO_JAX #2; purejaxql #9, #11, #12; Simulus #1, #2.
- Craftax pull requests (full list) checked to trace scoreboard edits: #23 (PPO-GTrXL added), #36 and #44 (M3 added, renamed Simulus).
- Craftax_Baselines pull requests (full list): #5 (RND GAE fix), #8 (DQN, closed), #10 (JAX/Orbax API update, open).

## Gaps

- Pending or rejected scoreboard candidates not covered here: PPO-LSTM from "modrax" ([Craftax PR #54](https://github.com/MichaelTMatthews/Craftax/pull/54), open), maskedPPO-GTrXL with reward shaping ([PR #52](https://github.com/MichaelTMatthews/Craftax/pull/52), closed), DQN for Craftax-Classic ([PR #59](https://github.com/MichaelTMatthews/Craftax/pull/59), closed), DQN in Craftax_Baselines ([Baselines PR #8](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/8), closed).
- Methods that appear only as comparison points in the scanned papers, without notes here: DreamerV3, IRIS, Δ-IRIS, Curious Replay, the Moon et al. 2024 PPO baseline (Craftax-Classic or Crafter, from [[Efficient MBRL (Dedieu 2025)]] Table 1); REM, TWM, STORM, DIAMOND (Atari or DMC only in Simulus).
- Results on the original Crafter benchmark are out of scope for this pass.
- Numbers shown only in figures: UED rewards, PQN (MLP) on Craftax-1B, PPO-RNN at 10B steps.
- Corrected ICM results: not published.
- Per-run wall-clock and parameter counts for the Craftax-paper baselines and PQN-RNN: not reported.
- Seed count and evaluation protocol for PPO-GTrXL: not stated.
- Efficient MBRL code: not found.
- Simulus-A (repository default since August 2026): no Craftax results yet.
- Craftax version used by each result: not reported in any source.
