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

Score = reward as % of the maximum (226 for Craftax, 22 for Craftax-Classic). Only rows inside the same block are comparable. "n.r." = not reported. ± is given exactly as the source gives it. Where a source gives raw returns, the % was computed in this pass (raw / 2.26 or raw / 0.22). Rows marked "rerun" are third-party runs of a method whose own note is linked.

**A. Craftax, Symbolic, 1B steps (Craftax-1B challenge): training-episode return**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PPO-GTrXL (Hamon 2024)]] | Craftax | Symbolic | 1B | 18.3 | n.r. (5 per-seed files in repo) | 1x A100 (2x V100 similar) | ~6.5 h | yes, MIT |
| [[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]] (h096, repo only) | Craftax | Symbolic | 1B | 17.9 (h040 without RND: 17.5, sd 1.0) | 2 (h040: 3) | H100 MIG slice or A100 | ~10-13 h | yes, no licence |
| [[PQN-RNN (Gallici 2025)]] | Craftax | Symbolic | 1B | 16.0 | 10 | 1x A40 | n.r. ("similar" to PPO-RNN) | yes, Apache-2.0 |
| [[PPO-RNN (Matthews 2024)]] | Craftax | Symbolic | 1B | 15.3 | 10 | RTX 4090 (paper's machine) | n.r. per run; "a run ... can finish in an hour" | yes, MIT |
| [[RND (Matthews 2024)]] | Craftax | Symbolic | 1B | 12.0 (code later fixed, PR #5) | 10 | RTX 4090 (paper's machine) | n.r. | yes, MIT |
| [[PPO (Matthews 2024)]] | Craftax | Symbolic | 1B | 11.9 | 10 | RTX 4090 (paper's machine) | n.r. per run; "a run ... can finish in an hour" | yes, MIT |
| [[ICM (Matthews 2024)]] | Craftax | Symbolic | 1B | 11.9 (author: paper's ICM results wrong) | 10 | RTX 4090 (paper's machine) | n.r. | yes, MIT |
| [[E3B (Matthews 2024)]] | Craftax | Symbolic | 1B | 11.0 | 10 | RTX 4090 (paper's machine) | n.r. | yes, MIT |
| PPO-LSTM port, rerun ([[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]]) | Craftax | Symbolic | 1B | 15.0 (sd 0.2) | 3 | as above | as above | yes, no licence |
| PPO port, rerun ([[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]]) | Craftax | Symbolic | 1B | 11.9 (sd 0.6) | 3 | as above | as above | yes, no licence |

**A2. Craftax, Symbolic, 1B steps, separate evaluation of the trained policy (same budget as A, different protocol)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PPO-LSTM (Jiralerspong 2026)]] | Craftax | Symbolic | 1B | 19.54 ± 0.34 s.e. (1024 episodes, final checkpoint) | 10 | 1x L40S | ~3h40m | yes, MIT |
| PPO-Hybrid TrXL + S5, unmasked ([[Feasibility Classification (Zabounidis 2026)]]) | Craftax | Symbolic | 1B | 19.4 (raw 43.9 ± 0.4 sd; 10,240 stochastic episodes per seed) | 4 | 1 accelerator, type n.r. | n.r. | no |
| [[Feasibility Classification (Zabounidis 2026)]] (Hybrid, masked + KL, learned masks) | Craftax | Symbolic | 1B | 19.1 (raw 43.2 ± 1.1 sd) | 4 | as above | n.r. | no |
| [[Sable Inference Search (Chalumeau 2025)]] (zero-shot, no search) | Craftax | n.r. | 1B | 18.8 (95% CI 18.1-19.5) | n.r. | 1x A100 80GB | n.r. | Drive files only |
| PPO-GTrXL, rerun ([[PPO-LSTM (Jiralerspong 2026)]]) | Craftax | Symbolic | 1B | 17.79 ± 0.82 s.e. | 10 | 1x L40S | ~8h45m | yes, MIT |
| PPO-RNN (GRU), rerun ([[Feasibility Classification (Zabounidis 2026)]]) | Craftax | Symbolic | 1B | 16.2 (raw 36.7 ± 0.1 sd) | 4 | 1 accelerator | n.r. | no |
| [[SimBa (Lee 2025)]] (PPO + SimBa; protocol of `avg_return` not documented) | Craftax | Symbolic | 1B | 12.8 (sd 0.2, from repo CSV) | 5 | n.r. | n.r. | CSV only; PPO code not released |
| [[RePPO (Nishimori 2026)]] (m = 1.2, no entropy) | Craftax | Symbolic | 1B | 11.87 (sd 0.16; 100 episodes per seed) | 5 | n.r. | n.r. | yes, no licence |
| PPO + entropy, rerun ([[RePPO (Nishimori 2026)]]) | Craftax | Symbolic | 1B | 11.66 (sd 0.33) | 5 | n.r. | n.r. | yes, no licence |
| RND + entropy, rerun ([[RePPO (Nishimori 2026)]]) | Craftax | Symbolic | 1B | 11.68 (sd 0.30) | 5 | n.r. | n.r. | yes, no licence |
| PPO (MLP), rerun ([[Feasibility Classification (Zabounidis 2026)]]) | Craftax | Symbolic | 1B | 11.9 (raw 26.9 ± 0.5 sd) | 4 | 1 accelerator | n.r. | no |
| PPO (MLP), rerun ([[SimBa (Lee 2025)]] repo CSV) | Craftax | Symbolic | 1B | 11.4 (sd 0.9) | 5 | n.r. | n.r. | CSV only |

**A3. Craftax, Symbolic, 1B steps, with prior knowledge or test-time search (not comparable with A or A2)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[maskedPPO-GTrXL (Doguhan58 2026)]] (action mask + reward shaping; best run) | Craftax | Symbolic | 1B | 33.8 best run; typical raw 70 ± 10 (~31) | best of n.r. | RTX 4090 | ~8 h | yes, MIT |
| [[Sable Inference Search (Chalumeau 2025)]] (30 s best-of-attempts search) | Craftax | n.r. | 1B | 25.8 (95% CI 25.7-26.1) | n.r. | 1x A100 80GB | n.r. + 30 s per episode | Drive files only |
| [[Feasibility Classification (Zabounidis 2026)]] (Hybrid, oracle masks at evaluation) | Craftax | Symbolic | 1B | 21.6 (raw 48.8 ± 0.7 sd) | 4 | 1 accelerator | n.r. | no |

**B. Craftax, Symbolic, 1B steps, UED study: evaluation on 20 fixed levels (not comparable with block A)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PLR (Matthews 2024)]] | Craftax | Symbolic | 1B | n.r. (Fig. 7 only); best UED method | 10 | n.r. | n.r. | yes, JaxUED, Apache-2.0 |
| [[Domain Randomisation (Matthews 2024)]] | Craftax | Symbolic | 1B | n.r.; second | 10 | n.r. | n.r. | yes, JaxUED |
| [[Robust PLR (Matthews 2024)]] | Craftax | Symbolic | 1B | n.r.; slightly above DR, below PLR | 10 | n.r. | n.r. | yes, JaxUED |
| [[ACCEL (Matthews 2024)]] (RSwap, Noise) | Craftax | Symbolic | 1B | n.r.; about DR level | 10 | n.r. | n.r. | yes, JaxUED |
| [[ACCEL (Matthews 2024)]] (Swap) | Craftax | Symbolic | 1B | n.r.; worst | 10 | n.r. | n.r. | yes, JaxUED |

**B2. Craftax, Symbolic, 1.07B steps, UED: PACE's 20 held-out levels (different levels and settings from block B)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PACE (Yuan 2026)]] | Craftax | Symbolic | 1.07B | 13.8 (raw 31.198, sd across levels 2.439) | 10 | 1x A100 80GB | n.r. | no (built on JaxUED) |
| Domain Randomisation, rerun ([[PACE (Yuan 2026)]]) | Craftax | Symbolic | 1.07B | 12.3 (raw 27.723 ± 3.839) | 10 | same | n.r. | no |
| PLR, rerun ([[PACE (Yuan 2026)]]) | Craftax | Symbolic | 1.07B | 11.2 (raw 25.380 ± 5.233) | 10 | same | n.r. | no |
| Robust PLR, rerun ([[PACE (Yuan 2026)]]) | Craftax | Symbolic | 1.07B | 10.5 (raw 23.780 ± 4.688) | 10 | same | n.r. | no |
| ACCEL, rerun ([[PACE (Yuan 2026)]]) | Craftax | Symbolic | 1.07B | 10.0 (raw 22.661 ± 4.005) | 10 | same | n.r. | no |

**C. Craftax, Symbolic, more than 1B steps, training-episode return (not comparable with the 1B challenge)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PPO-GTrXL (Hamon 2024)]] | Craftax | Symbolic | 4B | 20.6 | n.r. | n.r. | n.r. | yes, MIT |
| [[PPO-GTrXL (Hamon 2024)]] | Craftax | Symbolic | 8B | n.r. ("no significant improvement" over 4B) | n.r. | n.r. | n.r. | yes, MIT |
| [[PPO-RNN (Matthews 2024)]] | Craftax | Symbolic | 10B | n.r. ("barely improves" over 1B) | 4 | n.r. | n.r. | yes, MIT |

**C2. Craftax, Symbolic, about 2B steps, evaluation on 1024 fixed worlds with an 8192-step cap (DiCode protocol)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[DiCode (Mitsides 2026)]] (steps include generated levels) | Craftax | Symbolic | 2B | 21.5 (raw 48.55 ± 0.49 s.e.) | 8 | RTX 6000 Ada + 4x RTX 6000 Blackwell for the LLM | ~32.2 h | yes, MIT |
| [[AHC (Zhu 2026)]] (curriculum start states) | Craftax | Symbolic | 2.005B | 20.66 (raw 46.70 ± 0.57 sample sd) | 3 | 1x L40S | ~17.6 h | no |
| PPO-GTrXL, rerun ([[AHC (Zhu 2026)]]) | Craftax | Symbolic | 2.005B | 18.5 (raw 41.91 ± 1.01 sample sd) | 3 | RTX A6000 | ~23 h (69.96 GPU-h / 3) | no |
| PPO-GTrXL, rerun ([[DiCode (Mitsides 2026)]]) | Craftax | Symbolic | 2B | 18.4 (raw 41.59) | 8 | RTX 6000 Ada | ~9 h | yes, MIT |
| [[AHC (Zhu 2026)]] | Craftax | Symbolic | 0.996B | 17.1 (raw 38.69 ± 2.26) | 3 | 1x L40S | n.r. | no |
| PPO-GTrXL, rerun ([[AHC (Zhu 2026)]]) | Craftax | Symbolic | 0.996B | 16.2 (raw 36.63 ± 1.42) | 3 | RTX A6000 | n.r. | no |

**C3. Craftax, Symbolic, 500M steps (sampled evaluation, all methods action-masked)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[Concept Grounding (Adamski 2026)]] | Craftax | Symbolic | 500M | 11.87 ± 0.18 sd (max 26.42) | 20 | GPU n.r. | ~6.2 h | yes, no licence |
| PPO baseline, rerun ([[Concept Grounding (Adamski 2026)]]) | Craftax | Symbolic | 500M | 11.62 ± 0.13 sd | 20 | GPU n.r. | ~4.6 h | yes, no licence |

**D. Craftax, Symbolic, 1M steps (Craftax-1M challenge)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[EASimulus (Peng 2026)]] | Craftax | Symbolic | 1M | 7.23 (sd 0.19, computed from repo per-run file) | 5 (repo) | 1x RTX 4090 | ~7 days | yes, GPL-3.0 |
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
| [[ITC (Kim 2026)]] | Craftax | Pixels | 1M | 7.09 ± 0.20 s.e. | 5 | 1x RTX 3090 | n.r. | yes, MIT |
| [[Efficient MBRL (Dedieu 2025)]] (MFRL) | Craftax | Pixels | 1M | 4.63 ± 0.20 s.e. | 10 | 8x H100 | n.r. | no |

**F. Craftax-Classic, Pixels, 1M steps (Crafter-style protocol)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[ITC (Kim 2026)]] | Craftax-Classic | Pixels | 1M | 72.46 ± 0.45 s.e. | 10 | 1x RTX 3090 | 48.2 h | yes, MIT |
| [[Efficient MBRL (Dedieu 2025)]] (MBRL, slow) | Craftax-Classic | Pixels | 1M | 69.66 ± 1.20 s.e. | 10 | 8x H100 | 2749 min | no |
| Efficient MBRL (fast), rerun with ITC hyperparameters ([[ITC (Kim 2026)]]) | Craftax-Classic | Pixels | 1M | 68.55 ± 0.72 s.e. | 10 | 1x RTX 3090 | 46.3 h | yes, MIT |
| Efficient MBRL (fast), rerun with ITC code ([[ITC (Kim 2026)]]) | Craftax-Classic | Pixels | 1M | 68.14 ± 0.42 s.e. | 10 | 1x RTX 3090 | n.r. | yes, MIT |
| [[Efficient MBRL (Dedieu 2025)]] (MBRL, fast) | Craftax-Classic | Pixels | 1M | 67.42 ± 0.55 s.e. | 10 | 8x H100 | 759 min | no |
| [[Efficient MBRL (Dedieu 2025)]] (MFRL) | Craftax-Classic | Pixels | 1M | 55.49 ± 1.33 s.e. | 10 | 1x A100 (per §3.2) | 15 min | no |
| [[Moon PPO Baseline (Moon 2023)]] (Dedieu's reimplementation) | Craftax-Classic | Pixels | 1M | 47.40 ± 0.58 s.e. | n.r. for this row | n.r. | 26 min | original repo (Crafter), MIT |
| [[DreamerV3 (Hafner 2023)]] (Dedieu's rerun, 201M parameters) | Craftax-Classic | Pixels | 1M | 47.18 ± 3.88 s.e. | n.r. for this row | n.r. | 2100 min | author code, MIT |

**G. Craftax-Classic, Symbolic, 1B steps**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[PPO-RNN (Matthews 2024)]] | Craftax-Classic | Symbolic | 1B | about 90 (text; curve in App. F.3) | 10 | single GPU | 51 min | yes, MIT |
| [[Feasibility Classification (Zabounidis 2026)]] (RNN, masked + KL, learned masks; 10,240 stochastic episodes) | Craftax-Classic | Symbolic | 1B | 92.7 (raw 20.4 ± 0.1 sd; oracle masks 93.2) | 4 | 1 accelerator | n.r. | no |
| PPO-RNN (GRU), rerun ([[Feasibility Classification (Zabounidis 2026)]]) | Craftax-Classic | Symbolic | 1B | 90.0 (raw 19.8 ± 0.0) | 4 | 1 accelerator | n.r. | no |
| PPO (MLP) and PPO-Hybrid, reruns ([[Feasibility Classification (Zabounidis 2026)]]) | Craftax-Classic | Symbolic | 1B | 87.3 each (raw 19.2) | 4 | 1 accelerator | n.r. | no |

**H. Craftax-Classic, other budgets or metrics (not comparable with F or G, nor with each other)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[SGRL (Qi 2025)]] (LLM-written goals and masks) | Craftax-Classic | n.r. (repo default Pixels) | 10M / 5M | 67.7 / 59.1 (raw 14.9 ± 0.4 / 13.0 ± 0.3); Crafter score 43.9 / 33.8 | 5 | 1x RTX 4090 | n.r. | yes, no licence |
| PPO-RNN, rerun with survival penalty (CrafterDojo, see [[PPO-RNN (Matthews 2024)]]) | Craftax-Classic | Symbolic | 10B | 98.4 ± 0.04; Crafter score 97.5 | n.r. | n.r. | n.r. | yes, licence "Other" |
| [[Rule Learner Planner (dargilboa 2026)]] (reward-free, repo only) | Craftax-Classic | Symbolic | 1M | n.r.; Crafter score of own episodes 11.2 | 3 | CPU | n.r. | yes, MIT |
| [[DLSD (Rajaraman 2025)]] (hybrid reward; 1K-step episode cap) | Craftax-Classic | Symbolic | 2M | n.r.; Crafter score 3.72 (reward-free 2.27) | 5 | RTX A6000 | 5-10 h per seed | yes, no licence |

**I. Other metrics and settings (not comparable with any block above)**

| Implementation | Environment | Observation | Step budget | Score (% max) | Seeds | Hardware | Wall-clock | Code |
|---|---|---|---|---|---|---|---|---|
| [[SCALAR (Zabounidis 2026)]] (target-achievement success, LLM skills) | Craftax-Classic / Craftax | Symbolic + nearest-object offsets | 1B | n.r.; Collect Diamond 88.2% / Enter Gnomish Mines 9.1% of episodes | 5 | 1x H100 | ~1 h | probable repo, no licence |
| [[Multitask Preplay (Carvalho 2025)]] (oracle simulator, privileged goals) | Craftax | custom 56-dim vector | n.r. | 6.31 (bar label; best checkpoint) | n.r. | n.r. | n.r. | yes, MIT / no licence |
| [[GlyphBench LLM Agents (Creus Castanyer 2026)]] (zero-shot LLM, glyph text) | Craftax | text (glyphs) | no training | 11.4 (Sol, high effort); 15.8 (max effort); 51.4 (one GPT-6-Astra episode) | 5 episodes | API | n.r. | yes, MIT |

Sources for every cell are in the linked notes.

## How to read the table

- Like with like: compare only within one block; environment, observation type, step budget and metric must all match.
- Block A is the closest target for our agents: the scoreboard values are the README's single numbers, a mean over 10 seeds for the Craftax-paper rows, with no spread given ([Craftax README](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [Craftax paper](https://arxiv.org/abs/2402.16801) Fig. 3).
- Block A2 has the same budget but scores a separate evaluation of the trained policy (final checkpoint, sampled actions, many episodes); compare these with each other rather than with A. A3 uses prior knowledge (masks, shaped reward) or best-of-attempts search at test time.
- The scoreboard states that scores from outside the Craftax paper are "reported and have not been verified" (PPO-GTrXL, PQN-RNN, Simulus, Efficient MBRL); PPO-LSTM is proposed in an open PR and the maskedPPO-GTrXL PR was closed for using prior knowledge ([PR #54](https://github.com/MichaelTMatthews/Craftax/pull/54), [PR #52](https://github.com/MichaelTMatthews/Craftax/pull/52)).
- Metric differs between sources: Craftax-paper baselines log returns of training episodes with the stochastic policy; Simulus scores 100 test episodes at the end with sampling temperature 0.5; the UED blocks score checkpoints on 20 fixed levels (different levels in B and B2); DiCode and AHC cap evaluation episodes at 8192 steps; PQN-RNN, PPO-GTrXL and Efficient MBRL do not state their protocol.
- Score in Efficient MBRL and ITC means two things: "reward %" (used in the table) and a separate "score %" (geometric mean of achievement success rates); the table uses reward % throughout. The "Crafter score" in block H is that geometric mean.
- Spread: "s.e." is standard error over seeds; "sd" standard deviation; Simulus's and EASimulus's sd are computed in this pass from per-run files; PACE's ± is across levels, not seeds; everything else as labelled or none.
- Many rows are third-party reruns of covered methods; each method's note lists them. Reruns of PPO-RNN on Craftax-1B with separate evaluation land at 16.2-16.4% (36.7-37.05 raw), close to the scoreboard's 15.3; PyTorch ports land lower (PPO-LSTM port 15.0%, GTrXL port 8.4%).
- ICM numbers are unreliable: the author states they came from a buggy commit ([Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4)).
- RND numbers predate a fix to the intrinsic-value bootstrap ([Baselines PR #5](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/5), May 2025).
- The third-party reruns in block D are the Simulus authors' runs of the Craftax baselines ([Simulus paper](https://arxiv.org/abs/2502.11537) Fig. 1), slightly below the scoreboard values.
- Environment versions differ and are mostly not reported; Craftax errata change rewards before v1.5.0 and v1.6.0 ([Craftax README](https://github.com/MichaelTMatthews/Craftax#-errata)); AHC measures Craftax 1.6.1 returns 0.23-0.29 higher than 1.4.5 ([AHC paper](https://arxiv.org/abs/2610.15328) App. A.14); open bugs affect Craftax-Classic mob handling ([issue #57](https://github.com/MichaelTMatthews/Craftax/issues/57), [issue #60](https://github.com/MichaelTMatthews/Craftax/issues/60)).
- Wall-clock and hardware are rarely per run: the Craftax paper gives only benchmark-level statements, so "n.r." in those cells means no per-algorithm number exists.

## Rejected candidates

Checked in the second pass and not given a note (one line each; arXiv ids unless stated).

Closed scoreboard PRs:
- DQN for Craftax-Classic ([Craftax PR #59](https://github.com/MichaelTMatthews/Craftax/pull/59), closed a minute after opening): only an illustrative 100k-step example, "evaluation return=1.767±1.247", no seeds.
- DQN in Craftax_Baselines ([Baselines PR #8](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/8), closed seconds after opening): code and CartPole/Crafter configs, no Craftax results.

Craftax results only in figures (no numbers in text, tables or files):
- AGaLiTe (2310.15719, TMLR 2024): Craftax-Symbolic 1B against its own GTrXL-128, Fig. 4, 15 seeds.
- OPEN learned optimiser (2407.07082, NeurIPS 2024): Craftax-Classic 30M, zero-shot transfer, Fig. 5.
- Adam-Rel (2412.17113, NeurIPS 2024): Craftax-Classic 1B, Fig. 3; its Craftax code link has expired.
- How Should We Meta-Learn RL Algorithms? (2507.17668, RLC 2025): Craftax-Classic 30M meta-test panels, App. B.
- EGGROLL (2511.16652, ICML 2026): returns normalised by an unreported PPO value, Fig. 14.
- Dynamic Mixture of Experts (2511.18987): Craftax-Pixels 1B on PPO + SimBa, Fig. 3, no code.
- GTR Gaussian-reshaped trust region (2606.03382): Craftax 1B, Fig. 7, 3 seeds.
- C-TeC (2603.02008, ICLR 2026): reward-free Craftax-Classic 1B, achievements in Fig. 9 only.
- NCC (2505.20659, RLC 2025): UED on Craftax 1B, Fig. 3b only (per-seed wall-clock is tabulated).
- Is Exploration or Optimization the Problem for Deep RL? (2508.01329): PPO ± RND gap, Fig. 4c.

Numbers too vague or setting undefined:
- Scaling Empowered Agents (2610.13002): only "7-8 out of 22" achievements in the text, reward-free two-phase protocol, budget stated as both 1e9 and 2e9; code link 404.
- Diffusing Blame, ED-PPO (2606.31700, ALIFE 2026): raw returns (20.9 ± 2.9, BP-PPO 27.0) but Craftax variant, observation and budget not stated, and two inconsistent ED-PPO values.

Modified Craftax, custom tasks or non-reward metrics:
- Instruction relabelling OIR (2506.20061), CrafText (2505.11962), alem (2606.08340), Multi-Agent Craftax (2511.04904), MAC (2508.15679), NiceWebRL (2508.15693), HiSD (2601.23156, fully observable custom Craftax), LEO / CraftaxGC (2605.23551), curriculum foresight (2610.04604), complex-valued memories (2609.38598) and POBAX (2508.00046) on no-inventory Craftax, ForageWorld (2506.06981), ethical-penalty Craftax (2608.14642; its unconstrained PPO-GRU rerun is in [[PPO-RNN (Matthews 2024)]]), AGWM world-model metrics (2605.06841), CODE-SHARP goal success (2602.10085), GRID multi-agent (2606.18537), interpretability study with one PPO seed and per-achievement rates only (2610.14186).
- OpenReview submissions with custom goal metrics (abstracts only): PGOC (Dv00JHW1ZU, gjDhOi9byl), BEAT (KsHHTHQ8la), ATLAS (cBCDw5AsKg), GRAFT (4qf7UONkSq), Scaling Contrastive RL (9FzOhgTpoV), ZeroHero (mOioHq1A7d), Replace or Repair (rBf29ZC7f7).

Crafter only, or Craftax only cited:
- CrafterDojo C-VPT (2508.13530): the paper reports Crafter for its models (the repository evaluates `craftax_pixels`, so this is ambiguous); its PPO-RNN expert is in [[PPO-RNN (Matthews 2024)]].
- EWM (2610.13504), EMERALD (2507.04075; rebuttal mentions Craftax results not in arXiv v2), One Life to Learn (2510.12088, Crafter-OO), Counting Capabilities (OpenReview 5OyDOhMxXZ, bonus "inert on Craftax", no number).
- BALROG: environments are BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack, NetHack; no Craftax.
- IRIS, Δ-IRIS, Curious Replay, DART: only Crafter numbers exist, quoted in [[Efficient MBRL (Dedieu 2025)]] and [[ITC (Kim 2026)]] Table 1.
- Cite Craftax without experiments: 2603.06009, 2605.06869, 2603.23129, 2610.14807, 2605.01131, 2511.15378, 2407.03969, 2505.19683, 2606.00880, 2503.01450, 2605.21800, 2609.09059, and about 40 environment or library papers in the Semantic Scholar list with at most 3 mentions of Craftax.
- Non-arXiv citing papers: Q-learning novelty search (Monopoly only), object-empowerment tool selection (MiniHack only), autotelic LLM exploration (no Craftax numbers).

Repository-only results not given a note:
- I-yuki-0424/Decision-Process-order-driven: working notes without README, a retracted headline, arms added after seeing results, still changing on 2026-10-20.
- perfectteatimer/model-based-rl-for-craftex (DreamerV3 reimplementation, Classic-Pixels 1M, 15.34%) and nathanz3491/dreamerv3-double-rssm (Craftax-Symbolic 1.1M, rule-based mask): one training seed each.
- shuhan-wang1/CLIMB: draft continual DreamerV3, below its own baseline (DreamerV3 4.17 ± 0.26 raw at 500k).
- rekursiv-ai/priml: own Numba port of the Craftax rules plus a PufferLib variant (reset pool, masks, changed reward).
- AlexGoldie/discogen `baseline_scores.yaml` (100M-step PPO / PPO-RNN / transformer returns, provenance partial, no spread); ECE567 course projects (one seed each); PufferLib Craftax configs (variant environment); uppifyagency/fractal-AI-FMC (plans with the true simulator, 500-step episodes).
- LLM-harness repositories (mansicer/jev-plays, zcyy0/Long-horizon-agent-rl-craftax, mariklolik/sirius_llm_craftax, juandatan/tiny-sima): non-standard horizons, macro actions or one seed.
- Forks of all covered-method repositories (Craftax 54, Craftax_Baselines 20, purejaxql 19, jaxued 16, transformerXL_PPO_JAX 7, Simulus 3): no new numbers.

## Sources scanned

First pass (2026-10-20): implementations mentioned in the two Craftax repositories and the paper they cite.

- Craftax repository README at commit `c3c2e0d` (2026-06-20): scoreboard (Craftax-1B, Craftax-1M), Experiment, Gotchas, Errata, See also, Citation ([repo](https://github.com/MichaelTMatthews/Craftax)); `MAINTENANCE.md`, `CONTRIBUTING.md`, `tutorial.md` contain no algorithms.
- Craftax paper, arXiv:2402.16801 v2 (3 June 2024, ICML 2024): §4 baselines, App. E hyperparameters, App. F extra results, App. G UED.
- Craftax_Baselines repository at commit `7ce36fa` (2026-06-16): README, `ppo.py`, `ppo_rnn.py`, `ppo_rnd.py`, `models/`, `wrappers.py`.
- Craftax "See also" list: Crafter, NLE, PureJaxRL, JaxUED, Minimax, Gymnax, JaxMARL; none reports Craftax results except JaxUED, whose `examples/craftax/craftax_plr.py` implements the paper's UED runs.
- Primary sources of the scoreboard entries: transformerXL_PPO_JAX README and issues; PQN paper arXiv:2407.04811 v6 and purejaxql repository; Simulus paper arXiv:2502.11537 v4 and repository; Efficient MBRL paper arXiv:2502.01591 v1, v2, v3.
- GitHub issues read: all issues of Craftax_Baselines (#1, #2, #4, #7, #11); Craftax #19, #34, #37, #50, #53, #57, #60; transformerXL_PPO_JAX #2; purejaxql #9, #11, #12; Simulus #1, #2.
- Craftax pull requests (full list) checked to trace scoreboard edits: #23 (PPO-GTrXL added), #36 and #44 (M3 added, renamed Simulus).
- Craftax_Baselines pull requests (full list): #5 (RND GAE fix), #8 (DQN, closed), #10 (JAX/Orbax API update, open).

Second pass (2026-10-20): every further implementation with Craftax or Craftax-Classic numbers.

- Gap PRs read with their linked results: Craftax #52 and its repository (Doguhan58/craftax-maskedPPO-reward-shaping at `f728b95`), #54 with the Modrax repository (`ed6880f`) and its W&B report, #59 (diff), Craftax_Baselines #8 (diff).
- Semantic Scholar citations of arXiv:2402.16801 (105 papers via `/citations?limit=1000`); the arXiv HTML of all 93 with an arXiv id was text-searched for "Craftax", and every paper with more than 3 mentions was read for results.
- arXiv API search `all:craftax` (30 hits); OpenReview api2 search for "craftax" (185 notes; PDFs and forum pages sit behind a browser check, so only abstracts and reviews were read); Hugging Face papers and web searches for Craftax with world-model, memory, hierarchical, offline and LLM-agent terms; BALROG, DreamerV3, Δ-IRIS, Curious Replay and Achievement Distillation repositories checked for Craftax numbers.
- GitHub: `gh search repos` (craftax, "craftax classic", topic:craftax; about 330 repositories triaged), code search for the four Craftax env ids, `make_craftax_env_from_name` and result files; README diffs of all forks of the covered-method repositories.
- Repositories read for the new notes are listed with commit hashes in each note's Code section.

## Gaps

- OpenReview-only candidates whose PDFs could not be read (browser check), so not added: SOR "Beyond First Success" (ogwGa50wpH; its anonymous data release gives 5.48% against 4.00% for PPO on Craftax at about 1M steps, recomputed by a search agent, unverified), "An Empirical Study of RL Scaling Laws" (5MKROf4fHQ; claims a new Craftax state of the art, no number in the abstract), CALOR (rK859Ipz0p; reward-free Craftax-Classic 1B, abstract gives Crafter score 40.7), CCSF "Put Your Skills To Work" (RBnlUOBW1b; 8.9 of 22 achievements per episode), VLM goal planning (txfZLT8zYq), "Building State vs Using State" (acFsaKPRSr; "Craftax score" 15.09 → 16.56, metric undefined), Permanent and Transient representations with PQN (5XfxEQ2SCt), ReMDM masked-diffusion planner (VGyjG8Gy29; repository mathisweil/craftax-ReMDM-planner reports 11.81 raw on Craftax-Classic-Symbolic by imitation of a PPO-RNN expert), coding agents as task compilers (C1D0ByaRyi), Learning to Act While Thinking (j49htODgNn), display-error diagnostic with Δ-IRIS and DreamerV3 stone-achievement rates (W1Fk5fO7kg).
- Journal version of SGRL (Neural Networks 2026, closed access) and PACE's rebuttal tables with 5 more Craftax seeds: not read.
- Numbers shown only in figures: Craftax-paper UED rewards (block B), PQN (MLP) on Craftax-1B, PPO-RNN at 10B, and the figure-only papers listed above.
- Corrected ICM results: not published.
- Per-run wall-clock and parameter counts for the Craftax-paper baselines and PQN-RNN: not reported (a GTrXL reimplementation has 4.62M parameters, [[PPO-LSTM (Jiralerspong 2026)]]).
- Seed count and evaluation protocol for the original PPO-GTrXL: not stated; reruns in A2 and C2 now bracket it.
- Code not found: Efficient MBRL, Feasibility Classification, AHC, PACE, Diffusing Blame, Dynamic MoE; Sable only as Drive files; SimBa's PPO code not released; SCALAR's code is a probable match, not linked by the paper.
- Training details missing: Sable's Craftax hyperparameters and seeds; Multitask Preplay's seeds and budget for the standard setting; SimBa's `avg_return` protocol.
- Simulus-A (repository default since August 2026): no Craftax results yet.
- PPO-LSTM scoreboard PR #54: open, not merged.
- Craftax version used by each result: rarely reported (ITC pins 1.4.5, AHC uses 1.4.5, SOR 1.6.1).
