---
title: "ITC (Kim 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Youngin Kim
  - Ray Sun
  - Inho Kim
  - Bumsoo Park
  - Hyun Oh Song
year: 2026
venue: "ICML 2026 (per the repository description)"
paper: "https://arxiv.org/abs/2605.16457"
code: "https://github.com/snu-mllab/Identifiable-Token-Correspondence"
envs:
  - Craftax-Classic-Pixels-v1
  - Craftax-Pixels-v1
tags:
  - implementation
  - craftax
  - craftax-classic
  - model-based
  - world-model
  - transformer
  - ppo
created: 2026-10-20
---
# ITC (Kim 2026)

## TL;DR

- Identifiable Token Correspondence: the fast transformer world model of [[Efficient MBRL (Dedieu 2025)]] plus an optimal-transport decoding step that matches next-frame tokens to tokens of the previous frame, and 3D spatio-temporal RoPE ([paper](https://arxiv.org/abs/2605.16457) §3).
- Craftax-Classic (Pixels), 1M steps: reward 72.46 ± 0.45% and score 35.60 ± 0.92% (s.e., 10 seeds), above the Dedieu slow agent's 69.66% ([paper](https://arxiv.org/abs/2605.16457) Table 1).
- Craftax (Pixels), 1M steps: reward 7.09 ± 0.20%, score 2.40 ± 0.04% (s.e., 5 seeds) ([paper](https://arxiv.org/abs/2605.16457) §4.2, Table 6).
- Comparable with blocks E and F of [[Existing Implementations]] (Pixels, 1M); one RTX 3090, 48.2 h per Craftax-Classic run (Table 4).
- An earlier version, "TWISTED", was an ICLR 2026 submission (rejected) by the same authors with the same Craftax-Classic numbers (OpenReview `1O8Jye1P0k`, abstract only).

## Method

- Base agent: Dedieu et al.'s (fast) TWM agent: nearest-neighbour patch tokenizer, GPT-2-style block-causal transformer, PPO with an RNN policy trained on real and imagined data ([paper](https://arxiv.org/abs/2605.16457) §3, §4.1).
- Optimal-transport decoding: Sinkhorn (OTT-JAX) matches each next-frame token either to an identified token of the previous frame (reuse) or to a transformer-generated token, which reduces duplicated or vanishing objects ([paper](https://arxiv.org/abs/2605.16457) §3, App. B).
- Positional encoding: 3D spatio-temporal RoPE (VideoRoPE) instead of absolute positions ([paper](https://arxiv.org/abs/2605.16457) §2.2, App. A.2.2).
- Hybrid rule on Craftax and Craftax-Classic: optimal-transport output in the screen interior, transformer output on the screen edges and the inventory ([paper](https://arxiv.org/abs/2605.16457) App. B).

## Training setup and cost

| Hyperparameter | Craftax-Classic | Craftax | Source |
|---|---|---|---|
| World model blocks / heads / embedding / MLP | 3 / 8 / 128 / 512 | same | [paper](https://arxiv.org/abs/2605.16457) Table 9, Table 11 |
| Dropout | 0.1 | 0.1 | Table 9 |
| Sequence length | 20 | 20 | Table 9 |
| Patches, tokens per frame | 7 x 7, 81 | 10 x 10, 143 | Tables 9, 11 |
| Codebook size | 4096 | 4096 | Table 9 |
| World model optimiser | Adam, lr 1e-3, 500 updates | same | Table 9 |
| Replay buffer | 128k | 48k | Table 9, Table 11 |
| Optimal transport c_d / c_w / Sinkhorn ε / iterations | 0.6 / 0.3 / 1e-5 / 10 | same | Table 9 |
| Policy environments / rollout / warm-up | 48 / 96 / 50k | 16 / 96 / 50k | Table 10, Table 11 |
| Imagination updates / γ / λ / lr | 300 / 0.925 / 0.625 / 4.5e-4 | same | Table 10 |

- Hardware: one RTX 3090; 48.2 h per Craftax-Classic run against 46.3 h for the baseline (+2.8% training time) ([paper](https://arxiv.org/abs/2605.16457) Table 4, §4.1).
- Craftax (full) wall-clock and parameter count: not reported.

## Results

| Benchmark | Metric | Value (mean ± s.e.) | Seeds | Source |
|---|---|---|---|---|
| Craftax-Classic 1M, Pixels | reward, % of max (22) | 72.46 ± 0.45 | 10 | [paper](https://arxiv.org/abs/2605.16457) Table 1 |
| Craftax-Classic 1M, Pixels | score, % (geometric mean of achievement success rates) | 35.60 ± 0.92 | 10 | Table 1 |
| Craftax-Classic 0.5M, Pixels | reward, % | 63.10 ± 1.24 | 10 | Table 1 |
| Craftax-Classic 0.5M, Pixels | score, % | 20.12 ± 0.80 | 10 | Table 1 |
| Craftax 1M, Pixels | reward, % of max (226) | 7.09 ± 0.20 | 5 | §4.2, Table 6 |
| Craftax 1M, Pixels | score, % | 2.40 ± 0.04 | 5 | Table 6 |

- Reward is "averaged over episodes of the final 50,000 environment interactions" (training episodes, not a separate evaluation); score is cumulative over training ([paper](https://arxiv.org/abs/2605.16457) Table 1 caption).
- Reproduced Dedieu baseline, Craftax-Classic 1M, reward / score (Table 1): with ITC's code 68.14 ± 0.42 / 24.89 ± 0.74; with ITC's hyperparameters 68.55 ± 0.72 / 27.24 ± 0.86; added to [[Efficient MBRL (Dedieu 2025)]].
- Ablation at 1M (Table 2): + 3D RoPE alone 71.85 ± 0.63 / 33.94 ± 1.10; + ITC 72.46 / 35.60.
- Optimal-transport hyperparameter sweep keeps reward between 71.10 and 72.46 (Table 5); exact next-state accuracy 51.13% against 46.94% for the baseline (Table 3).
- Quoted, not rerun: DreamerV3 53.2 ± 8.0 / 14.5 ± 1.6, IRIS 25.0 ± 3.2 / 6.66, Δ-IRIS 35.0 ± 3.2 / 9.30 (all Crafter numbers), DART 55.45 ± 3.39, Simulus 6.59 on Craftax ([paper](https://arxiv.org/abs/2605.16457) Tables 1, 6).

## Problems and limitations

- No limitations section in the paper.
- Craftax (full) uses only 5 seeds.
- Training-episode reward rather than a separate evaluation.
- Repository issues (all open, none about Craftax): checkpoint release request (#1), per-game Atari scores (#2), reading-list invitation (#3).

## Code

- Repository: https://github.com/snu-mllab/Identifiable-Token-Correspondence (`crafter_and_craftax/train.py --config_path=config/itc.yaml`; Craftax: `config/full_craftax_env.yaml`; reproduced baselines: `baseline_reproduced.yaml`, `baseline_our_hyperparams.yaml`); commit read in this pass: `4dfa22a` (2026-06-26).
- Framework: JAX 0.5, Flax 0.10, OTT-JAX; `craftax==1.4.5`.
- Licence: MIT.
