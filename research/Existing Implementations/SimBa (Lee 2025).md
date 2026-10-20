---
title: "SimBa (Lee 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Hojoon Lee
  - Dongyoon Hwang
  - Donghu Kim
  - Hyunseung Kim
  - Jun Jet Tai
  - Kaushik Subramanian
  - Peter R. Wurman
  - Jaegul Choo
  - Peter Stone
  - Takuma Seno
year: 2025
venue: "ICLR 2025 (spotlight)"
paper: "https://arxiv.org/abs/2410.09754"
code: "https://github.com/SonyResearch/simba (no PPO or Craftax code)"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - architecture
  - model-free
created: 2026-10-20
---
# SimBa (Lee 2025)

## TL;DR

- Network architecture for scaling RL parameters: running-statistics observation normalisation, pre-LayerNorm residual MLP blocks and a final LayerNorm; swapped in for the MLP actor and critic of PPO on Craftax ([paper](https://arxiv.org/abs/2410.09754) §4, §6.2).
- The paper shows Craftax only as curves; the repository's result files give the numbers: at 1B steps, PPO + SimBa 28.97 raw return (sd 0.54; 12.8% of 226) against PPO 25.74 (sd 2.06; 11.4%), 5 seeds (computed in this pass from `results/ppo_simba.csv` and `results/ppo.csv`).
- Same budget and observation as block A of [[Existing Implementations]], but the metric (`avg_return`) is the repository's logged return, protocol not documented.

## Method

- SimBa block: observation normalisation with running statistics, residual feed-forward blocks with pre-LayerNorm, and a post-LayerNorm before the output head, to inject a simplicity bias ([paper](https://arxiv.org/abs/2410.09754) §4).
- On Craftax, SimBa replaces the MLP actor-critic of PPO; other settings from the Craftax paper ([paper](https://arxiv.org/abs/2410.09754) §6.2, Table 10).

## Training setup and cost

| Hyperparameter (Craftax PPO + SimBa) | Value ([paper](https://arxiv.org/abs/2410.09754) Table 10) |
|---|---|
| Critic | 2 SimBa blocks, hidden 512, lr 1e-4 |
| Actor | 1 SimBa block, hidden 256, lr 1e-4 |
| Optimiser | AdamW, β (0.9, 0.999), ε 1e-8, weight decay 1e-2 |
| Environments | 1024 |
| Steps | 1B |
| Other PPO settings | Craftax paper |

- Hardware: RTX 3070, A100 or H100 across the paper's experiments (App. G); Craftax wall-clock and parameter count: not reported.
- Craftax is reported in raw average scores, not % of max ([paper](https://arxiv.org/abs/2410.09754) App. H.4).

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic, PPO + SimBa | `avg_return` (raw) at 999,948,288 steps, mean (sd) | 28.97 (0.54), i.e. 12.8% (0.24%) of 226 | 5 | computed in this pass from [results/ppo_simba.csv](https://github.com/SonyResearch/simba/blob/main/results/ppo_simba.csv) |
| Craftax-1B, Symbolic, PPO (MLP) | same | 25.74 (2.06), i.e. 11.4% (0.91%) | 5 | computed from [results/ppo.csv](https://github.com/SonyResearch/simba/blob/main/results/ppo.csv) |
| Craftax-1B, Symbolic | average return and success on three tasks (iron sword or pickaxe, defeat Orc Mage) | figure only | not stated (Fig. 25: 5) | [paper](https://arxiv.org/abs/2410.09754) §6.2, Fig. 10, Fig. 25 |

- Per-seed final values: PPO + SimBa 29.08, 29.43, 29.52, 28.38, 28.41; PPO 27.80, 25.13, 22.69, 25.60, 27.49 (seeds 0, 1000, 2000, 3000, 4000).
- The PPO row is a third-party rerun of [[PPO (Matthews 2024)]]; added to that note.
- Whether `avg_return` is training-episode or evaluation return is not documented.

## Problems and limitations

- No limitations stated for Craftax.
- The PPO + Craftax code is not in the repository; a maintainer said it "was a bit messy" and offered it by email ([simba issue #7](https://github.com/SonyResearch/simba/issues/7), closed).
- Craftax numbers exist only as repository CSVs and paper curves.

## Code

- Repository: https://github.com/SonyResearch/simba (off-policy agents; Craftax GIFs in `docs/images/`; result CSVs in `results/`); commit read in this pass: `7d0358b` (2025-02-25).
- Framework: JAX.
- Licence: Apache-2.0 (LICENSE file).
- Used as the PPO backbone in Dynamic MoE (Kim 2025, Craftax-Pixels, figure only; listed under rejected candidates in [[Existing Implementations]]).
