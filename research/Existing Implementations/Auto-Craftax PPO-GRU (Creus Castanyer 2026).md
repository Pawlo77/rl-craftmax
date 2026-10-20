---
title: "Auto-Craftax PPO-GRU (Creus Castanyer 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Roger Creus Castanyer (repository owner; experiments run by an AI agent)
year: 2026
venue: "GitHub repository with an auto-generated report (no paper)"
paper: "https://github.com/roger-creus/auto-craftax/blob/main/report/report.md"
code: "https://github.com/roger-creus/auto-craftax"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - recurrent
  - exploration
  - automated-research
created: 2026-10-20
---
# Auto-Craftax PPO-GRU (Creus Castanyer 2026)

## TL;DR

- A week-long autonomous research campaign in which an AI agent (Claude Code, through the xgenius framework) proposed 184 hypotheses and ran them on SLURM clusters to maximise Craftax-1B ([README](https://github.com/roger-creus/auto-craftax); [report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) Abstract).
- Best configuration h096 (PyTorch PPO, GRU, CNN + MLP observation encoder, 128-step rollouts, max gradient norm 1.0, γ 0.999, RND coefficient 0.005): 40.50 mean training-episode return (17.9% of 226), 2 seeds; without RND (h040) 39.55 ± 2.32 (sd), 3 seeds ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.3, §7).
- Close to block A of [[Existing Implementations]] (Symbolic, 1B, training return), but repository-only, few seeds, and selected among many hypotheses.

## Method

- PPO with a GRU (512 hidden) and a structured observation encoder: a 2-layer CNN (83 → 32 → 64 channels) over the 9 x 11 map plus an MLP over the player stats, concatenated and projected ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.2).
- RND intrinsic reward with a small coefficient (0.005) ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.5).
- Changes that helped, by the report's ranking: 128-step rollouts, CNN encoder, max gradient norm 1.0, GRU over LSTM, entropy annealing 0.03 → 0.005, γ 0.999, RND 0.005 ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §5).
- Changes that did not: PyTorch GTrXL, raw reward bonuses, PopArt, BatchNorm (pilot gains vanished at 1B), larger networks, cosine learning-rate schedule, higher GAE λ, 8 PPO epochs ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §5).

## Training setup and cost

| Setting | Value ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.4, §7) |
|---|---|
| Steps | 1B |
| Rollout steps | 128 |
| Max gradient norm | 1.0 |
| γ / GAE λ | 0.999 / 0.8 |
| Epochs / minibatches / clip | 4 / 8 / 0.2 |
| Hidden size | 512 |
| Learning-rate schedule | linear decay |
| RND coefficient | 0.005 |

- Hardware: H100 MIG slices (3g.40gb) on three clusters and A100 40 GB on one; 1 GPU, 8 CPUs, 32-48 GB RAM per job ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §2.1).
- Wall-clock: about 10-13 h per 1B run, about 2 h per 200M pilot; about 873 GPU-hours over 139 completed jobs ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §2.2, §8).
- Parameters: not reported.

## Results

Mean training-episode return (raw, max 226) at 1B steps ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §3, §4.3, §4.5, §7).

| Configuration | Return | % of 226 | Seeds |
|---|---|---|---|
| h096: PPO-GRU + CNN + 128 steps + grad 1.0 + RND 0.005 | 40.50 (seeds 40.02, 40.98) | 17.9 | 2 |
| h040: same without RND | 39.55 ± 2.32 (sd; best seed 41.46) | 17.5 | 3 |
| h023: PPO-LSTM + 128 steps | 38.10 ± 2.61 | 16.9 | not stated |
| PPO baseline (port) | 26.83 ± 1.31 | 11.9 | 3 |
| PPO-LSTM baseline (port) | 33.88 ± 0.50 | 15.0 | 3 |
| PQN baseline (port) | 20.95 ± 1.04 | 9.3 | 3 |
| PQN-LSTM baseline (port) | 24.23 ± 2.95 | 10.7 | 3 |
| GTrXL reproduction (PyTorch, h007) | 19.01 ± 0.84 | 8.4 | 3 |

- h096 enters the dungeon in 96-100% of episodes; no configuration entered the Gnomish Mines ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §5, §7).
- Baseline rows are PyTorch ports; added as third-party reruns to [[PPO (Matthews 2024)]], [[PPO-RNN (Matthews 2024)]], [[PQN-RNN (Gallici 2025)]] and, for the failed GTrXL reproduction, [[PPO-GTrXL (Hamon 2024)]].

## Problems and limitations

- Selected from 184 hypotheses with 2-3 seeds per configuration, so the best result is optimistically biased; the h096 mean uses only 2 seeds.
- Pilot gains often did not survive at 1B (BatchNorm: +13% at 200M, −21% at 1B) ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.6).
- The agent reported that `infos['Achievements/...']` is always 0 for non-done environments, which broke its Go-Explore and reward-shaping attempts ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.5, §8).
- Its PyTorch GTrXL reached only 46% of the published score, attributed to implementation differences ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §4.1).
- 24% of jobs were preempted; mixed H100 MIG and A100 hardware ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §8).
- No human-reviewed paper; the report is auto-generated.

## Code

- Repository: https://github.com/roger-creus/auto-craftax (`python src/rl/ppo_lstm.py --use-gru --use-structured-obs --num-steps 128 --max-grad-norm 1.0 --gamma 0.999 --rnd --rnd-coef 0.005`; raw results in `results/`); commit read in this pass: `358d6c0` (2026-03-24).
- Framework: PyTorch, Singularity containers.
- Licence: none in the repository (the xgenius framework is MIT).
- Same owner as [[GlyphBench LLM Agents (Creus Castanyer 2026)]].
