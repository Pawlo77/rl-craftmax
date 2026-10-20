---
title: "AHC (Zhu 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Jiayu Zhu
  - Luyuan Li
  - Yuyuan Shi
  - Hanyi Dong
year: 2026
venue: "arXiv preprint (v1, Oct 2026)"
paper: "https://arxiv.org/abs/2610.15328"
code: "not found (no link in the paper; GitHub search negative)"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - curriculum
  - ppo
  - transformer
  - start-state-curriculum
created: 2026-10-20
---
# AHC (Zhu 2026)

## TL;DR

- Anchor-and-Harvest Curriculum: a rule-based, non-LLM curriculum over 147 achievement, recipe-family and depth task units, which harvests progress signatures from training and replays them as practice start states on procedurally generated floors; the student is the PPO-GTrXL of [[DiCode (Mitsides 2026)]] with all 43 actions unmasked ([paper](https://arxiv.org/abs/2610.15328) §2, §4).
- Craftax (Symbolic), 2.005B rollout transitions: native return 46.70 ± 0.57 (sample sd, 3 seeds; 20.66% of 226) against 41.91 ± 1.01 for a fresh PPO-GTrXL with the same student and protocol ([paper](https://arxiv.org/abs/2610.15328) abstract, Tables 9, 10).
- At 0.996B: 38.69 ± 2.26 against 36.63 ± 1.42 ([paper](https://arxiv.org/abs/2610.15328) Table 9).
- Evaluation on 1024 fixed worlds x 10 stochastic rollouts with an 8192-step cap, like DiCode; not the training-episode metric of block A in [[Existing Implementations]].

## Method

- Task units: 147 achievement, recipe-family and depth units, from an engine-informed dependency graph ([paper](https://arxiv.org/abs/2610.15328) §4).
- Harvested progress signatures (floor, gate progress, inventory, armour, equipment) are realised on procedurally generated floor-indexed worlds and used as practice start states ([paper](https://arxiv.org/abs/2610.15328) §4).
- Adaptive difficulty through softened dynamics (σ), learnability x downstream-value scoring, eligibility and frontier rules, and an "anchor" share of original-task practice ([paper](https://arxiv.org/abs/2610.15328) §4).
- +20 completion bonus during curriculum practice only; all 43 actions available, no masking ([paper](https://arxiv.org/abs/2610.15328) §4).
- Policy conditioned on a 67-dimensional achievement indicator, all ones for the native task ([paper](https://arxiv.org/abs/2610.15328) §2).

## Training setup and cost

| Hyperparameter | Value ([paper](https://arxiv.org/abs/2610.15328) Table 26) |
|---|---|
| GTrXL | 2 layers, 8 heads, width 256, memory 128, gradient window 64 |
| Environments x rollout | 1024 x 128 |
| Epochs / minibatches | 4 / 8 |
| γ / GAE λ / clip ε | 0.999 / 0.8 / 0.2 |
| Entropy / value coefficient | 0.002 / 0.5 |
| Adam ε / gradient cap | 1e-5 / 1.0 |
| Learning rate | 2e-4 to 2e-6 |
| Schedule | 200 initialisation updates + 151 sessions x 100 updates |

- Environment: Craftax 1.4.5; budgets 0.9961472B and 2.0054016B PPO rollout transitions ([paper](https://arxiv.org/abs/2610.15328) §2, §5).
- Cost ([paper](https://arxiv.org/abs/2610.15328) Table 15, §5.5): primary AHC cohort 53.01 L40S GPU-hours for 3 runs (about 17.6 h each); native PPO-GTrXL 69.96 A6000 GPU-hours for 3 runs; additional AHC cohort at least 54.594 A100 GPU-hours.
- Parameters: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-Symbolic, 2.005B, primary AHC (L40S) | native return (raw), mean ± sample sd | 46.70 ± 0.57 (20.66 ± 0.25% of max) | 3 | [paper](https://arxiv.org/abs/2610.15328) Tables 9, 10 |
| same, additional AHC cohort (A100) | same | 46.08 ± 1.23 | 3 | Table 9 |
| Craftax-Symbolic, 0.996B, primary / additional | same | 38.69 ± 2.26 / 40.38 ± 1.69 | 3 / 3 | Table 9 |
| 2.005B, single-rollout protocol | same | 46.754 ± 0.574 | 3 | Table 36 |
| 2.005B ablations D4-A / D4-H / D4-AH | same | 46.10 ± 1.18 / 46.42 ± 1.55 / 45.14 ± 2.79 | 3 each | Tables 12, 14 |

- Protocol: 1024 fixed worlds x 10 stochastic rollouts, cap 8192 steps, native rewards and resets ([paper](https://arxiv.org/abs/2610.15328) §2).
- Same-protocol PPO-GTrXL (third-party rerun of [[PPO-GTrXL (Hamon 2024)]]): 36.63 ± 1.42 at 0.996B, 41.91 ± 1.01 at 2.005B (per seed 42.35 / 40.74 / 42.62; RTX A6000) ([paper](https://arxiv.org/abs/2610.15328) Tables 9, 12); added to that note.
- Capability profile (Table 12): Intermediate-18 macro success 48.18 ± 0.87%, Gnomish Mines entry 26.38%, warrior 20.25%; no weight-8 achievement solved.
- Cross-study context quoted, not run (Table 37): DiCode 48.55 at 2B, DiCode collaborator rerun 48.77 ± 2.10 (App. A.13, A.17).

## Problems and limitations

- Domain-specific rules and a hand-built dependency graph; harvested progress and modified training dynamics ([paper](https://arxiv.org/abs/2610.15328) §7).
- No matched non-LLM curriculum baseline (PLR, SFL, DR not run); 3 seeds; different GPU models across cohorts; Welch intervals not adjusted for multiplicity; primary cohort designated during revision ([paper](https://arxiv.org/abs/2610.15328) §7).
- Total environment interaction (including curriculum worlds) not metered ([paper](https://arxiv.org/abs/2610.15328) §7).
- The 8192-step evaluation cap is below Craftax's native episode length; Craftax 1.6.1 reads 0.23-0.29 higher than 1.4.5 ([paper](https://arxiv.org/abs/2610.15328) §7, App. A.14).
- No code.

## Code

- Not found (no link in the paper; GitHub search negative in this pass).
- Student and protocol follow [[DiCode (Mitsides 2026)]], whose code is public.
