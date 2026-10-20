---
title: "Robust PLR (Matthews 2024)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Michael Matthews
  - Michael Beukman
  - Benjamin Ellis
  - Mikayel Samvelyan
  - Matthew Jackson
  - Samuel Coward
  - Jakob Foerster
year: 2024
venue: "ICML 2024 (PMLR 235)"
paper: "https://arxiv.org/abs/2402.16801"
code: "https://github.com/DramaCow/jaxued/blob/main/examples/craftax/craftax_plr.py"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ued
  - curriculum
  - ppo
created: 2026-10-20
---
# Robust PLR (Matthews 2024)

## TL;DR

- Robust Prioritized Level Replay (PLR⊥, Jiang et al. 2021a) on Craftax-1B via JaxUED; Craftax paper UED baseline.
- Worse than vanilla [[PLR (Matthews 2024)]] and only slightly better than [[Domain Randomisation (Matthews 2024)]] ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Exact reward: not reported (curves only); 10 runs on 20 fixed evaluation levels.

## Method

- As [[PLR (Matthews 2024)]], but gradient updates only on levels sampled from the buffer, not on newly generated random levels ([paper](https://arxiv.org/abs/2402.16801) §4.2).
- Replay rate 0.8 instead of 0.5 ([paper](https://arxiv.org/abs/2402.16801) Table 14).
- In JaxUED: `craftax_plr.py` with `--no-exploratory_grad_updates` (the flag defaults to on).
- Policy: feed-forward MLP; no recurrence.

## Training setup and cost

- Hyperparameters: [paper](https://arxiv.org/abs/2402.16801) Table 14 (as in [[PLR (Matthews 2024)]]: lr 0.0003, 1024 envs, γ 0.99, GAE λ 0.9, 5 epochs, 2 minibatches, buffer 4000, MaxMC, rank, temperature 1.0, staleness 0.3) except replay rate 0.8.
- Budget: 1B environment steps; equal interactions mean fewer gradient updates than DR or PLR ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Hardware, wall-clock, parameters, memory: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | evaluation reward on 20 fixed levels, % of max (226) | not reported as a number; below PLR, slightly above DR | 10 | [paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7 |

- Third-party rerun (2026-10-20 pass): [[PACE (Yuan 2026)]] reruns Robust PLR (PLR⟂) with a JaxUED PPO student on Craftax-Symbolic for 1.07B steps and evaluates the final checkpoints on 20 held-out levels: mean episodic reward 23.780 (sd across levels 4.688), i.e. 10.5% of 226 over 10 seeds; IQM of min-max normalised reward 0.409 [0.320, 0.520] ([PACE paper](https://arxiv.org/abs/2605.01358) Tables 4, 5). Same kind of protocol as the table above but different levels, hyperparameters and budget; in that rerun DR ranks above PLR, unlike the Craftax paper.

## Problems and limitations

- Contradicts earlier evidence that PLR⊥ beats PLR; suggested reasons: Craftax always generates solvable, high-quality levels, and PLR⊥ does fewer gradient updates per environment step ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Evaluation metric differs from the training-episode return of the scoreboard baselines.

## Code

- Repository: https://github.com/DramaCow/jaxued, `examples/craftax/craftax_plr.py`.
- Framework: JAX. Licence: Apache-2.0 (declared in `pyproject.toml`).
- Citation: Matthews et al., ICML 2024, arXiv:2402.16801; method: Jiang et al., "Replay-Guided Adversarial Environment Design", NeurIPS 2021.
