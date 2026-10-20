---
title: "SGRL (Qi 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Yajie Qi
  - Wei Wei
  - Lin Li
  - Lijun Zhang
  - Zhidong Gao
  - Da Wang
  - Huizhong Song
year: 2025
venue: "arXiv preprint (v1, Sep 2025); journal version in Neural Networks 2026 (not read)"
paper: "https://arxiv.org/abs/2509.22008"
code: "https://github.com/jiejieqi/SGRL"
envs:
  - Craftax-Classic-Pixels-v1
tags:
  - implementation
  - craftax-classic
  - ppo
  - llm-guided
  - exploration
  - action-masking
created: 2026-10-20
---
# SGRL (Qi 2025)

## TL;DR

- Structured Goal-guided RL: an LLM writes a goal-generator function once, a second LLM maps goals to action masks, and PPO is trained on goal-conditioned, masked policies; LLM calls are rare, so training is far faster than ELLM or AdaRefiner ([paper](https://arxiv.org/abs/2509.22008) Table 1).
- Craftax-Classic, 5M steps: score 33.8 ± 1.5%, reward 13.0 ± 0.3 (raw, max 22); 10M steps: 43.9 ± 2.6% and 14.9 ± 0.4; 5 seeds ([paper](https://arxiv.org/abs/2509.22008) Tables 1, 2).
- Budgets (5M, 10M) differ from every other block of [[Existing Implementations]], and the LLM priors make it a knowledge-assisted method.
- Journal version: "Structured Goal-Guided Reinforcement Learning with Large Language Models", Neural Networks 2026, DOI 10.1016/j.neunet.2026.109769 (closed access, not read).

## Method

- Goal generation: DeepSeek-R1 writes a reusable Python function that maps a text rendering of the state to the top-3 goals with priority weights; the LLM revises the weights every 2M steps ([paper](https://arxiv.org/abs/2509.22008) App. B.3.1).
- Goal-conditioned action pruning: DeepSeek-V3 maps each goal to a binary action mask, cached in a goal-mask bank; the mask is relaxed stochastically with a 3-stage cosine-annealed coefficient ([paper](https://arxiv.org/abs/2509.22008) Eq. 3-4).
- Policy: PPO on the state concatenated with goal embeddings (paraphrase-MiniLM-L6-v2), with masked logits ([paper](https://arxiv.org/abs/2509.22008) Eq. 5-6).
- LLM decoding: temperature 0.5, top-p 1.0, max 100 tokens ([paper](https://arxiv.org/abs/2509.22008) App. B.1).

## Training setup and cost

| Hyperparameter (Craftax-Classic) | Value | Source |
|---|---|---|
| Learning rate, optimiser | 7e-4, AdamW | [paper](https://arxiv.org/abs/2509.22008) Table 3 |
| Batch size | 512 | Table 3 |
| Environments | 256 | Table 3 |
| Update epochs | 4 | Table 3 |
| Clip ε | 0.2 | Table 3 |
| γ | 0.97 | Table 3 |
| Entropy / value coefficient | 0.01 / 0.5 | Table 3 |
| Activation | ReLU | Table 3 |
| Steps | 5M, 10M | Table 3 |
| Rollout steps / minibatches / GAE λ | 16 / 8 / 0.8 | repo `train.py` defaults |

- Hardware: one RTX 4090 (24 GB) with an i9-14900K for Craftax-Classic ([paper](https://arxiv.org/abs/2509.22008) §5.1.2).
- Throughput: 18.5 x 10² steps/s for SGRL against 135.3 x 10² for PPO, 0.9 x 10² for ELLM and 0.3 x 10² for AdaRefiner ([paper](https://arxiv.org/abs/2509.22008) Table 1).
- Wall-clock and parameters: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-Classic 5M | score, % (geometric mean of achievement success rates) | 33.8 ± 1.5 | 5 | [paper](https://arxiv.org/abs/2509.22008) Table 1 |
| Craftax-Classic 5M | episode reward (raw, max 22) | 13.0 ± 0.3 (59.1% of max) | 5 | Table 1 |
| Craftax-Classic 5M | deepest achievement depth | 8 | 5 | Table 1 |
| Craftax-Classic 10M | score, % | 43.9 ± 2.6 | 5 | Table 2 |
| Craftax-Classic 10M | episode reward (raw) | 14.9 ± 0.4 (67.7% of max) | 5 | Table 2 |

- Baselines at 5M (score % / reward / depth; Table 1): AdaRefiner 28.5 ± 2.3 / 12.3 ± 0.9 / 7; ELLM 28.4 ± 2.5 / 12.2 ± 1.0 / 6; PPO 24.8 ± 5.7 / 11.9 ± 1.1 / 6 (a PPO rerun with the shared hyperparameters above, not the Craftax-paper settings).
- The "Human 50.5 ± 6.8 / 14.3 ± 2.3" row is copied from the original Crafter paper, not measured on Craftax-Classic.
- Ablations at 10M (score / reward; Table 2): static pruning 38.5 ± 1.9 / 14.2 ± 0.4; no pruning 40.0 ± 2.1 / 14.7 ± 0.2; no priority 35.3 ± 1.6 / 13.9 ± 0.6.
- Collect Diamond first unlocked at about 3.7M steps (figure only, §5.2).
- Not stated: type of ±, training versus evaluation episodes, greedy versus sampled actions; curves are smoothed (§5.1.2).
- Observation type is not named in the paper; the repository default is `Craftax-Classic-Pixels-v1`, and `run.sh` does not override it, while the text renderer reads symbolic map fields.

## Problems and limitations

- Reward and exploration depth are misaligned, so SGRL has no clear reward advantage ([paper](https://arxiv.org/abs/2509.22008) §5.2).
- Over-strict masking converges to suboptimal policies ([paper](https://arxiv.org/abs/2509.22008) §5.3.2).
- ELLM and AdaRefiner were only run to 5M steps because of LLM cost ([paper](https://arxiv.org/abs/2509.22008) App. D).
- §5.3.2 calls Figure 5 "on Crafter", but it is a Craftax-Classic ablation.
- Needs an external LLM endpoint (configured in `Craftax/lm.py`).

## Code

- Repository: https://github.com/jiejieqi/SGRL (`train.py`; all paper runs in `run.sh`, 5 seeds, `--total_timesteps 1e7`); commit read in this pass: `cbb1b0a` (2025-12-26, single commit); no issues.
- Framework: JAX, built on the Craftax code.
- Licence: none in the repository.
