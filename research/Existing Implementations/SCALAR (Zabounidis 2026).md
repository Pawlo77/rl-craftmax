---
title: "SCALAR (Zabounidis 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Renos Zabounidis
  - Yue Wu
  - Simon Stepputtis
  - Woojun Kim
  - Yuanzhi Li
  - Tom Mitchell
  - Katia Sycara
year: 2026
venue: "arXiv preprint (v1, Mar 2026); NeurIPS 2025 workshop (Bridging Language, Agent, and World Models), best paper honourable mention"
paper: "https://arxiv.org/abs/2603.09036"
code: "https://github.com/renos/flow-rl (probable; not linked in the paper)"
envs:
  - Craftax-Classic-Symbolic-v1
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - craftax-classic
  - hierarchical
  - llm-guided
  - ppo
  - skills
created: 2026-10-20
---
# SCALAR (Zabounidis 2026)

## TL;DR

- LLM-proposed skills (STRIPS-like operators with reward and termination code), a symbolic planner that chains them, and PPO options that learn each skill; LLM revisions from trajectory analysis and optional reset to "frontier" states ([paper](https://arxiv.org/abs/2603.09036) abstract, §4).
- Reports success rates on single target achievements, not total reward: Craftax-Classic Collect Diamond 88.2%, Craftax Enter Gnomish Mines 9.1% at 1B frames, 5 seeds ([paper](https://arxiv.org/abs/2603.09036) Table 1).
- Trained toward one target goal with an augmented symbolic observation, while baselines use the standard all-achievement reward, so it is not comparable with any reward block of [[Existing Implementations]].

## Method

- An LLM (GPT-5 Thinking Medium, temperature 0) proposes skills as operators with preconditions and effects, plus reward and termination code; proposals are filtered for feasibility and novelty ([paper](https://arxiv.org/abs/2603.09036) §4.1, App. D).
- A STRIPS planner chains trained skills to reach the preconditions of the target skill; each skill is a PPO option with its own mixture-of-experts head on a shared backbone, and composed policies are updated jointly with GAE across skill boundaries ([paper](https://arxiv.org/abs/2603.09036) §4.2-4.3).
- Pivotal Trajectory Analysis: the LLM revises preconditions, effects and resource counts from K = 10 successful or failed trajectories ([paper](https://arxiv.org/abs/2603.09036) §4.4).
- Frontier Checkpointing (FC variant): save the state where preconditions first hold and reset to it with probability 0.9 ([paper](https://arxiv.org/abs/2603.09036) §5.4, App. B.5).
- Observation for all methods: symbolic state (7 x 9 local map, inventory, vitals) plus relative offsets to the nearest blocks and mobs, so not the stock Craftax-Symbolic vector ([paper](https://arxiv.org/abs/2603.09036) App. D.1).
- Total LLM cost $2.83 (1.429M input, 125k output tokens); no LLM calls during RL training ([paper](https://arxiv.org/abs/2603.09036) App. D.6).

## Training setup and cost

| Hyperparameter | PPO FC / RNN / RND | PPO TrXL | Source |
|---|---|---|---|
| Learning rate | 2e-4, linear annealing | 2e-4, linear annealing | [paper](https://arxiv.org/abs/2603.09036) Table 3 |
| γ | 0.99 | 0.999 | Table 3 |
| GAE λ / clip ε | 0.8 / 0.2 | 0.8 / 0.2 | Table 3 |
| Entropy coefficient | 0.01 | 0.002 | Table 3 |
| Epochs / minibatches | 4 / 8 | 4 / 8 | Table 3 |
| Rollout steps / environments | 64 / 1024 | 128 / 1024 | Table 3 |
| Total frames | 1e9 | 1e9 | Table 3 |
| Optimistic reset ratio | 16 | 16 | Table 3 |

- Per-skill budget: 1e8 frames ([paper](https://arxiv.org/abs/2603.09036) App. D.2).
- Cost: about 1 hour per 1B-frame run on one H100 with 1024 environments ([paper](https://arxiv.org/abs/2603.09036) §5.1, App. D.6).
- Parameters: PPO-FC about 2.1M; SCALAR-FC about 2.1M shared plus about 0.8M per skill (8.5M to 11.7M in total for 8-12 skills, about 3.0M active); SCALAR-TRXL about 1.2M backbone plus about 0.3M per skill ([paper](https://arxiv.org/abs/2603.09036) App. D.3.7).
- PQN baselines: learning rate 1e-4 (MLP) or 3e-4 (RNN); TD-λ 0 or 0.5; ε 0.1→0.005 or 1.0→0.005 ([paper](https://arxiv.org/abs/2603.09036) Table 4).

## Results

Success rate (%) of the target achievement, 1B frames, mean of 5 seeds, spread not given in the table; curves are evaluated on 256 episodes with standard-error shading ([paper](https://arxiv.org/abs/2603.09036) Table 1, Fig. 2).

| Method | Classic: Eat Plant | Classic: Collect Diamond | Craftax: Enter Dungeon | Craftax: Unlock Gnomish Ladder | Craftax: Enter Gnomish Mines |
|---|---|---|---|---|---|
| SCALAR (FC) | 91.7 | 88.2 | 58.4 | 35.0 | 9.1 |
| without Frontier Checkpointing | 90.7 | 88.2 | 58.9 | 11.5 | 1.3 |
| also without Trajectory Analysis | 0.0 | 67.3 | 22.9 | 0.4 | 0.0 |
| SCALAR-TRXL | 0.0 | 45.0 | 85.2 | 34.1 | 8.2 |
| PPO-FC | 23.9 | 35.4 | 86.6 | 1.8 | 0.0 |
| PPO-RND | 21.9 | 34.5 | 87.4 | 1.9 | 0.0 |
| PPO-RNN | 0.0 | 40.9 | 89.7 | 0.0 | 0.0 |
| PPO-TRXL (transformerXL_PPO_JAX) | 0.0 | 31.5 | 96.5 | 0.4 | 0.0 |
| PQN | 19.0 | 0.0 | 77.4 | 0.8 | 0.0 |
| PQN-RNN | 55.3 | 46.9 | 92.5 | 0.0 | 0.0 |

- Baseline rows are third-party reruns of [[PPO (Matthews 2024)]], [[RND (Matthews 2024)]], [[PPO-RNN (Matthews 2024)]], [[PPO-GTrXL (Hamon 2024)]] and [[PQN-RNN (Gallici 2025)]] code, on the augmented observation ([paper](https://arxiv.org/abs/2603.09036) App. D.3).
- Online adaptation to modified recipes, diamond rate % for PPO / SCALAR / SCALAR + trajectory analysis ([paper](https://arxiv.org/abs/2603.09036) Table 2): unchanged 34.8 / 88.1 / 88.1; iron pickaxe needs 2 iron 5.1 / 5.4 / 86.1; table needs 3 wood 32.8 / 0.7 / 86.7.
- Total reward, % of max and Crafter score: not reported; all-achievement rates are figure only (Fig. 12-13).

## Problems and limitations

- Needs a predefined symbolic vocabulary, fixed skill order, and an LLM with good priors for the game ([paper](https://arxiv.org/abs/2603.09036) §6).
- Entropy regularisation causes spurious actions; SCALAR-TRXL fails on Eat Plant without explanation ([paper](https://arxiv.org/abs/2603.09036) §6, Table 1).
- Reset probability 0.999 overfits, with a 12-18% train-evaluation gap; Frontier Checkpointing needs state serialisation and is not used for TrXL ([paper](https://arxiv.org/abs/2603.09036) App. B.5, Fig. 9).
- Lags the baselines on Enter Dungeon because it skips combat that is not on the critical path ([paper](https://arxiv.org/abs/2603.09036) §5.2).
- Internal inconsistency: graduation threshold 30% in §5.1 and App. D.2, but "80% for standard skills" under Table 4.

## Code

- Not linked in the paper. Probable code: https://github.com/renos/flow-rl (first author's account; README describes the Craftax / Craftax-Classic skill pipeline, with `paper_experiments/` and `scripts/craftax_bottomup_wstatereset_shared.sh`); commit read in this pass: `e0d940f` (2026-01-23); no issues.
- Framework: JAX; baselines from Craftax_Baselines, purejaxql and transformerXL_PPO_JAX ([paper](https://arxiv.org/abs/2603.09036) App. D.3).
- Licence: none in the repository.
- Same first author: [[Feasibility Classification (Zabounidis 2026)]].
