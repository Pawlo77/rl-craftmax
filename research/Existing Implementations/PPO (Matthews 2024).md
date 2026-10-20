---
title: "PPO (Matthews 2024)"
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
code: "https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo.py"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - model-free
created: 2026-10-20
---
# PPO (Matthews 2024)

## TL;DR

- Feed-forward PPO baseline of the Craftax paper, built on PureJaxRL; the whole pipeline (env + learner) is JIT-compiled on one GPU.
- Craftax-1B (Symbolic): 11.9% of max reward; Craftax-1M (Symbolic): 2.2% ([Craftax README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard)); 10 seeds ([paper](https://arxiv.org/abs/2402.16801) Fig. 3, Fig. 6).
- How to run it and the evaluation protocol: [[Craftax Baselines]].

## Method

- Algorithm: PPO with clipped objective, GAE and linear learning-rate annealing ([paper](https://arxiv.org/abs/2402.16801) §4.1, App. E; `ppo.py`).
- Symbolic network: separate actor and critic MLPs on the flattened observation, width 512, tanh ([paper](https://arxiv.org/abs/2402.16801) §4.1 says "4 layer MLP of width 512"; `models/actor_critic.py` `ActorCritic`: 3 hidden layers of 512 plus an output layer per head).
- Pixel network: `ActorCriticConv` (3 conv layers, 32 filters, 5x5, each followed by 3x3 max-pool) in `ppo.py`; no reported result uses it.
- Convolutions on the map part of the symbolic observation were tried; fully connected networks performed better ([paper](https://arxiv.org/abs/2402.16801) §4.1).
- No memory, no intrinsic reward.
- Optimistic resets: one new world per 16 workers, reset states sampled from that pool ([paper](https://arxiv.org/abs/2402.16801) App. C; `--optimistic_reset_ratio 16`); about 2x speed-up on Craftax (App. C).

## Training setup and cost

| Hyperparameter | Craftax-1B ([paper](https://arxiv.org/abs/2402.16801) Table 7) | Craftax-1M ([paper](https://arxiv.org/abs/2402.16801) Table 11) | `ppo.py` default |
|---|---|---|---|
| Environment workers | 1024 | 256 | 1024 |
| Steps per rollout | 64 | 16 | 64 |
| Update epochs | 4 | 4 | 4 |
| Minibatches | 8 | 8 | 8 |
| Learning rate | 0.0002 | 0.0003 | 2e-4 |
| γ | 0.99 | 0.99 | 0.99 |
| GAE λ | 0.8 | 0.8 | 0.8 |
| Clip ε | 0.2 | 0.2 | 0.2 |
| Entropy coefficient | 0.01 | 0.01 | 0.01 |
| Value coefficient | 0.5 | 0.5 | 0.5 |
| MLP layer size | 512 | 512 | 512 |
| Activation | tanh | tanh | tanh |
| Anneal LR | True | True | True |
| Max gradient norm | not reported | not reported | 1.0 |

- Tuning 1B: one seed per configuration, one hyperparameter at a time from an informal start ([paper](https://arxiv.org/abs/2402.16801) App. E.1).
- Tuning 1M: random search over PPO hyperparameters ([paper](https://arxiv.org/abs/2402.16801) App. E.2).
- Even at 1M steps, more parallel workers performed better ([paper](https://arxiv.org/abs/2402.16801) §3.4).
- Hardware: speed tests on one GeForce RTX 4090 (24 GB), i9-13900K, 32 GB RAM ([paper](https://arxiv.org/abs/2402.16801) App. B).
- Wall-clock: "a run on a single RTX 4090 can finish in an hour" for 1B steps ([paper](https://arxiv.org/abs/2402.16801) §3.4); a benchmark-level statement, per-algorithm wall-clock not reported.
- Best-case end-to-end throughput: 266,961 steps/s on Craftax and 405,618 steps/s on Craftax-Classic, both with 4096 workers ([paper](https://arxiv.org/abs/2402.16801) Table 1).
- Pixels: the author estimates about 10k steps/s for PPO with pixels, "about a day" for 1e9 steps ([Craftax issue #34](https://github.com/MichaelTMatthews/Craftax/issues/34)).
- Parameters: not reported. GPU memory: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | reward, % of max (226) | 11.9 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); curve in [paper](https://arxiv.org/abs/2402.16801) Fig. 3 |
| Craftax-1M, Symbolic | reward, % of max (226) | 2.2 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); curve in [paper](https://arxiv.org/abs/2402.16801) Fig. 6 |

- The paper gives curves only (shaded ±1 standard error); the exact values are the README's; standard deviations: not reported.
- Metric in code: mean return of the episodes that finish inside each training batch, stochastic policy (`ppo.py`, `info["returned_episode"]`); no separate evaluation run.
- Basic achievements are learned quickly; no progress in the Advanced and Very Advanced categories ([paper](https://arxiv.org/abs/2402.16801) §4.3, Fig. 4).
- At 1M, all five baselines perform similarly and only very basic achievements progress ([paper](https://arxiv.org/abs/2402.16801) §4.4).
- Craftax-Classic: the abstract credits PPO with 90% of optimal reward at 1B steps in under an hour, but App. F.3 shows PPO-RNN; see [[PPO-RNN (Matthews 2024)]].

Third-party reruns found in the 2026-10-20 pass (protocols differ; details in the linked notes):

| Source | Benchmark | Metric | Value | Seeds |
|---|---|---|---|---|
| [[RePPO (Nishimori 2026)]] (Craftax_Baselines code and settings) | Craftax-1B, Symbolic | % of max, 100 evaluation episodes per seed, mean (sd) | 11.66 (0.33) with entropy 0.01; 9.31 (1.00) without | 5 |
| [[Feasibility Classification (Zabounidis 2026)]] (unmasked PPO-MLP) | Craftax 1B, Symbolic; Craftax-Classic 1B, Symbolic | raw return, 10,240 stochastic evaluation episodes per seed, mean ± sd | 26.9 ± 0.5 (11.9%); Classic 19.2 ± 0.0 (87.3%) | 4 |
| [[SimBa (Lee 2025)]] (repository CSV) | Craftax-1B, Symbolic | `avg_return` (raw), mean (sd) | 25.74 (2.06), i.e. 11.4% | 5 |
| [[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]] (PyTorch port) | Craftax-1B, Symbolic | training-episode return (raw), mean ± sd | 26.83 ± 1.31 (11.9%) | 3 |
| [[Concept Grounding (Adamski 2026)]] (PyTorch, action-masked, 3 epochs, 16 minibatches) | Craftax 500M, Symbolic | % of max, 100k sampled evaluation episodes per agent, mean ± sd | 11.62 ± 0.13 | 20 |
| [[SGRL (Qi 2025)]] (lr 7e-4, γ 0.97, 256 environments) | Craftax-Classic 5M | raw reward / score %, mean ± (type not stated) | 11.9 ± 1.1 / 24.8 ± 5.7 | 5 |
| [[Rule Learner Planner (dargilboa 2026)]] (64 environments) | Craftax-Classic, Symbolic, 300k / 1M | Crafter score of own training episodes, % | 3.6 / 5.0 | 3 |
| [[SCALAR (Zabounidis 2026)]] (PPO-FC, augmented observation) | Craftax-Classic and Craftax 1B | target-achievement success, % | e.g. Collect Diamond 35.4, Enter Gnomish Mines 0.0 | 5 |

- Figure-only PPO reruns (no numbers) appear in SimBa (Fig. 10), Adam-Rel (Craftax-Classic, Fig. 3), OPEN (Craftax-Classic 30M, Fig. 5) and C-TeC; see the rejected list in [[Existing Implementations]].

## Problems and limitations

- No progress into the second floor (Gnomish Mines) at 1B steps ([paper](https://arxiv.org/abs/2402.16801) §4.3).
- `--save_policy` raises `ValueError: Unrecognized name format`; the orbax checkpointing it uses is deprecated (open [Baselines issue #2](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/2)).
- `OptimisticResetVecEnvWrapper` relied on undefined `jax.random.choice` behaviour when fewer workers finish than the reset pool size; fixed in commit `7ce36fa` (March 2026); the author judges that the affected states were discarded anyway ([Baselines issue #11](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/11)).
- Compilation broke on the `jax.tree_map` deprecation; fixed ([Baselines issue #7](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/7)).
- Weights & Biases logging is on by default (`--use_wandb`, `ppo.py`).

## Code

- Repository: https://github.com/MichaelTMatthews/Craftax_Baselines (`python ppo.py`); commit read in this pass: `7ce36fa` (2026-06-16).
- Framework: JAX, Flax, optax, distrax (`requirements.txt`).
- Licence: MIT.
- Citation: Matthews, M., Beukman, M., Ellis, B., Samvelyan, M., Jackson, M., Coward, S., Foerster, J. "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning." ICML 2024. arXiv:2402.16801.
