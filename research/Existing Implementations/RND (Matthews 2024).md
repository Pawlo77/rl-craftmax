---
title: "RND (Matthews 2024)"
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
code: "https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnd.py"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - exploration
  - intrinsic-reward
created: 2026-10-20
---
# RND (Matthews 2024)

## TL;DR

- PPO plus Random Network Distillation (Burda et al. 2018) as a global novelty bonus; Craftax paper baseline.
- Craftax-1B (Symbolic): 12.0% of max reward; Craftax-1M (Symbolic): 2.2% ([README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard)); 10 seeds ([paper](https://arxiv.org/abs/2402.16801) Fig. 3, Fig. 6).
- No gain over plain PPO at 1B ([paper](https://arxiv.org/abs/2402.16801) §4.3).
- How to run it: [[Craftax Baselines]].

## Method

- Intrinsic reward: squared error between a trained predictor network and a fixed random target network on the next observation, times a coefficient (`ppo_rnd.py`; [paper](https://arxiv.org/abs/2402.16801) §4.1).
- Target and predictor: MLPs with 3 hidden layers (`num_layers=3`, `--rnd_layer_size 256`) and output size `--rnd_output_size` (`models/rnd.py`, `ppo_rnd.py`).
- Policy network `ActorCriticRND`: separate extrinsic and intrinsic value heads (`models/rnd.py`).
- Advantage = extrinsic GAE + 0.01 x intrinsic GAE, then normalised (`ppo_rnd.py`, `RND_GAE_COEFF`).
- Intrinsic return is non-episodic by default (`--rnd_is_episodic False`).
- Base PPO and network as in [[PPO (Matthews 2024)]].

## Training setup and cost

| Hyperparameter | Considered ([paper](https://arxiv.org/abs/2402.16801) Table 10) | Chosen, 1B and 1M ([paper](https://arxiv.org/abs/2402.16801) Table 10) | `ppo_rnd.py` default |
|---|---|---|---|
| Intrinsic reward coefficient β | 0.01, 0.1, 1.0 | 1.0 | 1.0 |
| RND output dimension | 4, 32, 256, 2048 | 256 | 512 |
| Distillation network learning rate | 0.00003, 0.0003, 0.003 | 0.0003 | 3e-4 |
| RND loss coefficient | 0.0001, 0.001, 0.01 | 0.01 | 0.01 |
| RND GAE coefficient | 0.01, 0.1, 1.0 | 0.01 | 0.01 |

- PPO hyperparameters: those of [[PPO (Matthews 2024)]] ([paper](https://arxiv.org/abs/2402.16801) App. E.1).
- Exploration network update epochs: 1 (`--exploration_update_epochs` default).
- Hardware, wall-clock, parameters, memory: not reported per algorithm; see [[PPO (Matthews 2024)]] for the benchmark-level statements.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | reward, % of max (226) | 12.0 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [paper](https://arxiv.org/abs/2402.16801) Fig. 3 |
| Craftax-1M, Symbolic | reward, % of max (226) | 2.2 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [paper](https://arxiv.org/abs/2402.16801) Fig. 6 |
| Craftax-1M, Symbolic (third-party rerun) | reward, % of max (226) | 2.11 | 10 | [Simulus paper](https://arxiv.org/abs/2502.11537) Fig. 1; runs in [Simulus repo](https://github.com/leor-c/Simulus/blob/main/results/data/craftax/craftax_baselines.json) |

- Standard deviations: not reported; figures show ±1 standard error.
- None of the exploration methods improved reward at 1B; the authors hypothesise the reward is dense enough without a bonus, which may distract the agent ([paper](https://arxiv.org/abs/2402.16801) §4.3).

Third-party reruns found in the 2026-10-20 pass:

| Source | Benchmark | Metric | Value | Seeds |
|---|---|---|---|---|
| [[RePPO (Nishimori 2026)]] (PPO-V + RND, Craftax_Baselines settings) | Craftax-1B, Symbolic | % of max, 100 evaluation episodes per seed, mean (sd) | 11.68 (0.30) with entropy 0.01; 9.62 (1.54) without | 5 |
| [[Rule Learner Planner (dargilboa 2026)]] (PPO-RND with reward; RND reward-free) | Craftax-Classic, Symbolic, 300k / 1M | Crafter score of own training episodes, % | with reward 3.8 / 5.3; reward-free 1.4 / 1.3 | 3 |
| [[SCALAR (Zabounidis 2026)]] (PPO-RND, augmented observation) | Craftax-Classic and Craftax 1B | target-achievement success, % | e.g. Collect Diamond 34.5, Enter Gnomish Mines 0.0 | 5 |

- [[DLSD (Rajaraman 2025)]] reports a separate, DQN-based reward-free RND (Crafter score 1.65 at 2M steps); not this implementation.

## Problems and limitations

- No measurable benefit from the bonus at either budget ([paper](https://arxiv.org/abs/2402.16801) §4.3, §4.4).
- Bug: the intrinsic GAE was bootstrapped from the extrinsic last value (`last_val_e` instead of `last_val_i`); fixed in May 2025 ([Baselines PR #5](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/5)), after the paper (v2, June 2024), so the published RND numbers come from the unfixed code; corrected results: not published.
- The code default output dimension (512) differs from the tuned value in Table 10 (256); running the README command reproduces the code default, not Table 10.
- Shared wrapper and checkpoint issues are listed in [[PPO (Matthews 2024)]].

## Code

- Repository: https://github.com/MichaelTMatthews/Craftax_Baselines (`python ppo_rnd.py`).
- Framework: JAX, Flax, optax, distrax.
- Licence: MIT.
- Citation: Matthews et al., "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning", ICML 2024, arXiv:2402.16801; method: Burda et al., "Exploration by Random Network Distillation", arXiv:1810.12894.
