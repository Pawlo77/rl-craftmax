---
title: "ICM (Matthews 2024)"
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
  - exploration
  - intrinsic-reward
created: 2026-10-20
---
# ICM (Matthews 2024)

## TL;DR

- PPO plus an Intrinsic Curiosity Module (forward-model prediction error as bonus); Craftax paper baseline.
- Craftax-1B (Symbolic): 11.9% of max reward; Craftax-1M (Symbolic): 2.2% ([README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard)); 10 seeds ([paper](https://arxiv.org/abs/2402.16801) Fig. 3, Fig. 6).
- The author states the paper's ICM results are wrong: they came from a commit whose bonus was only paid on episode termination ([Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4)).
- How to run it: [[Craftax Baselines]].

## Method

- Encoder: MLP with 3 hidden layers of `--icm_layer_size` mapping the observation to a latent of `--icm_latent_size` (`models/icm.py`, `ppo.py`).
- Forward model: predicts the next latent from the latent and the one-hot action.
- Inverse model: predicts the action from the current and next latents; trains the encoder to ignore what the agent cannot affect ([paper](https://arxiv.org/abs/2402.16801) §4.1).
- Intrinsic reward: mean squared forward-model error times `--icm_reward_coeff`, masked to zero on episode end in the current code (`ppo.py`).
- Reward fed to PPO: extrinsic + intrinsic, one value head (`ppo.py`).
- Base PPO and network as in [[PPO (Matthews 2024)]].

## Training setup and cost

| Hyperparameter | Craftax-1B ([paper](https://arxiv.org/abs/2402.16801) Table 8) | Craftax-1M ([paper](https://arxiv.org/abs/2402.16801) Table 12) | `ppo.py` default |
|---|---|---|---|
| Intrinsic reward coefficient β | 1.0 (of 0.1, 0.5, 1.0) | 10.0 (of 0.01 to 100.0) | 1.0 |
| World-model learning rate | 0.0003 | 0.0003 | 3e-4 |
| Forward loss coefficient | 1.0 | 0.1 | 1.0 |
| Inverse loss coefficient | 1.0 | 0.5 | 1.0 |
| World-model layer size | 256 | 256 | 256 |
| Latent size | 32 (of 32, 64, 128) | 16 (of 16, 32, 64, 128) | 32 |

- PPO hyperparameters: 1B as [paper](https://arxiv.org/abs/2402.16801) Table 7; 1M as Table 11 (random search, App. E.2).
- Exploration network update epochs: 4 (`--exploration_update_epochs` default in `ppo.py`).
- Hardware, wall-clock, parameters, memory: not reported per algorithm.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | reward, % of max (226) | 11.9 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [paper](https://arxiv.org/abs/2402.16801) Fig. 3 |
| Craftax-1M, Symbolic | reward, % of max (226) | 2.2 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [paper](https://arxiv.org/abs/2402.16801) Fig. 6 |

- Standard deviations: not reported.
- Corrected results: not published as of this pass (arXiv still at v2 of 3 June 2024; README unchanged).

## Problems and limitations

- Bug: the old code computed `error = (latent_next_obs - latent_next_obs_pred) * done[:, None]`, so the bonus was paid only on episode termination; the fix multiplies by `(1 - done)`, but "the ICM sweep was run on an old commit" and "the ICM results in the paper are wrong" ([Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4), January 2025).
- The issue was raised for Craftax-1M; whether the 1B ICM number is affected is not stated separately.
- An independent rerun with the Table 12 settings (5 seeds) did not reproduce the reported 1M performance; results only as a plot in the issue ([Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4)).
- The paper cites ICM as Sukhbaatar et al. 2018 ([paper](https://arxiv.org/abs/2402.16801) §4.1); the README links Pathak et al. 2017 ([arXiv:1705.05363](https://arxiv.org/abs/1705.05363)).
- No gain over PPO at 1B ([paper](https://arxiv.org/abs/2402.16801) §4.3).

## Code

- Repository: https://github.com/MichaelTMatthews/Craftax_Baselines (`python ppo.py --train_icm`).
- Framework: JAX, Flax, optax, distrax.
- Licence: MIT.
- Citation: Matthews et al., "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning", ICML 2024, arXiv:2402.16801; method: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", arXiv:1705.05363.
