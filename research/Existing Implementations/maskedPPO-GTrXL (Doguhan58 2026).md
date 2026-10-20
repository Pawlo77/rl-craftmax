---
title: "maskedPPO-GTrXL (Doguhan58 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Doguhan58 (GitHub handle; real name not given)
year: 2026
venue: "GitHub repository (lab rotation project; no paper)"
paper: "not available"
code: "https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - transformer
  - action-masking
  - reward-shaping
created: 2026-10-20
---
# maskedPPO-GTrXL (Doguhan58 2026)

## TL;DR

- PPO-GTrXL built on Reytuag's implementation ([[PPO-GTrXL (Hamon 2024)]]) plus invalid-action masking and a densified, shaped reward ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping)).
- Craftax-1B (Symbolic): 33.8% of max reward "in our best training run"; typical runs "70 ± 10" raw return (31% ± 4.4% of 226) ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#results)).
- Rejected from the scoreboard: the maintainer judged that masking plus reward shaping uses "quite extensive prior knowledge", so it "would not be a fair comparison" ([Craftax PR #52](https://github.com/MichaelTMatthews/Craftax/pull/52), closed).
- Not comparable with block A of [[Existing Implementations]]: prior knowledge in the action space and the reward, best-of-runs number.

## Method

- Agent: PPO with a gated Transformer-XL, following Reytuag's code ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping)).
- Invalid-action masking: per-step rules that mask actions whose preconditions fail (e.g. SLEEP only if energy < max, crafting only near a table, DESCEND only on a ladder) (`action_mask.md`, `src/modules/environments/action_mask_wrapper.py`).
- Reward shaping: repeated rewards for resources, kills (scaled by floor), armour, enchantments, level-ups and cleared floors, plus depth-based potential-based shaping φ(s) = 2.0 x floor with γ 0.999 (`rewards.md`, `src/modules/environments/reward_shaping_wrapper.py`).
- Reported `returned_episode_returns` exclude the shaping rewards ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#results)).
- Feeds the last action to the network (`add_last_action: True`, `src/arguments.py`).

## Training setup and cost

| Hyperparameter | Value (`src/arguments.py`) |
|---|---|
| Environment | Craftax-Symbolic-v1, optimistic resets (ratio 16) |
| Total steps | 1,000,000,000 |
| Environments | 1024 |
| Steps per rollout | 128 |
| Minibatches / epochs | 8 / 4 |
| Learning rate | 3e-4, linear decay |
| Max gradient norm | 1.0 |
| γ / GAE λ | 0.999 / 0.8 |
| Clip ε | 0.2 |
| Entropy / value coefficient | 0.01 / 0.5 |
| Transformer | embed 256, 8 heads, head dim 64, 2 layers, memory 256, gradient window 128 |
| Hidden size | 512 |

- Hardware and wall-clock: about 8 hours on an RTX 4090 (24 GB) for the default 1B-step run, under WSL2 ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#results)).
- Parameters: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | normalised return, % of max (226), unshaped reward | 33.8 (best training run) | best of an unstated number | [README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#results); [Craftax PR #52](https://github.com/MichaelTMatthews/Craftax/pull/52) |
| Craftax-1B, Symbolic | `returned_episode_returns` (raw) | "70 ± 10" (spread type not stated) | not stated | [README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#results) |

- Metric: training-episode returns logged to W&B (`returned_episode_returns`); evaluation is off by default (`enabled: False` in the evaluation config, `src/arguments.py`).
- Curve: `results/results.png` only.

## Problems and limitations

- Uses game knowledge in both the action mask and the reward; the author notes this "takes away from the sparse reward challenge aspect of Craftax" ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#reward-shaping)).
- "Training has high variance"; the headline is the best run, not a mean ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#results)).
- More than 1B steps not tested; trained on WSL2, which JAX calls experimental ([README](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping#results)).
- Random seed by default (`seed` drawn from 1-10000 in `src/arguments.py`), so runs are not reproducible without setting it.

## Code

- Repository: https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping (`uv run train.py`); commit read in this pass: `f728b95` (2026-04-21); includes a final checkpoint (`src/checkpoints/ppo_gtrxl_2818/`).
- Framework: JAX, Flax, uv.
- Licence: MIT.
- Reused by [[PPO-LSTM (Jiralerspong 2026)]] (action-mask code, credited).
