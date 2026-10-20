---
title: "PPO-LSTM (Jiralerspong 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Marco Jiralerspong
year: 2026
venue: "W&B report and pending Craftax scoreboard PR (no paper)"
paper: "https://api.wandb.ai/links/gauthier-gidel/qemj9p01"
code: "https://github.com/marcojira/modrax/blob/main/examples/ppo_recurrent_craftax.py"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - recurrent
  - model-free
created: 2026-10-20
---
# PPO-LSTM (Jiralerspong 2026)

## TL;DR

- Reimplementation of the [[PPO-GTrXL (Hamon 2024)]] setup in the Modrax library (JAX, Flax NNX), with the GTrXL layers swapped for one LSTM layer and small head changes ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01)).
- Craftax-1B (Symbolic): 19.54% ± 0.34 (s.e.) of max reward against 17.79% ± 0.82 for the authors' GTrXL rerun; 10 seeds each; 1024 evaluation episodes per checkpoint ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Final score").
- Trains in "less than half the time" of GTrXL on an L40S (3h40m against 8h45m per 1B-step run) ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Efficiency").
- Proposed as 19.5 on the Craftax-1B scoreboard in [Craftax PR #54](https://github.com/MichaelTMatthews/Craftax/pull/54) (open on 2026-10-20, not merged); not peer reviewed.

## Method

- PPO with a recurrent policy; hyperparameters "as faithful to the original as possible" to Reytuag's GTrXL implementation ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01)).
- LSTM network: Obs → Linear(512) → LSTM(512) → {LayerNorm → Linear(actions), LayerNorm → Linear(1)} ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Architecture").
- GTrXL comparison network: Obs → Linear(256) → GTrXL(256, 2 layers) → the same LayerNorm + linear heads; the original's MLP heads are replaced by linear layers ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01)).
- No action masking in the reported runs: `use_action_mask` defaults to `False` in `src/modrax/env/craftax.py`, and the example config does not set it.
- Optimistic resets on (`optimistic_reset=True`, `num_reset_envs` 16; `examples/ppo_recurrent_craftax.py`, `src/modrax/env/craftax.py`).

## Training setup and cost

| Hyperparameter | Value (`examples/ppo_recurrent_craftax.py`) |
|---|---|
| Environment | Craftax-Symbolic-v1 |
| Total steps | 1,000,000,000 |
| Environments | 1024 |
| Steps per rollout | 128 |
| Minibatches | 8 |
| Update epochs | 4 |
| Learning rate | 2e-4, linear decay |
| Gradient clip | 1.0 |
| γ | 0.999 |
| GAE λ | 0.8 |
| Entropy coefficient | 0.002 |
| LSTM hidden / layers | 512 / 1 |
| GTrXL (comparison) | hidden 256, 8 heads, 2 layers, segment 64, memory 128 |

- Hardware: one NVIDIA L40S per run ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Architecture").
- Wall-clock per 1B-step run including evaluation: about 3h45m for LSTM and 8h45m for GTrXL on an L40S; on an "A100L", GTrXL about 5h30m and LSTM about 3h40m ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Architecture", "Efficiency").
- Parameters: 6.36M (LSTM) against 4.62M (GTrXL) ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Efficiency").

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic, PPO-LSTM | reward, % of max (226), mean ± s.e. | 19.54 ± 0.34 | 10 | [W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Final score" |
| Craftax-1B, Symbolic, PPO-GTrXL (authors' rerun) | same | 17.79 ± 0.82 (18.61 ± 0.10 without one bad seed at 10.4) | 10 | same |

- Protocol: the 1B-step checkpoint is evaluated on 1024 parallel episodes, maximum length 25k steps; mean episodic return divided by 2.26 ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Final score"). Greedy or sampled actions: not stated; the code's `policy` samples from the softmax.
- Unlike the Craftax-paper baselines, this is a separate evaluation of the final checkpoint, not training-episode return.
- LSTM episodes are longer than GTrXL's (final evaluation length roughly 1800 against 1250 steps, bar chart only); LSTM enters the dungeon earlier, the Gnomish Mines more often, and eats plants in about 40% of episodes against about 20% ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Some Achievements").
- Informal extra findings (older code, "messier"): GRU slightly worse than LSTM; 2 LSTM layers worse than 1; embedding above 512 (up to 4096) does not help ([W&B report](https://api.wandb.ai/links/gauthier-gidel/qemj9p01), "Other Results").

## Problems and limitations

- No paper and no peer review; the authors call it "not worthy of a paper" ([Craftax PR #54](https://github.com/MichaelTMatthews/Craftax/pull/54)).
- The scoreboard PR is unmerged, so the number is not on the official scoreboard.
- Modrax is "under development and its API may change" ([README](https://github.com/marcojira/modrax)); the README also says the base dependencies install CUDA-enabled JAX.
- W&B logging is on by default (`WandbConfig(enabled=True)`).
- The README's report link (`rkdqficc`) and the PR's link (`qemj9p01`) point to the same report title; this note read the PR's link.

## Code

- Repository: https://github.com/marcojira/modrax (`uv run python examples/ppo_recurrent_craftax.py --architecture lstm --seed <seed>`); commit read in this pass: `ed6880f` (2026-09-07).
- Framework: JAX, Flax NNX, uv.
- Licence: MIT.
- Action-mask code (unused in these runs) is credited to [[maskedPPO-GTrXL (Doguhan58 2026)]].
