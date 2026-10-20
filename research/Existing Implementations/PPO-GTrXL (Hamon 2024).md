---
title: "PPO-GTrXL (Hamon 2024)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Gautier Hamon
year: 2024
venue: "Software release (HAL hal-04659863); method paper: Parisotto et al., ICML 2020"
paper: "https://arxiv.org/abs/1910.06764"
code: "https://github.com/Reytuag/transformerXL_PPO_JAX"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - transformer
  - memory
  - model-free
created: 2026-10-20
---
# PPO-GTrXL (Hamon 2024)

## TL;DR

- PPO with a gated Transformer-XL policy (GTrXL, Parisotto et al.), in JAX on the PureJaxRL template; no accompanying paper, results are in the repository README.
- Craftax-1B (Symbolic): 18.3% of max reward, top of the 1B scoreboard ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax); [Craftax scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard)).
- 4B steps: 20.6% ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax)).
- About 6 h 30 min for 1B steps on one A100 ([README](https://github.com/Reytuag/transformerXL_PPO_JAX); [issue #2](https://github.com/Reytuag/transformerXL_PPO_JAX/issues/2)).

## Method

- Policy and value share a Transformer-XL trunk with GRU-style gating (`"gating": True`, `"gating_bias": 2.0`) and relative multi-head attention (`rel_multi_head.py`, `transformerXL.py`); actor and critic heads of 2 dense layers each (`ActorCriticTransformer` in `trainer_PPO_trXL_pmap.py`).
- Memory: the last `WINDOW_MEM` = 128 step embeddings are cached; at training time embeddings are recomputed over a `WINDOW_GRAD` = 64 window where gradients flow ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#config-parameters-); `train_PPO_trXL_pmap.py`).
- Size: 2 layers, 8 heads, embedding 256, qkv features 256, hidden layers 256 (`train_PPO_trXL_pmap.py` default config).
- Code adapted from Hugging Face's Transformer-XL and simplified following MarcoMeter's episodic-transformer-memory-ppo (simpler gradient propagation and positional encoding) ([README](https://github.com/Reytuag/transformerXL_PPO_JAX)).
- No intrinsic reward, no world model.
- Environment: `CraftaxSymbolicEnvNoAutoReset` when `ENV_NAME` is `"craftax"` (`trainer_PPO_trXL.py`).

## Training setup and cost

| Hyperparameter | Value (`train_PPO_trXL_pmap.py` default, the config the README says was used) |
|---|---|
| Learning rate | 2e-4, annealed |
| Environments | 1024 |
| Steps per update | 128 |
| Total steps | 1e9 |
| Update epochs | 4 |
| Minibatches | 8 |
| γ | 0.999 |
| GAE λ | 0.8 |
| Clip ε | 0.2 |
| Entropy coefficient | 0.002 |
| Value coefficient | 0.5 |
| Max gradient norm | 1.0 |
| Activation | ReLU |

- Parameters: about 5M ("a 5M transformer", [README](https://github.com/Reytuag/transformerXL_PPO_JAX)).
- Hardware: one A100; "more or less similar" time on 2 V100 ([issue #2](https://github.com/Reytuag/transformerXL_PPO_JAX/issues/2)).
- Wall-clock: about 6 h 30 min, about 23,000 s, for 1B steps with 1024 environments ([README](https://github.com/Reytuag/transformerXL_PPO_JAX); [issue #2](https://github.com/Reytuag/transformerXL_PPO_JAX/issues/2)).
- Multi-GPU: `train_PPO_trXL_pmap.py` uses all visible GPUs via `pmap` ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#training)).
- Tuning: "without much parameter search" ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax)).
- GPU memory: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | normalised return, % of max (226) | 18.3 | not stated; 5 per-seed result files in each of `results_craftax/base_A100_1e9` and `results_craftax/pmap2V100_1e9steps_memory128_layer2_gamma999` | [README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax); [repo results](https://github.com/Reytuag/transformerXL_PPO_JAX/tree/main/results_craftax) |
| Craftax 4B steps, Symbolic | same | 20.6 | not stated | [README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax) |
| Craftax 8B steps, Symbolic | same | "did not lead to significant improvement" | not stated | [README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax) |

- Standard deviation: not reported.
- At 1B, `ENTER_GNOMISH_MINES` is reached much more often than in the Craftax paper, and one seed reaches the third floor (Sewers) ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax)).
- At 4B, the agent visits the Sewers "a decent amount of time" and gets several Advanced achievements, which no Craftax-paper baseline reached even with 10B steps ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax)).
- The Craftax scoreboard marks results from outside the Craftax paper as reported, not verified ([Craftax README](https://github.com/MichaelTMatthews/Craftax#-scoreboard)).

Third-party reruns found in the 2026-10-20 pass (protocols differ; details in the linked notes):

| Source | Benchmark | Metric | Value | Seeds |
|---|---|---|---|---|
| [[PPO-LSTM (Jiralerspong 2026)]] (Modrax reimplementation, linear heads) | Craftax-1B, Symbolic | % of max, final checkpoint, 1024 evaluation episodes, mean ± s.e. | 17.79 ± 0.82 (18.61 ± 0.10 without one seed at 10.4) | 10 |
| [[DiCode (Mitsides 2026)]] (same agent as DiCode) | Craftax-Symbolic, 2B steps | final mean return (raw), 1024 held-out worlds, sampled actions, 8192-step cap | 41.59 (18.4%); no s.e. printed | 8 |
| [[AHC (Zhu 2026)]] (DiCode's student, fresh native runs) | Craftax-Symbolic, 0.996B / 2.005B | native return (raw), 1024 worlds x 10 stochastic rollouts, mean ± sample sd | 36.63 ± 1.42 / 41.91 ± 1.01 | 3 |
| [[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]] (PyTorch reproduction h007) | Craftax-1B, Symbolic | training-episode return (raw), mean ± sd | 19.01 ± 0.84 (8.4%), "46% of the published reference" | 3 |
| [[SCALAR (Zabounidis 2026)]] (PPO-TRXL, augmented observation) | Craftax-Classic and Craftax 1B | target-achievement success, % | e.g. Enter Dungeon 96.5, Enter Gnomish Mines 0.0 | 5 |

- New cost figures from [[PPO-LSTM (Jiralerspong 2026)]]: GTrXL reimplementation 4.62M parameters, about 8h45m per 1B-step run on an L40S and about 5h30m on an "A100L"; DiCode reports about 9 h for 2B steps on an RTX 6000 Ada ([[DiCode (Mitsides 2026)]]).
- [[Sable Inference Search (Chalumeau 2025)]] calls 18.3% the state of the art its base policy matched; [[maskedPPO-GTrXL (Doguhan58 2026)]] builds on this code with masking and reward shaping.

## Problems and limitations

- A bug in the version that produced the reported curves gave the position-wise MLP one layer instead of two plus layer norm; after the fix, "performances seem to remain similar"; the exact graph needs commit `db3e52a` ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax)).
- Seed count and evaluation protocol are not stated in the README.
- Saturates: 8B steps add no significant gain over 4B ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax)).
- Stored results and parameters are pickled `.npy` files that need `allow_pickle=True` to load ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#results-on-craftax)).
- `WINDOW_GRAD` must divide `NUM_STEPS` ([README](https://github.com/Reytuag/transformerXL_PPO_JAX#config-parameters-)).

## Code

- Repository: https://github.com/Reytuag/transformerXL_PPO_JAX (`python3 train_PPO_trXL_pmap.py`, with `ENV_NAME` set to `"craftax"`); commit read in this pass: `d914d55` (2026-02-16).
- Framework: JAX, Flax, PureJaxRL template.
- Licence: MIT.
- Citation: Hamon, G. "transformerXL_PPO_JAX", software, HAL hal-04659863, July 2024, https://inria.hal.science/hal-04659863; method: Parisotto et al., "Stabilizing Transformers for Reinforcement Learning", arXiv:1910.06764.
