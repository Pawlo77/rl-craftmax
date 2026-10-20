---
title: "PQN-RNN (Gallici 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Matteo Gallici
  - Mattie Fellows
  - Benjamin Ellis
  - Bartomeu Pou
  - Ivan Masmitja
  - Jakob Nicolaus Foerster
  - Mario Martin
year: 2025
venue: "ICLR 2025"
paper: "https://arxiv.org/abs/2407.04811"
code: "https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_rnn_craftax.py"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - q-learning
  - recurrent
  - model-free
created: 2026-10-20
---
# PQN-RNN (Gallici 2025)

## TL;DR

- Parallelised Q-Network (PQN): online Q(λ)-learning with many vectorised environments, normalisation instead of a target network, and no replay buffer; RNN variant evaluated on Craftax-1B.
- Craftax-1B (Symbolic): 16.0% of max reward against 15.3% for PPO-RNN ([paper](https://arxiv.org/abs/2407.04811) §5.3; [Craftax scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard)); mean of 10 seeds ([paper](https://arxiv.org/abs/2407.04811) §5, App. C).
- More sample-efficient than PPO-RNN with similar training time ([paper](https://arxiv.org/abs/2407.04811) §5.3).

## Method

- Q-network trained online on the batch just collected from parallel environments, in minibatches and epochs, with no target network and no replay buffer ([paper](https://arxiv.org/abs/2407.04811) §4).
- Stability from LayerNorm (and optionally ℓ2) in the Q-network; the paper proves LayerNorm + ℓ2 gives convergent TD even off-policy ([paper](https://arxiv.org/abs/2407.04811) §3).
- Targets: Q(λ) returns ([paper](https://arxiv.org/abs/2407.04811) §4, App. B).
- Exploration: ε-greedy, ε decayed from 1.0 ([paper](https://arxiv.org/abs/2407.04811) Table 4).
- RNN for Craftax: "the same as the MLP one, with an additional LSTM layer before the last layer" ([paper](https://arxiv.org/abs/2407.04811) App. C); code uses `nn.OptimizedLSTMCell` and feeds the last action as input (`pqn_rnn_craftax.py`, `ADD_LAST_ACTION`).
- Input normalisation: BatchNorm on the input layer; best results with BatchNorm on the input and LayerNorm in the rest of the network ([paper](https://arxiv.org/abs/2407.04811) §5.5, Fig. 6b).
- Optimiser: RAdam ([paper](https://arxiv.org/abs/2407.04811) App. C).
- Craftax implementation details otherwise follow the Craftax paper ([paper](https://arxiv.org/abs/2407.04811) App. C).

## Training setup and cost

| Hyperparameter | [Paper](https://arxiv.org/abs/2407.04811) Table 4 | Repo `config/alg/pqn_rnn_craftax.yaml` |
|---|---|---|
| Environments | 1024 | 1024 |
| Steps per update | 128 | 128 |
| ε start / finish | 1.0 / 0.005 | 1.0 / 0.005 |
| ε decay (fraction of updates) | 0.1 | 0.1 |
| Minibatches | 4 | 4 |
| Epochs | 4 | 4 |
| Normalise input | True | True |
| Normalisation type | batch_norm | layer_norm |
| Hidden size | 512 | 512 |
| Layers / RNN layers | 1 / 1 | 1 / 1 |
| Add last action | True | True |
| Learning rate | 0.0003, linear decay | 0.0003, linear decay |
| Max gradient norm | 0.5 | 0.5 |
| Reward scale | 1.0 | 1.0 |
| γ | 0.99 | 0.99 |
| λ | 0.5 | 0.5 |

- Hardware: single NVIDIA A40, whole pipeline JIT-compiled on the GPU ([paper](https://arxiv.org/abs/2407.04811) App. C).
- Wall-clock: "similar" to PPO-RNN; exact time not reported ([paper](https://arxiv.org/abs/2407.04811) §5.3).
- Against a GPU replay buffer of 1M transitions (Flashbax): same final performance, about 6x longer training, about 31 GB of GPU memory saved; a buffer of the ~8000-float Craftax observations would need about 30 GB ([paper](https://arxiv.org/abs/2407.04811) §5.3, §5.5, Table 2, Fig. 6d).
- Parameters: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic, PQN-RNN | reward, % of max (226) | 16.0 | 10 | [paper](https://arxiv.org/abs/2407.04811) §5.3, Fig. 5b; [Craftax scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard) |
| Craftax-1B, Symbolic, PQN (MLP, no RNN) | reward, % of max (226) | not reported as a number (curve vs PPO) | 10 | [paper](https://arxiv.org/abs/2407.04811) App. D, Fig. 16a |

- Standard deviation: not reported.
- Craftax-1M: not reported.
- Evaluation protocol for Craftax (training episodes or greedy test episodes): not stated; the config has test-time evaluation off by default (`TEST_DURING_TRAINING: False`).

## Problems and limitations

- ε-greedy exploration is "too simple" for hard-exploration problems (said for Atari; [paper](https://arxiv.org/abs/2407.04811) §5.2).
- Paper (batch_norm) and repository (layer_norm) disagree on the normalisation type for Craftax.
- Open bug report: the backward-scan Q(λ) target uses Q(s_T) where Q(s_{T-1}) is needed for the second-to-last step, so every earlier target in the rollout is off; reported for `pqn_gymnax.py` and "the equivalent block in the other `pqn_*.py` scripts" ([purejaxql issue #12](https://github.com/mttga/purejaxql/issues/12), August 2026, open).
- Agents failed to learn when MinAtar was run with 20 or more parallel seeds; the maintainer suspects JAX and did not fix it ([purejaxql issue #9](https://github.com/mttga/purejaxql/issues/9)).
- All environments share the same ε at a given step (by design; [purejaxql issue #11](https://github.com/mttga/purejaxql/issues/11)).

## Code

- Repository: https://github.com/mttga/purejaxql (`python purejaxql/pqn_rnn_craftax.py +alg=pqn_rnn_craftax`; MLP: `pqn_craftax.py`); commit read in this pass: `47af6d7` (2025-11-12).
- Framework: JAX, Flax, Hydra configs.
- Licence: Apache-2.0.
- Citation: Gallici, M., Fellows, M., Ellis, B., Pou, B., Masmitja, I., Foerster, J. N., Martin, M. "Simplifying Deep Temporal Difference Learning." ICLR 2025. arXiv:2407.04811.
