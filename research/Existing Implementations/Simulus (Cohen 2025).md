---
title: "Simulus (Cohen 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Lior Cohen
  - Kaixin Wang
  - Bingyi Kang
  - Uri Gadot
  - Shie Mannor
year: 2025
venue: "arXiv preprint (v4, May 2026)"
paper: "https://arxiv.org/abs/2502.11537"
code: "https://github.com/leor-c/Simulus"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - model-based
  - world-model
  - transformer
  - intrinsic-reward
created: 2026-10-20
---
# Simulus (Cohen 2025)

## TL;DR

- Token-based world-model agent (successor of REM): modular tokenizers, a RetNet world model with parallel next-token prediction, ensemble-disagreement intrinsic reward, prioritised world-model replay, and regression-as-classification for reward and value.
- Craftax-1M (Symbolic): 6.59% of max reward, top of the 1M scoreboard ([paper](https://arxiv.org/abs/2502.11537) Fig. 1; [Craftax scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard) lists 6.6); 5 seeds ([paper](https://arxiv.org/abs/2502.11537) §3.1).
- Costly: about 94 hours per Craftax run on one RTX 4090 ([paper](https://arxiv.org/abs/2502.11537) App. C).
- Earlier name: M3 (renamed in [Craftax PR #44](https://github.com/MichaelTMatthews/Craftax/pull/44)); the repository's citation still uses an earlier title, "Uncovering Untapped Potential in Sample-Efficient World Model Agents" (v4 changed the title, per its arXiv comment).

## Method

- Representation V: per-modality tokenizers; for Craftax the map is a "2D categorical" grid (9 x 11 tiles, 4 symbols each) and the 48 state features are continuous values quantised to 125 tokens after symlog ([paper](https://arxiv.org/abs/2502.11537) §2.1, App. A.1, A.3.2).
- One Craftax observation = 444 tokens arranged as 147 embeddings ([paper](https://arxiv.org/abs/2502.11537) §3.2).
- World model M: retentive network (RetNet) with the parallel observation prediction (POP) mechanism, predicting all next-observation tokens jointly, plus reward and termination heads ([paper](https://arxiv.org/abs/2502.11537) §2.2).
- Intrinsic reward: Jensen-Shannon divergence across an ensemble of 4 next-observation heads (epistemic uncertainty), given to the controller during imagination ([paper](https://arxiv.org/abs/2502.11537) §2.2).
- Prioritised world-model replay: a fraction α of each batch sampled by softmax of stored world-model losses ([paper](https://arxiv.org/abs/2502.11537) §2.2).
- Controller C: actor-critic trained only in imagination with λ-returns and REINFORCE with a value baseline; values and rewards predicted by HL-Gauss classification over 128 exponential bins ([paper](https://arxiv.org/abs/2502.11537) §2.3).
- Memory: recurrent state of the RetNet; retention decay range widened for Craftax to encourage long-term memory ([paper](https://arxiv.org/abs/2502.11537) App. A.2).

## Training setup and cost

| Hyperparameter (Craftax) | Value | Source |
|---|---|---|
| Epochs | 10,000 | [paper](https://arxiv.org/abs/2502.11537) Table 2 |
| Environment steps per epoch | 100 | Table 2 |
| Imagination horizon H | 20 | Table 2 |
| Observation sequence length K | 147 | Table 2 |
| Tokenizer vocabularies | (37, 5, 40, 20, 4, 125) | Table 2 |
| Batch size (M, C) | (8, 128) | Table 2 |
| Training steps per epoch (M, C) | (100, 50) | Table 2 |
| Training starts after epoch (M, C) | (250, 300) | Table 2 |
| World model layers / heads / dim | 5 / 4 / 256 | Table 3 |
| Retention decay range | [8, 40] | Table 3 |
| Dropout | 0.1 | Table 3 |
| Reward weights w_ext / w_int | 100 / 1 | Table 4 |
| Controller encoder MLP | [512, 512], shared backbone | Table 4 |
| Optimiser, learning rate (V, M, C) | AdamW, (1e-4, 2e-4, 2e-4) | Table 1 |
| Prioritised replay fraction α / initial loss ν0 | 0.3 / 10 | Table 1 |
| Ensemble size | 4 | Table 1 |
| Eval sampling temperature | 0.5 | Table 1 |

- Hardware: single RTX 4090 for Craftax ([paper](https://arxiv.org/abs/2502.11537) App. C).
- Wall-clock: about 94 hours (3.9 days) per Craftax run ([paper](https://arxiv.org/abs/2502.11537) App. C).
- Environment steps: 1M (10,000 epochs x 100 steps).
- Parameters: not reported. GPU memory: not reported.
- Tuning: limited by compute; remaining hyperparameters tuned empirically or taken from REM ([paper](https://arxiv.org/abs/2502.11537) App. A.2).

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1M, Symbolic | reward, % of max (226), mean of 100 test episodes at the end of training | 6.59 | 5 | [paper](https://arxiv.org/abs/2502.11537) Fig. 1, §3.1 |
| same, from the per-run file | raw final score, mean (sd across seeds) | 14.90 (1.11), i.e. 6.59% (0.49%) | 5 | computed in this pass from [results/data/craftax/Simulus.json](https://github.com/leor-c/Simulus/blob/main/results/data/craftax/Simulus.json) (last evaluation, epoch 10,000) |

- Baselines in the same figure (Simulus authors' reruns): PPO-RNN 2.13, RND 2.11, E3B 1.91; TWM of Dedieu et al. 5.44 (the v1 value of [[Efficient MBRL (Dedieu 2025)]]) ([paper](https://arxiv.org/abs/2502.11537) Fig. 1).
- Outperforms all model-free baselines in the Craftax-1M training curves ([paper](https://arxiv.org/abs/2502.11537) Fig. 5).

## Problems and limitations

- Craftax was excluded from the ablations for compute reasons, so the contribution of each component on Craftax is unknown ([paper](https://arxiv.org/abs/2502.11537) §3.3).
- Quantising continuous features gives "excessive sequence lengths" ([paper](https://arxiv.org/abs/2502.11537) §5).
- Token-based world-model agents remain slower to train than other sample-efficient baselines ([paper](https://arxiv.org/abs/2502.11537) §5).
- Little hyperparameter tuning; the authors expect further gains from tuning ([paper](https://arxiv.org/abs/2502.11537) App. A.2).
- Since August 2026 the repository default is Simulus-A (attention instead of RetNet, FSQ image tokenizer), which has not been evaluated on Craftax; reproducing the paper's numbers needs the original Simulus configuration ([repo README](https://github.com/leor-c/Simulus#updates); [Simulus issue #2](https://github.com/leor-c/Simulus/issues/2)).
- The README's Craftax checkpoint link points to the Hugging Face repository `leorc/M3` (old name) ([repo README](https://github.com/leor-c/Simulus)).

## Code

- Repository: https://github.com/leor-c/Simulus (`python src/main.py benchmark=craftax`); weights: https://huggingface.co/leorc/Simulus; commit read in this pass: `7577556` (2026-10-10).
- Framework: PyTorch, Hydra, Docker compose.
- Licence: GPL-3.0.
- Citation: Cohen, L., Wang, K., Kang, B., Gadot, U., Mannor, S. "Simulus: Combining Improvements in Sample-Efficient World Model Agents." arXiv:2502.11537 (v4, 2026).
