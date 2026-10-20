---
title: "Efficient MBRL (Dedieu 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Antoine Dedieu
  - Joseph Ortiz
  - Xinghua Lou
  - Carter Wendelken
  - Wolfgang Lehrach
  - J. Swaroop Guntupalli
  - Miguel Lazaro-Gredilla
  - Kevin Murphy
year: 2025
venue: "ICML 2025"
paper: "https://arxiv.org/abs/2502.01591"
code: "not released (none linked in the paper or the Craftax scoreboard)"
envs:
  - Craftax-Classic-Pixels-v1
  - Craftax-Pixels-v1
tags:
  - implementation
  - craftax
  - craftax-classic
  - model-based
  - world-model
  - transformer
  - ppo
created: 2026-10-20
---
# Efficient MBRL (Dedieu 2025)

## TL;DR

- Paper "Improving Transformer World Models for Data-Efficient RL" (Google DeepMind); listed as "Efficient MBRL" on the Craftax scoreboard.
- PPO-RNN agent trained on real data plus imagined rollouts of a transformer world model (TWM), with three additions: Dyna with warm-up, patch nearest-neighbour tokenizer (NNT), block teacher forcing (BTF).
- Craftax-Classic (Pixels), 1M steps: 69.66 ± 1.20% reward, 31.77 ± 1.43% score; first agent above the human-expert reward of 65.0% ([paper](https://arxiv.org/abs/2502.01591) Table 1).
- Craftax (Pixels), 1M steps: 7.20 ± 0.09% reward in v2/v3; 5.44 ± 0.25% in v1, which is the scoreboard's 5.4 ([paper v3](https://arxiv.org/abs/2502.01591v3) Table 2; [v1](https://arxiv.org/abs/2502.01591v1) Table 2).
- Uses pixel observations, unlike the symbolic Craftax-1M protocol of the scoreboard.

## Method

- Model-free base ("MFRL"): PPO with an Impala-style CNN, a GRU of 256 units whose output is concatenated to the image embedding, and 2048-wide actor and critic MLPs with residual blocks and layer norm ([paper](https://arxiv.org/abs/2502.01591) §3.2, App. A.1, Table 3).
- Dyna with warm-up: the policy is updated on real trajectories every iteration and, after T_BP = 200k steps, also on TWM rollouts ([paper](https://arxiv.org/abs/2502.01591) §3.4, Algorithm 1).
- Patch tokenizer: the 63 x 63 image is split into its 9 x 9 grid of 7 x 7 tiles, each tokenised separately ([paper](https://arxiv.org/abs/2502.01591) §3.5).
- NNT: a patch becomes a new codebook entry if no existing code is within Euclidean threshold τ = 0.75; codes never change once created (codebook size 4096) ([paper](https://arxiv.org/abs/2502.01591) §3.5, §4.1, App. A.2.1).
- BTF: the TWM predicts all tokens of the next timestep jointly from the history and samples them in parallel; twice as fast as autoregressive generation ([paper](https://arxiv.org/abs/2502.01591) §3.6).
- TWM: GPT-2 style, 3 layers, 8 heads, embedding 128, RoPE, sequence length 20, key-value caching ([paper](https://arxiv.org/abs/2502.01591) App. A.2.2, Table 4).
- No intrinsic reward.

## Training setup and cost

| Setting | Craftax-Classic | Craftax (full) | Source |
|---|---|---|---|
| Observation | 63 x 63 pixels, 7 x 7 patches, 9 x 9 grid | 130 x 110 pixels, 10 x 10 patches, 13 x 13 grid | [paper](https://arxiv.org/abs/2502.01591) Table 6 |
| MFRL environments x rollout | 48 x 96 | 64 x 64 | Table 6 |
| MBRL environments x rollout | 48 x 96 | 16 x 96 | Table 6 |
| TWM rollout horizon | 20 | 20 | Table 6 |
| Replay buffer | 128,000 (Table 5) | see problems below | Tables 5, 6 |
| Environment steps | 1M | 1M | §4, §4.5 |

- PPO (MFRL, Table 3): γ 0.925, λ 0.625, clip 0.2, entropy 0.01, Adam lr 0.00045, max gradient norm 0.5, 8 minibatches, 4 epochs, linear LR annealing (none for MBRL, App. A.1.3).
- MBRL (Table 5): TWM updates 500 per iteration (4k for the "slow" agent), policy updates in imagination 150, NNT tokenizer updates 25, burn-in 5.
- Hardware: each experiment on 8 H100 GPUs ([paper](https://arxiv.org/abs/2502.01591) §4); the MFRL agent trains in 15 minutes on one A100 ([paper](https://arxiv.org/abs/2502.01591) §3.2).
- Wall-clock, Craftax-Classic ([paper](https://arxiv.org/abs/2502.01591) Table 1): best MBRL (fast) 759 min, best MBRL (slow) 2749 min (about 2 days), best MFRL 15 min.
- Wall-clock, Craftax (full): not reported.
- Parameters ([paper](https://arxiv.org/abs/2502.01591) Table 1): best MBRL 58.5M, best MFRL 55.6M; Craftax (full) models: not reported.
- GPU memory: TWM sequence length 20 is "the largest value that will fit into memory on 8 H100 GPUs" ([paper](https://arxiv.org/abs/2502.01591) App. A.2.2).

## Results

| Benchmark | Agent | Reward % (mean ± s.e.) | Score % (mean ± s.e.) | Seeds | Source |
|---|---|---|---|---|---|
| Craftax-Classic 1M, Pixels | best MBRL (slow) | 69.66 ± 1.20 | 31.77 ± 1.43 | 10 | [paper](https://arxiv.org/abs/2502.01591) Table 1 |
| Craftax-Classic 1M, Pixels | best MBRL (fast) | 67.42 ± 0.55 | 27.91 ± 0.63 | 10 | Table 1 |
| Craftax-Classic 1M, Pixels | best MFRL | 55.49 ± 1.33 | 16.77 ± 1.11 | 10 | Table 1 |
| Craftax-Classic 1M, Pixels | M1 baseline MBRL (IRIS-like) | 31.93 ± 2.22 | 4.98 ± 0.50 | 10 | Table 1 |
| Craftax 1M, Pixels | best MBRL (slow) | 7.20 ± 0.09 | 2.31 ± 0.04 | 10 | [paper v3](https://arxiv.org/abs/2502.01591v3) Table 2 |
| Craftax 1M, Pixels | best MBRL (v1) | 5.44 ± 0.25 | 1.53 ± 0.10 | 10 | [paper v1](https://arxiv.org/abs/2502.01591v1) Table 2 |
| Craftax 1M, Pixels | best MFRL | 4.63 ± 0.20 | 1.22 ± 0.07 | 10 | [paper](https://arxiv.org/abs/2502.01591) Table 2 |

- "±" is the standard error over 10 seeds ([paper](https://arxiv.org/abs/2502.01591) §4).
- Reward % = reward divided by the maximum (22 for Classic, 226 for Craftax); score = geometric mean of per-achievement success rates, weighting rare achievements more ([paper](https://arxiv.org/abs/2502.01591) §1 footnote 1, §4.1, Table 6).
- Reference points quoted in Table 1: human expert 65.0 ± 10.5 reward, DreamerV3 53.2 ± 8 (Crafter), the authors' DreamerV3 rerun 47.18 ± 3.88, IRIS 25.0, Δ-IRIS 35.0.
- Ablations (fast agent, ablation table captioned Fig. 4 in [paper](https://arxiv.org/abs/2502.01591) §4.3): no Dyna 55.02 ± 5.34; no NNT 60.66 ± 1.38; no BTF 64.96 ± 1.13; T_BP = 0 33.54 ± 10.09; continuous 7 x 7 patches 21.20 ± 0.55; MFRL without RNN 41.82 ± 0.97.
- Evaluation protocol for the Craftax numbers (training or separate test episodes): not reported.

## Problems and limitations

- NNT relies on the known tile grid of Craftax observations (ground-truth patch size 7 x 7); 5 x 5 and 9 x 9 patches reduce reward to 57.28 and 45.55 ([paper](https://arxiv.org/abs/2502.01591) §3.5, §4.3, Fig. 4).
- Without quantisation (continuous patches) performance collapses ([paper](https://arxiv.org/abs/2502.01591) §4.3).
- Starting imagination too early (T_BP = 0) collapses performance; every start shows a temporary drop as the policy meets imperfect TWM rollouts ([paper](https://arxiv.org/abs/2502.01591) §4.1, §4.3).
- Superhuman reward but score well below the human expert's 50.5 ([paper](https://arxiv.org/abs/2502.01591) §4.2, Table 1).
- TWM rollouts show map inconsistencies and infeasible hallucinations ([paper](https://arxiv.org/abs/2502.01591) §4.4).
- Craftax (full) result was revised between versions (5.44 in v1, 7.20 in v2/v3); the scoreboard still shows the v1 value.
- App. E text says the Craftax (full) buffer was reduced to 48k to fit in GPU, but Table 6 lists 48,000 for Classic and 128,000 for full, and Table 5 lists 128,000 for Classic; the paper is inconsistent.
- The scoreboard puts it under Craftax-1M, which the Craftax paper defines on symbolic observations; this agent uses pixels.
- Code: not released as far as this pass found.

## Code

- Repository: not released (Craftax scoreboard code column is "-"; no code link in the paper).
- Framework: PPO started from the PureJaxRL `ppo.py` (JAX) ([paper](https://arxiv.org/abs/2502.01591) App. A.1); framework of the world model: not reported.
- Licence: not applicable.
- Citation: Dedieu, A., Ortiz, J., Lou, X., Wendelken, C., Lehrach, W., Guntupalli, J. S., Lazaro-Gredilla, M., Murphy, K. "Improving Transformer World Models for Data-Efficient RL." ICML 2025. arXiv:2502.01591.
