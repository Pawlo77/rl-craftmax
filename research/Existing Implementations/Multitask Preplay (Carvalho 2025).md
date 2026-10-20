---
title: "Multitask Preplay (Carvalho 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Wilka Carvalho
  - Sam Hall-McMaster
  - Honglak Lee
  - Samuel J. Gershman
year: 2025
venue: "arXiv preprint (v2, May 2026)"
paper: "https://arxiv.org/abs/2507.05561"
code: "https://github.com/wcarvalho/multitask_preplay"
envs:
  - Craftax (custom 56-dim observation wrapper)
tags:
  - implementation
  - craftax
  - model-based
  - dyna
  - q-learning
  - recurrent
  - privileged-information
created: 2026-10-20
---
# Multitask Preplay (Carvalho 2025)

## TL;DR

- Cognitive-science paper with an AI part: Dyna-style background planning with a recurrent Q-network, where replay also simulates and optimises off-task goals that co-occur in the environment ("preplay") ([paper](https://arxiv.org/abs/2507.05561) Methods).
- Standard Craftax setting: 6.31% of max reward and geometric score 2.60, read from bar labels in a supplementary figure; seeds and step budget for this number not stated ([paper](https://arxiv.org/abs/2507.05561) Extended Data Fig. 26, PDF p. 53).
- Uses the true simulator as its world model and a hand-written goal-availability function, so it is privileged and not comparable with any block of [[Existing Implementations]].

## Method

- Recurrent Q-network (LSTM 256, MLP 1024) trained with goal-conditioned Q(λ), a CQL term and an "all-goals" loss ([paper](https://arxiv.org/abs/2507.05561) Methods, Table 1).
- Counterfactual simulation ("preplay") uses an oracle world model, i.e. the true simulator, for both Dyna and Multitask Preplay ([paper](https://arxiv.org/abs/2507.05561) Methods, "Counterfactual simulation"; Discussion).
- Goals come from φ_avail(s), a function marking which achievements are currently attainable, written with an LLM from the Craftax logic ([paper](https://arxiv.org/abs/2507.05561) Methods).
- Observation: a 56-dimensional normalised continuous vector, not the standard 8268-dimensional symbolic observation ([paper](https://arxiv.org/abs/2507.05561) Methods).
- The main AI experiment is generalisation to 10,000 held-out Craftax maps as the number of training environments varies (Fig. 5, figure only, non-standard protocol).

## Training setup and cost

| Hyperparameter | Value | Source |
|---|---|---|
| γ | 0.992 | [paper](https://arxiv.org/abs/2507.05561) Table 1 |
| Replay buffer / batch | 50k / 32 | Table 1 |
| Gradient clip | 80 | Table 1 |
| LSTM / MLP width | 256 / 1024 | Table 1 |
| Environments (Craftax) | 32 | Table 2 |
| Sequence length | 40 | Table 2 |
| Learning rate | 3e-4 | Table 2 |
| Simulations per step / length | 10 / 20 | Table 2 |
| CQL coefficient / goal weight / off-task goal weight | 1e-2 / 1 / 2 | Table 2 |
| ε-greedy | per environment, roughly 0.13 to 0.89 | Table 2 |

- Step budget for the standard-setting number: not reported; in the generalisation experiment model-based methods ran 1M steps and model-free 10M (Fig. 5). Repository config `configs/alg/preplay_craftax_ai.yaml` sets `TOTAL_TIMESTEPS` 40,000,000 (not confirmed as the reported run).
- Hardware, wall-clock and parameter count: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax, standard setting | reward, % of max (226), evaluation, mean per episode | 6.31 (bar label; error bar = SEM across runs) | not stated | [paper](https://arxiv.org/abs/2507.05561) Ext. Data Fig. 26A |
| same | geometric score, exp(mean ln(1 + s_i)) - 1 | 2.60 (bar label) | not stated | Ext. Data Fig. 26B |

- Same figure quotes, not reruns: TWM of [[Efficient MBRL (Dedieu 2025)]] 7.2 ± 0.09% / 2.31 ± 0.04, [[PPO-RNN (Matthews 2024)]] 2.3% (values hard-coded in `figures_supplemental/craftax_baselines_comparison.ipynb`).
- The plotting code takes values "at the best eval timepoint" (`figures/craftax_AI_results.py`, `get_best_scores_from_group`), an optimistic checkpoint choice.

## Problems and limitations

- Oracle world model; the authors say it should be learned in future work ([paper](https://arxiv.org/abs/2507.05561) Discussion).
- Privileged goal-availability function and a custom observation.
- Seeds and budget not given for the headline number; best-checkpoint selection.
- The authors attribute the gap to TWM to the RNN being less expressive than a transformer ([paper](https://arxiv.org/abs/2507.05561) Ext. Data Fig. 26 caption).

## Code

- Repository: https://github.com/wcarvalho/multitask_preplay (MIT, commit `37fa4d8`, 2025-08-26; human experiments and figures); AI code in the submodule https://github.com/wcarvalho/preplay-AI (no licence file, commit `46f14e8`, 2026-06-30). Craftax entry: `craftax_trainer.py` with `configs/config_craftax.yaml`, `configs/alg/preplay_craftax_ai.yaml`. No issues in either repository.
- Framework: JAX.
