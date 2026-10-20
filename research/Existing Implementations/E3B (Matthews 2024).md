---
title: "E3B (Matthews 2024)"
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
# E3B (Matthews 2024)

## TL;DR

- PPO plus Exploration via Elliptical Episodic Bonuses (Henaff et al. 2022): an episodic novelty bonus in a learned embedding space; Craftax paper baseline.
- Craftax-1B (Symbolic): 11.0% of max reward; Craftax-1M (Symbolic): 2.2% ([README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard)); 10 seeds ([paper](https://arxiv.org/abs/2402.16801) Fig. 3, Fig. 6).
- Significantly lowers reward at 1B compared with PPO ([paper](https://arxiv.org/abs/2402.16801) §4.3).
- How to run it: [[Craftax Baselines]].

## Method

- Embedding: the ICM encoder, trained with the ICM inverse and forward losses; the ICM bonus itself is switched off (`--icm_reward_coeff 0`, asserted in `ppo.py`).
- Bonus: b = φᵀ C⁻¹ φ for the current embedding φ, with C⁻¹ updated each step by a rank-one (Sherman-Morrison) update (`ppo.py`).
- C⁻¹ is reset to I / λ at the start of every episode, so the bonus is episodic ([paper](https://arxiv.org/abs/2402.16801) §4.1; `ppo.py`).
- Reward fed to PPO: extrinsic + β x bonus; one value head.
- Base PPO and network as in [[PPO (Matthews 2024)]]; ICM networks as in [[ICM (Matthews 2024)]].

## Training setup and cost

| Hyperparameter | Craftax-1B ([paper](https://arxiv.org/abs/2402.16801) Table 9) | Craftax-1M ([paper](https://arxiv.org/abs/2402.16801) Table 13) | `ppo.py` default |
|---|---|---|---|
| Intrinsic reward coefficient β | 0.001 (of 0.0005, 0.001, 0.005) | 0.01 (of 0.0001 to 1.0) | 1.0 |
| Ridge regulariser λ | 0.1 (of 0.005, 0.1, 0.5) | 0.5 (of 0.001, 0.01, 0.1, 0.5) | 0.1 |

- PPO hyperparameters: 1B as [paper](https://arxiv.org/abs/2402.16801) Table 7; 1M as Table 11.
- ICM encoder settings for E3B: not reported separately.
- Hardware, wall-clock, parameters, memory: not reported per algorithm.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | reward, % of max (226) | 11.0 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [paper](https://arxiv.org/abs/2402.16801) Fig. 3 |
| Craftax-1M, Symbolic | reward, % of max (226) | 2.2 (raw 4.972) | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); raw value quoted in [Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4) |
| Craftax-1M, Symbolic (third-party rerun) | reward, % of max (226) | 1.91 (raw 4.306) | 10 | [Simulus paper](https://arxiv.org/abs/2502.11537) Fig. 1; [Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4) |

- Standard deviations: not reported.
- E3B achieves `EAT_PLANT` most often of all baselines, so the bonus does push exploration of other parts of the world ([paper](https://arxiv.org/abs/2402.16801) §4.3, Fig. 5).

- Third-party run (2026-10-20 pass): reward-free E3B on Craftax-Classic (Symbolic), Crafter score of its own training episodes 1.4% at 300k and 1.3% at 1M steps, 3 seeds ([[Rule Learner Planner (dargilboa 2026)]]; [headline table](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/figures/headline_table.md)); reward-free, so not comparable with the table above.

## Problems and limitations

- Lowest 1B reward of the five baselines; the authors suspect the bonus distracts the agent in a reward-dense game ([paper](https://arxiv.org/abs/2402.16801) §4.3).
- The README command (`python ppo.py --train_icm --use_e3b --icm_reward_coeff 0`) leaves β at the code default 1.0, not the tuned 0.001 (1B) or 0.01 (1M).
- An independent 10-seed rerun at 1M scored 1.9% against the reported 2.2% ([Baselines issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4)).
- The ICM-masking bug of [[ICM (Matthews 2024)]] concerned the ICM bonus; its effect on E3B is not stated.

## Code

- Repository: https://github.com/MichaelTMatthews/Craftax_Baselines (`python ppo.py --train_icm --use_e3b --icm_reward_coeff 0`).
- Framework: JAX, Flax, optax, distrax.
- Licence: MIT.
- Citation: Matthews et al., "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning", ICML 2024, arXiv:2402.16801; method: Henaff et al., "Exploration via Elliptical Episodic Bonuses", NeurIPS 2022, arXiv:2210.05805.
