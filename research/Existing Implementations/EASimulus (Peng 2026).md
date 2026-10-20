---
title: "EASimulus (Peng 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Zhao-Han Peng
  - Shaohui Li
  - Zhi Li
  - Shulan Ruan
  - Yu Liu
  - You He
year: 2026
venue: "ICLR 2026 (poster)"
paper: "https://arxiv.org/abs/2601.19336"
code: "https://github.com/MarquisDarwin/EAWM"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - model-based
  - world-model
  - transformer
  - auxiliary-loss
created: 2026-10-20
---
# EASimulus (Peng 2026)

## TL;DR

- Event-Aware World Model (EAWM) applied to [[Simulus (Cohen 2025)]]: an auxiliary loss that predicts automatically labelled "events" (changes in the observation), gated by a parameter-free event segmentor ([paper](https://arxiv.org/abs/2601.19336) §3).
- Craftax-1M (Symbolic): 7.23% of max reward, against about 6.6% for Simulus in the same figure ([paper](https://arxiv.org/abs/2601.19336) §4.2, Fig. 2b); repository per-seed file: 16.35 raw return, sd 0.43, 5 runs (computed in this pass).
- Comparable with block D of [[Existing Implementations]] (Symbolic, 1M); about 7 days per run on one RTX 4090 (App. M).

## Method

- Event labels from an automatic generator: thresholded change in log-brightness for pixels, normalised change for ordinal vectors, class change for categorical vectors; no hand labelling ([paper](https://arxiv.org/abs/2601.19336) §3.2, App. D).
- Generic Event Segmentor gates the event loss and weights the observation-prediction loss; event prediction uses focal and cross-entropy losses ([paper](https://arxiv.org/abs/2601.19336) §3.3).
- EASimulus: added to the Simulus RetNet token world model with per-modality events for the multimodal Craftax observation; the policy does not use event predictions ([paper](https://arxiv.org/abs/2601.19336) App. B, Table 12).
- A second variant, EADream (on DreamerV3), is evaluated only on Atari and DMC, not on Craftax.

## Training setup and cost

| Hyperparameter | Value | Source |
|---|---|---|
| Event-prediction weight β_e / observation weight β_o | 1.0 / 1.0 | [paper](https://arxiv.org/abs/2601.19336) Table 11 |
| Focal α / γ | 0.15 / 4 | Table 11 |
| Threshold / weight: image | 0.5 / 1.0 | Table 12 |
| Threshold / weight: ordinal | 1.0 / 0.1 | Table 12 |
| Threshold / weight: nominal, 2D categorical | 0.5 / 0.1 | Table 12 |
| World model | RetNet, 5 layers, 4 heads, embedding 256 | repo `EASimulus/config/world_model/craftax.yaml` |

- Other settings follow Simulus (official code, default hyperparameters; [paper](https://arxiv.org/abs/2601.19336) App. E.4); the ablations use 6 randomly selected Atari games, none on Craftax ([paper](https://arxiv.org/abs/2601.19336) §4.3).
- Hardware and wall-clock: RTX 4090 (24 GB); Craftax 1M about 7 days per run for EASimulus and about 6 days for Simulus ([paper](https://arxiv.org/abs/2601.19336) App. M).
- Parameters: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1M, Symbolic | reward, % of max (226) | 7.23 | not stated in the paper | [paper](https://arxiv.org/abs/2601.19336) §4.2, Fig. 2b |
| same, from the per-run file | final raw return, mean (sd across runs) | 16.35 (0.43), i.e. 7.23% (0.19%) | 5 | computed in this pass from [results/Craftax1M/EASimulus.json](https://github.com/MarquisDarwin/EAWM/blob/master/results/Craftax1M/EASimulus.json) (last point, 1,000,000 steps) |

- Per-run final returns: 16.02, 16.09, 17.01, 16.07, 16.55.
- Paper's comparison bars (no printed values): Simulus about 6.6%, Best-TWM ([[Efficient MBRL (Dedieu 2025)]] v1) about 5.4% (Fig. 2b); the conclusion claims +10% over Simulus.
- The repository's `Simulus.json` and `craftax_baselines.json` are identical copies of the files in the Simulus repository, not new reruns.
- Evaluation protocol: not stated in the paper; Simulus-style test episodes are likely (the repo's evaluation script uses 100 episodes).

## Problems and limitations

- Simple event segmentor; no multi-task model and no vision-language grounding ([paper](https://arxiv.org/abs/2601.19336) §6).
- Craftax gets one aggregate number with no seed count, spread, achievements or Craftax-Classic result in the paper.
- About a week per Craftax run on one RTX 4090.
- `scripts/train_craft.sh` passes keys (`motion_pred`, `amas`) that appear nowhere else in the code (unverified whether they have an effect).

## Code

- Repository: https://github.com/MarquisDarwin/EAWM (`EASimulus/scripts/train_craft.sh`, which runs `python src/main.py benchmark=craftax`; evaluation `scripts/eval_craft.sh`; checkpoint on Hugging Face `darwin05/EAWM`); commit read in this pass: `49eebbe` (2026-10-02). Issues #1-#3 closed (missing scripts, weights, a Papers with Code check).
- Framework: PyTorch 2.5.1, Hydra.
- Licence: GPL-3.0 (LICENSE file).
