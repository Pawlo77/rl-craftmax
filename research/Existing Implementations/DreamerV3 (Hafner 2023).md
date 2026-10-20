---
title: "DreamerV3 (Hafner 2023)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Danijar Hafner
  - Jurgis Pasukonis
  - Jimmy Ba
  - Timothy Lillicrap
year: 2023
venue: "arXiv preprint (2301.04104, latest version Apr 2024)"
paper: "https://arxiv.org/abs/2301.04104"
code: "https://github.com/danijar/dreamerv3"
envs:
  - Craftax-Classic-Pixels-v1
  - Craftax-Classic-Symbolic-v1
tags:
  - implementation
  - craftax-classic
  - model-based
  - world-model
  - recurrent
created: 2026-10-20
---
# DreamerV3 (Hafner 2023)

## TL;DR

- General model-based agent: a recurrent state-space world model learned from experience, with actor and critic trained on imagined trajectories, and fixed hyperparameters across domains ([paper](https://arxiv.org/abs/2301.04104) abstract).
- The DreamerV3 paper reports the original Crafter, not Craftax; the Crafter number (reward 53.2 ± 8%, score 14.5 ± 1.6%) is often quoted next to Craftax-Classic results but marked as Crafter ([[Efficient MBRL (Dedieu 2025)]] Table 1).
- The only paper-backed Craftax-Classic number is a third-party rerun by Dedieu et al.: reward 47.18 ± 3.88% at 1M steps, Pixels, 201M parameters, 2100 min ([Dedieu paper v3](https://arxiv.org/abs/2502.01591v3) Table 1).
- Comparable with block F of [[Existing Implementations]] (Craftax-Classic, Pixels, 1M).

## Method

- World model: recurrent state-space model learned from experience ([paper](https://arxiv.org/abs/2301.04104) abstract).
- Actor and critic trained on imagined rollouts of the world model, with robustness techniques so one fixed hyperparameter set works across domains ([paper](https://arxiv.org/abs/2301.04104) abstract).
- The Craftax-Classic rerun uses 201M parameters ([Dedieu paper](https://arxiv.org/abs/2502.01591) Table 1).

## Training setup and cost

- Dedieu rerun: "based on the code from the author", 201M parameters, 2100 min for 1M environment steps; differs slightly from the published Crafter number "perhaps due to hyperparameter discrepancies" ([Dedieu paper](https://arxiv.org/abs/2502.01591) Table 1 and caption). The GPU for this row is not stated (the paper says its own experiments run on 8 H100 GPUs, §4; single-A100 timings are marked only for IRIS and Δ-IRIS).
- Repository-only rerun (dargilboa): `medium` size on the symbolic vector of Craftax-Classic, trained on the reward ([explanatorium RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md) B8, caveats).

## Results

| Source | Benchmark | Metric | Value | Seeds | Location |
|---|---|---|---|---|---|
| Dedieu et al. rerun ("Our DreamerV3") | Craftax-Classic 1M, Pixels | reward, % of max (22), mean ± s.e. | 47.18 ± 3.88 | not stated for this row (paper: 10 seeds, mean ± s.e.) | [Dedieu paper v3](https://arxiv.org/abs/2502.01591v3) Table 1 |
| same | same | score, % | not reported | | Table 1 |
| DreamerV3 paper (quoted) | original Crafter 1M | reward % / score % | 53.2 ± 8 / 14.5 ± 1.6 (Crafter, not Craftax) | | Dedieu Table 1, marked * |
| dargilboa rerun (repository) | Craftax-Classic, Symbolic, 300k | Crafter score of own training episodes, % | 2.6 (2.6-2.7) | 3 | [explanatorium headline table](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/figures/headline_table.md) |

- Quoted by [[ITC (Kim 2026)]] and [[Efficient MBRL (Dedieu 2025)]]: both agents exceed DreamerV3 on Craftax-Classic at 1M.

## Problems and limitations

- No Craftax or Craftax-Classic result from the DreamerV3 authors; every number here is a third-party run.
- Dedieu's rerun is below the published Crafter value, attributed to hyperparameter discrepancies ([Dedieu paper](https://arxiv.org/abs/2502.01591) Table 1 caption).
- Slow: 2100 min per 1M steps against 15-26 min for the model-free agents in the same table ([Dedieu paper](https://arxiv.org/abs/2502.01591) Table 1).
- The repository-only rerun uses a smaller model on symbolic input and is below the Crafter paper's DreamerV2 at the same budget ([explanatorium RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md) caveats).

## Code

- Repository: https://github.com/danijar/dreamerv3 (no Craftax configuration; code search in this pass found none).
- Framework: JAX.
- Licence: MIT.
