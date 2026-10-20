---
title: "Concept Grounding (Adamski 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Max Adamski
year: 2026
venue: "NeSy 2026 (PMLR), per the repository's BibTeX"
paper: "https://openreview.net/forum?id=10IqdKCR6N"
code: "https://github.com/maxadamski/concept-grounding"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - neuro-symbolic
  - auxiliary-loss
  - action-masking
created: 2026-10-20
---
# Concept Grounding (Adamski 2026)

## TL;DR

- "Abstraction and Knowledge in Reinforcement Learning: Evaluating Concept Grounding on Craftax": each 83-dim map tile is encoded to a 32-dim embedding that a one-layer decoder must classify into expert-defined binary concepts; the loss is added to PPO ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) Methods).
- Craftax (Symbolic), 500M steps: average score 11.87% ± 0.18 of max reward against 11.62% ± 0.13 for the PPO baseline (mean ± sd over 20 agents), with half the parameters (4.39M against 9.54M) ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) Table 1).
- 500M budget, action masking on all methods and 100k sampled evaluation episodes per agent: not comparable with the 1B blocks of [[Existing Implementations]].

## Method

- Encoder: 3 layers of 83 units (ReLU) per tile, output 32-dim embedding; flattened embeddings are concatenated with the 51-dim global state (observation 3219-dim instead of 8268) ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) App. "Architectural details").
- Grounding: a single-layer decoder predicts expert-defined binary propositions (a concept taxonomy) with a weighted binary cross-entropy; total loss L_PPO + 0.5 L_K ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) Methods, Results).
- Compared variants: baseline (raw observation), compression (encoder only), autoencoder (symmetric decoder), grounding; plus a shuffled-taxonomy control (App. E).
- Actor and critic: 3 x 512 ReLU MLPs, no recurrence.
- Illegal actions are masked by setting their logits to −∞ ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) App. "Training details" footnote); the mask in `env.py` (`policy_mask`) also always disables REST ("rest is useless?") and other actions, so the effective action set is reduced for every method.

## Training setup and cost

| Hyperparameter | Value ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) App. "Training details") |
|---|---|
| Steps | 500M |
| Environments x rollout | 1024 x 64 (batch 65,536) |
| Minibatches / epochs | 16 (4096 steps each) / 3 |
| Optimiser | AdamW, constant lr 2e-4 |
| GAE λ / γ | 0.8 / 0.99 |
| Max gradient norm | 1.0 |
| Clip ε / value / entropy coefficient | 0.2 / 0.5 / 0.01 |
| Concept loss weight λ_K | 0.5 |

- Wall-clock per agent: baseline 4.6 ± 0.6 h, compression 6.7 ± 0.2 h, autoencoder 7.9 ± 0.9 h, grounding 6.2 ± 0.4 h (PyTorch) ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) App. "Training details").
- Hardware: university GPUs; model not named. Parameters: Table 1.

## Results

Craftax (Symbolic), 500M steps, 20 agents per method; each evaluated on 100k episodes over 4096 worlds with sampled actions; score = total episodic reward as % of max; mean ± sd across agents ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) Table 1).

| Method | Parameters | Average score (% of max) | Maximum score (% of max) |
|---|---|---|---|
| Baseline (PPO) | 9.54M | 11.62 ± 0.13 | 22.75 ± 0.41 |
| Compression | 4.39M | 11.82 ± 0.19 | 25.24 ± 1.00 |
| Autoencoder | 4.42M | 11.77 ± 0.17 | 25.27 ± 1.20 |
| Grounding | 4.39M | 11.87 ± 0.18 | 26.42 ± 1.15 |

- Shuffled taxonomy: 11.65% ± 0.16 (App. E).
- Per-achievement rates with Mann-Whitney tests and Holm correction: App. D, Table 2; an average grounded agent enters the Gnomish Mines in only 230 of 100k evaluation episodes.
- The baseline row is a PyTorch, masked, non-recurrent PPO rerun; added to [[PPO (Matthews 2024)]] with that caveat.

## Problems and limitations

- No gain in average score; only maximum scores and robustness to seeds improve ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) Results, Discussion).
- Needs symbolic observations and knowledge engineering ([paper](https://github.com/maxadamski/concept-grounding/blob/main/paper/paper.tex) Discussion).
- No agent got beyond the Gnomish Mines; non-recurrent only.
- Not stated by the author: the hard action mask (including always-masked actions) changes the task.
- The OpenReview PDF is behind a browser check; this note read `paper/paper.tex` in the repository.

## Code

- Repository: https://github.com/maxadamski/concept-grounding (`train.py --agent agents/kb32.py --ground 0.5`, `test.py`, `scripts/train.sh`; vendored Craftax in `craftax/`); commit read in this pass: `62de337` (2026-09-02); no issues.
- Framework: PyTorch.
- Licence: none in the repository.
