---
title: "DiCode (Mitsides 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Konstantinos Mitsides
  - Maxence Faldor
  - Antoine Cully
year: 2026
venue: "ICML 2026"
paper: "https://arxiv.org/abs/2602.08194"
code: "https://github.com/konstantinosmitsides/dreaming-in-code"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - curriculum
  - unsupervised-environment-design
  - llm-guided
  - ppo
  - transformer
created: 2026-10-20
---
# DiCode (Mitsides 2026)

## TL;DR

- "Dreaming in Code": a foundation model (Qwen3-235B-A22B-Thinking) writes executable JAX level code (initial state, task parameters, goals) for a Craftax re-implementation, chosen by learnability from an archive; a PPO-GTrXL student trains on 20% target Craftax plus generated and replayed levels ([paper](https://arxiv.org/abs/2602.08194) §3).
- Craftax (Symbolic), 2B steps: final mean return 48.55 ± 0.49 (s.e., 8 seeds; 21.5% of 226) on 1024 held-out worlds, against 41.59 for PPO-GTrXL with the same agent ([paper v2](https://arxiv.org/abs/2602.08194) §4.2, Table 1).
- 2B budget (counting steps on generated levels) and a capped evaluation: not comparable with the 1B blocks of [[Existing Implementations]]; close to [[AHC (Zhu 2026)]], which uses the same student and protocol.

## Method

- Level space: a JAX "MiniCraftax" API with a WorldBuilder (15 methods) for the initial state, 12 TaskParams for dynamics, and goals as sets of Craftax achievements ([paper](https://arxiv.org/abs/2602.08194) §3.1, App. A.1).
- Generation cycle: archive selection by learnability, a natural-language description, then code; compile and runtime checks gate new levels; generation runs asynchronously ([paper](https://arxiv.org/abs/2602.08194) §3.2-3.3).
- The foundation model is conditioned on the parent level's code and the agent's performance on the parent and on standard Craftax (closed loop) ([paper](https://arxiv.org/abs/2602.08194) §3.2).
- Student: PPO-GTrXL, shared with all baselines; observations augmented with one-hot task conditioning ([paper](https://arxiv.org/abs/2602.08194) §4.1, Table 2).
- Training mix: 20% standard Craftax, the rest replayed and new levels (Table 5).

## Training setup and cost

| Hyperparameter (PPO-GTrXL) | Value ([paper](https://arxiv.org/abs/2602.08194) Table 2) |
|---|---|
| Environments x rollout | 1024 x 128 |
| Epochs / minibatches | 4 / 8 |
| Learning rate | 2e-4, annealed (to 2e-6 for DiCode, to 0 for baselines) |
| γ / GAE λ / clip ε | 0.999 / 0.8 / 0.2 |
| Entropy coefficient | 0.002 |
| GTrXL | embedding 256, 8 heads, 2 layers, memory 128 |

- Budget: 2e9 environment steps for every method, including generated levels for DiCode ([paper](https://arxiv.org/abs/2602.08194) §4.1).
- Hardware: RL on one RTX 6000 Ada; foundation model served on 4x RTX 6000 Blackwell ([paper](https://arxiv.org/abs/2602.08194) Tables 6, 7).
- Wall-clock: PPO-GTrXL about 9 h at 2e9 steps; DiCode about 32.2 h (RL 15.9 h, JAX recompilation 14.4 h, waiting for the model 1.9 h); level generation 23.7 h, asynchronous ([paper](https://arxiv.org/abs/2602.08194) App. A.4, Table 7).
- Other settings: SFL Table 3, PLR and DR Table 4, DiCode Table 5, foundation model Table 6. Parameters: not reported.

## Results

| Method | Final mean return (raw), mean ± s.e. | % of 226 | Seeds | Source |
|---|---|---|---|---|
| DiCode (Qwen3-235B) | 48.55 ± 0.49 | 21.5 | 8 | [paper v2](https://arxiv.org/abs/2602.08194) Table 1, §4.2 |
| Random parent sampling | 46.44 ± 0.41 | 20.5 | 5 | Table 1 |
| Qwen 80B | 45.88 ± 0.40 | 20.3 | 5 | Table 1 |
| Qwen 30B | 44.38 ± 0.66 | 19.6 | 5 | Table 1 |
| Goal only | 42.82 ± 0.39 | 18.9 | 5 | Table 1 |
| Open loop | 41.12 ± 0.77 | 18.2 | 5 | Table 1 |
| PPO-GTrXL (same agent) | 41.59 (no s.e. printed) | 18.4 | 8 | §4.2 |

- Percentages computed in this pass. v1 reported DiCode 48.33 ± 0.63 and PPO-GTrXL 41.54 with 5 seeds ([paper v1](https://arxiv.org/abs/2602.08194v1) Table 7).
- Protocol: 50 checkpoints, each evaluated on a fixed held-out set of 1024 standard Craftax worlds with sampled actions (`pi.sample`) and episodes capped at 8192 steps (`src/dicode/craftax_evaluation.py`; the Craftax default cap is 100,000).
- Achievement success, DiCode against the best baseline ([paper](https://arxiv.org/abs/2602.08194) §4.2, Fig. 3): Make Iron Armour 47% vs 15%; Enter Gnomish Mines 32% vs 9%; Defeat Gnome Warrior 14.7% vs 0%; Defeat Gnome Archer 11.2% vs 0%; Make Diamond Sword 6% vs 3%.
- PLR, DR and SFL baselines (8 seeds): figure only (Figs. 2, 3, 6).
- PPO-GTrXL row added to [[PPO-GTrXL (Hamon 2024)]] as a third-party rerun.

## Problems and limitations

- The fixed game engine means the model cannot invent new mechanics; each new domain needs its own API layer; the learnability score can be gamed by stochastic levels near 0.5 success ([paper](https://arxiv.org/abs/2602.08194) §6).
- Large JAX recompilation overhead ([paper](https://arxiv.org/abs/2602.08194) App. A.4); DiCode runs are not seed-deterministic (repo README).
- Ablations use 5 seeds against 8 for the main run.
- Needs a locally served 235B model on four GPUs.

## Code

- Repository: https://github.com/konstantinosmitsides/dreaming-in-code (`experiments/training/run_dicode.py`; baselines `experiments/training/ppo_gtrxl.py`, `python -m baselines.plr`, `python -m baselines.sfl`; evaluation `experiments/evaluation/run_paper_evaluation.py`); commit read in this pass: `de6f4a0` (2026-09-12). Issue #1 (closed): baselines were missing at first and have since been added.
- Framework: JAX, Hydra; repo config `env_name: Craftax-Symbolic-v1`.
- Licence: MIT; `baselines/` Apache-2.0.
