---
title: "Rule Learner Planner (dargilboa 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - dargilboa (GitHub handle; real name not given)
year: 2026
venue: "GitHub repository (no paper)"
paper: "https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md"
code: "https://github.com/dargilboa/explanatorium"
envs:
  - Craftax-Classic-Symbolic-v1
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax-classic
  - world-model
  - reward-free
  - exploration
  - symbolic
created: 2026-10-20
---
# Rule Learner Planner (dargilboa 2026)

## TL;DR

- "Conjecture and refutation in Craftax": a reward-free agent that learns an explicit theory of the game as probabilistic rules (action + conjunction of conditions), refutes them with a code-length test, and plans its own experiments by backward chaining through the rules; runs on CPU as one continuous life ([README](https://github.com/dargilboa/explanatorium)).
- Craftax-Classic (Symbolic), never sees the reward: Crafter score of its own episodes 8.9% at 300k steps and 11.2% at 1M (seeds 1, 42, 100), against 5.0-5.9% at 1M for PPO, PPO-RNN and PPO-RND trained on the reward under the same protocol ([craftax_wm/RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md) B8, B10).
- Cumulative score over a single continuous life at small budgets: not comparable with any block of [[Existing Implementations]].

## Method

- Rules per observable change: an action and a conjunction of conditions (observed features, invented counters and block classes) with a probability; conditions are kept only if they save more bits than they cost (sequential covering, MDL) ([README](https://github.com/dargilboa/explanatorium#the-method)).
- Refutation: after every step rules pay −log₂ of the probability they gave the outcome; a one-sided three-sigma test against their promised entropy triggers a refit on all experience ([README](https://github.com/dargilboa/explanatorium#the-method)).
- Experiments: the planner seeks untested situations or known failures and chains backward through the rules to reach them ([README](https://github.com/dargilboa/explanatorium#the-method)).
- No game knowledge: a test checks that no game entity is named in the learner or planner; a shuffle-and-rename ablation scores about the same ([README](https://github.com/dargilboa/explanatorium#the-method)).

## Training setup and cost

- One continuous life per seed (episodes restart on death, learning continues); budgets 10k-1M environment steps, all chosen by the learner ([craftax_wm/RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md)).
- Settings not swept; the same fixed values in all three testbeds ([craftax_wm/RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md) caveats).
- Hardware: CPU for the learner (quick start "CPU, about a minute"); wall-clock per run and parameters: not reported.

## Results

Crafter score (%) of the learner's own episodes, cumulative from step 0, mean of seeds 1, 42, 100 (range) ([craftax_wm/figures/headline_table.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/figures/headline_table.md); [RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md) B8, B10).

| Learner (Craftax-Classic, Symbolic) | 100k | 300k | 1M |
|---|---|---|---|
| Rule learner + planner, no reward | 6.2 (6.2-6.3) | 8.9 (8.5-9.2) | 11.2 (11.0-11.4) |
| PPO + reward (Craftax_Baselines, 64 envs) | 2.8 | 3.6 (3.5-3.7) | 5.0 (4.6-5.5) |
| PPO-RNN + reward | 2.6 | 3.4 (3.3-3.5) | 5.9 (5.5-6.4) |
| PPO-RND + reward | 2.9 | 3.8 (3.6-3.8) | 5.3 (4.9-5.6) |
| DreamerV3 (medium, symbolic) + reward | 2.3 | 2.6 (2.6-2.7) | n.r. |
| RND / ICM / E3B, reward-free | 1.6 / 1.3 / 1.5 | 1.4 / 0.9 / 1.4 | 1.3 / 0.5 / 1.3 |

- At 1M the planner's episodes reach stone 82%, stone pickaxe 56%, iron 13.5%; no iron tools, diamonds or eaten plants ([RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md) B10).
- Also reports one-step prediction accuracy on Craftax-Classic and full Craftax (0.636 at 300k on the full game) and Crafter-OO results; these are world-model metrics, not agent scores.
- Baseline rows are third-party reruns of [[PPO (Matthews 2024)]], [[PPO-RNN (Matthews 2024)]], [[RND (Matthews 2024)]], [[ICM (Matthews 2024)]], [[E3B (Matthews 2024)]] and [[DreamerV3 (Hafner 2023)]] under this protocol.

## Problems and limitations

- The authors' baselines score below the Crafter paper's own runs at the same budgets; DreamerV3 runs at `medium` size on the symbolic vector ([RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md) caveats).
- Full Craftax: the planner helps only late, and one of three seeds does not take off ([RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md)).
- Hand-picked counter modulators in the headline configuration; without them Classic score drops to 7.7% at 300k ([RESULTS.md](https://github.com/dargilboa/explanatorium/blob/main/craftax_wm/RESULTS.md)).
- Repository-only, no paper or review; created 2026-10-10.

## Code

- Repository: https://github.com/dargilboa/explanatorium (`craftax_wm/run_generic.sh`, `python -m craftax_wm.rules_wm`); commit read in this pass: `0e837e0` (2026-10-10); no issues.
- Framework: Python (NumPy), JAX Craftax for the environment and baselines.
- Licence: MIT.
