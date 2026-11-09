---
title: "Start here"
type: handbook
domain: vault
parent: "[[wut_murl_craftax]]"
tags: [handbook, onboarding]
created: 2026-11-09
updated: 2026-11-09
---
# Start here

A reading path for someone new: scope, what was done, the results and how to read them, then the methods and the code. Every step is one note; read in order, stop where you have enough.

## The project in five lines

- Craftax is a survival game in JAX with 67 achievements in four tiers (25 at 1 point, 18 at 3, 15 at 5, 9 at 8, 226 in total). The score is the share of those points an agent collects.
- Three standard agents (PPO, PPO-RNN, PQN-RNN) and four additions (count-based progress bonus, self-imitation, SMDP Q-learning over hand-written skills, Dyna-Q over the skills), plus a combination of the two best compatible ones, are compared under one protocol.
- Budgets: 1M steps (10 seeds, tuned, laptop) and 1B steps (3 seeds, cluster, indicative only).
- Finding: at 1M nothing significantly beats the controls; at 1B PQN-RNN is level with PPO-RNN; the skill methods lose at both.
- Deliverables: 3-page report ([[2026-10-20-craftax-report]]), videos ([[2026-10-20-craftax-visualisation-production]]), Hugging Face release ([[2026-10-20-craftax-huggingface-release]]), checkpoint and talk notes.

## Reading order

| Step | Read | You learn |
|---|---|---|
| 1 | [[Course project requirements]] | what the course asks and when |
| 2 | [[2026-10-20-craftax-overview]] | the whole plan on one page: phases, runs, what we can say at the end |
| 3 | [[Craftax]] then [[Game]] | the environment: observations, actions, floors, achievements, scoring |
| 4 | [[Craftax scoring and evaluation]] | exactly what the score is |
| 5 | [[Methods]] | the seven methods side by side and how they relate (table below) |
| 6 | [[Experiment decisions]] | why the protocol is what it is: seeds, tuning, metrics, rulings |
| 7 | [[Final results]] | the 1M tables and headline findings; its last section is 1M vs 1B |
| 8 | [[Tables 1B]], [[Figures 1B]], [[Figures SCALE]] | the 1B results and the budget comparison |
| 9 | [[Decision-tree interpretation]] | what the trained policies do, read from distilled trees |
| 10 | [[I1 - PPO-RNN 1B above the reproduction band]], [[Fixed issues]] | known caveats and fixed bugs |
| 11 | [[Checkpoint meeting]], [[Presentation and discussion]] | how the work is presented and the questions to expect |
| 12 | [[EDEN compute]], then the package notes under [[wut_murl_craftax]] | how it ran and where the code is |

## The methods and where each comes from

| Method | What it does in one line | Our design note | Original idea |
|---|---|---|---|
| PPO | policy-gradient control | [[PPO design]] | Schulman et al. 2017, [arXiv 1707.06347](https://arxiv.org/abs/1707.06347); Craftax baseline: [[PPO (Matthews 2024)]] |
| PPO-RNN | PPO with a GRU memory | [[PPO-RNN design]] | GRU: Cho et al. 2014, [arXiv 1406.1078](https://arxiv.org/abs/1406.1078); [[PPO-RNN (Matthews 2024)]] |
| PQN-RNN | online Q(λ) with memory, no replay buffer or target network | [[PQN-RNN design]] | Gallici et al. 2025, [arXiv 2407.04811](https://arxiv.org/abs/2407.04811); [[PQN-RNN (Gallici 2025)]] |
| Count-based progress bonus | extra reward 1/√N for rare game-progress states | [[Count-Based Progress Bonus design]] | Bellemare et al. 2016, [arXiv 1606.01868](https://arxiv.org/abs/1606.01868); SimHash: Tang et al. 2017, [arXiv 1611.04717](https://arxiv.org/abs/1611.04717); [[Count-Based Progress Bonus]] |
| Self-imitation (SIL) | imitate own steps whose return beat the critic | [[Self-Imitation Learning design]] | Oh et al. 2018, [arXiv 1806.05635](https://arxiv.org/abs/1806.05635); [[Self-Imitation Learning]] |
| SMDP skills | Q-learning over 25 hand-written options | [[SMDP Q-Learning over Hand-Defined Skills design]], [[Craftax skill library]] | Sutton, Precup, Singh 1999 (options); [[SMDP Q-Learning over Hand-Defined Skills]] |
| Dyna-Q over skills | adds a learned skill-level model and planning | [[Dyna-Q over a Skill-Level Progress Model design]] | Sutton 1990 (Dyna); Sutton and Barto, chapter 8; [[Dyna-Q over a Skill-Level Progress Model]] |

- The 25 skills were written by us from the game's rules and achievement list (start condition, success test, step limit); they were not extracted from data or from any agent's play. That is why the two skill methods are reported in their own block.
- Intrinsic-reward baselines of the Craftax paper (RND, ICM, E3B) are not run here: [[RND (Matthews 2024)]], [[ICM (Matthews 2024)]], [[E3B (Matthews 2024)]].
- Further candidates and rejected ideas: [[Candidate Algorithms]]; every published Craftax agent: [[Existing Implementations]]; textbook background: [[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]].

## How to read the results

- **Score:** mean training-episode return over the last 2 % of the budget, as a percentage of 226. 2.2 % means about 5 points. Evaluation scores on 64 fixed seeds are reported next to it.
- **Mean ± s.e. and IQM with a bootstrap interval:** the mean is the headline; the interquartile mean is robust to one odd seed; intervals are over seeds.
- **Probability of improvement (PoI):** the chance that a random run of method A beats a random run of B; 0.5 is a tie. Reported with Holm-corrected Mann–Whitney p-values, only at 1M.
- **1B has 3 seeds:** read the intervals, do not read the order of close methods. The 1M-to-1B comparison is in [[Figures SCALE]] and [[Tables SCALE]].
- **Reproduction gate:** each control must land inside a band around its published score before a method is compared with it; PPO-RNN at 1B is above it, see [[I1 - PPO-RNN 1B above the reproduction band]].
- **Stability and obstacles:** the per-method `training analysis` notes (under results) count collapses, instability and forgotten skills per seed; they explain why a mean is low.
- **Decision trees:** a small tree is fitted to a trained policy's actions; its size and accuracy say how simple the learned behaviour is ([[Decision-tree interpretation]]).

## Where the artefacts are

| What | Where |
|---|---|
| Report | `report/report.pdf`, source `report/report.tex`, numbers `report/numbers.tex` (generated) |
| Tables and figures | [[Tables 1M]], [[Figures 1M]], [[Tables 1B]], [[Figures 1B]], [[Tables SCALE]], [[Figures SCALE]] |
| Per-method analysis | [[PPO training analysis]], [[PPO-RNN training analysis]], [[PQN-RNN training analysis]], [[Count-bonus PPO-RNN training analysis]], [[Count-bonus PQN-RNN training analysis]], [[SIL training analysis]], [[Combo training analysis]], [[SMDP skills training analysis]], [[Dyna-Q training analysis]] |
| Videos and GIFs | [[Visualisation]], [[Video cases 1M]], [[Visualisation decisions]] |
| Code | package notes under `architecture/`, listed in [[wut_murl_craftax]] |
| Plans | [[2026-10-20-craftax-project]] with one file per area |
