---
title: "Final results"
type: concept
domain: analysis
parent: "[[2026-10-20-craftax-project]]"
tags: [concept, results, analysis, comparison, craftax-1m]
created: 2026-10-28
updated: 2026-10-28
related-code:
  - "[[src/wut_murl_craftax/analysis/compare.py]]"
  - "[[src/wut_murl_craftax/analysis/figures.py]]"
  - "[[src/wut_murl_craftax/analysis/figures_game.py]]"
  - "[[src/wut_murl_craftax/analysis/latex.py]]"
  - "[[src/wut_murl_craftax/analysis/literature.py]]"
  - "[[src/wut_murl_craftax/analysis/results.py]]"
---
# Final results

## Scope

- Craftax-1M only: 1B pending (`final-1b` not run, R75); the 1B sections below are placeholders for the same tables and figures at 1B.
- Units: scores are the final training score, mean training-episode return in % of the 226-point maximum (last 2% of the budget, at least 100 episodes); eval scores are the fixed-seed final checkpoint, stochastic and greedy, same unit. 10 seeds per method, tuned config.
- Phases: `final-1m` (tuned runs and the six ablations) and `combo-1m` (the combination of [[Count-Based Progress Bonus design]] and [[Self-Imitation Learning design]] on PPO-RNN, [[Experiment decisions]], 2026-10-24).
- Numbers come from `outputs/results/tables/1m/*.csv` and `report/numbers.tex`, written by the comparison run of 2026-10-28 (`wut_murl_craftax.analysis.results`, plan [[2026-10-20-craftax-comparison-and-results]]); each table section names its file. Figures are in `outputs/results/figures/1m/` (one panel per PNG, no text on the images); the report prints their captions from [[Figures 1M]] and [[Tables 1M]], which list every figure and table file with its description.
- Two blocks: main (`count-bonus-ppo-rnn`, `count-bonus-pqn-rnn`, `sil`, `combo`) and "uses hand-written skills" (`smdp-skills`, `dyna-q`, [[Craftax skill library]]); the three controls are the reference rows of both and a method is never ranked against the other block. PoI = probability of improvement; Holm p = Mann-Whitney p, Holm-adjusted within the block.
- Reporting rulings: runs belong to a phase by the `phases` tag, ablations are reported apart (R77); "not logged" for metric families the 26 runs reused from tuning lack (R78); hardware and timing are context only (R97); tuning optimism uses fresh seeds only (R102); the tuning-runs obstacles table is deferred (R107). All in [[Experiment decisions]], "Phase 4 analysis rulings at 1M".

## Headline findings (1M)

- Count-bonus PPO-RNN has the highest mean (2.274 +- 0.037, IQM 2.278 [2.178, 2.369]) but is not separable from the controls: PoI 0.74 [0.49, 0.93] over PPO (Holm p 0.530) and 0.68 [0.42, 0.89] over PPO-RNN (Holm p 1.000), against means 2.173 and 2.188. Source: `main.csv`.
- SIL (2.137) and the combination (2.132) do not beat PPO or PPO-RNN (PoI 0.33 to 0.37, Holm p 1.000), and the combination is below count-bonus PPO-RNN alone (PoI of the combination over it 0.17 [0.03, 0.38]): at 1M the two components do not add up. Sources: `main.csv`, `poi_matrix.csv`.
- The PQN-RNN family and both skill methods are well below the PPO family and unstable: means 1.481 (count-bonus PQN-RNN), 1.356 (PQN-RNN), 1.077 (Dyna-Q), 0.831 (SMDP skills) against 2.132 to 2.274 for the PPO family, SIL and the combination; Holm p 0.001 to 0.002 against both PPO controls for the three methods that carry one; seed sd 0.375 to 0.651 against 0.046 to 0.117; runs with a collapse 3, 4, 5, 8 of 10 against 0 or 1. Neither skill method is separable from PQN-RNN (Holm p 0.273 and 0.108). Sources: `main.csv`, `stability.csv`.
- Reproduction check: all three controls pass (PPO gap -0.027 within h 0.128; PPO-RNN gap -0.112 within h 0.135; PQN-RNN passes its reference-code comparison); no issue note opened. Source: `reproduction_gate.csv`.
- Every method reaches only floor 0 in the final eval episodes (deepest floor share 1.0), 47 to 54 of the 67 achievements are never unlocked by any seed at any checkpoint (training and eval), and zombies are the top cause of death everywhere (46.6% to 61.1%). Sources: `game_summary.csv`, the nine training-analysis notes below.

## T1: main results

Mean +- s.e., IQM [95% CI] (stratified bootstrap) and the eval scores per method; `n` = 10 everywhere. Source: `main.csv`.

Main block:

| Method | Analysis | Mean +- s.e. | IQM [95% CI] | Eval stochastic | Eval greedy |
|---|---|---|---|---|---|
| ppo (control) | [[PPO training analysis]] | 2.173 +- 0.029 | 2.173 [2.105, 2.237] | 2.164 | 0.909 |
| ppo-rnn (control) | [[PPO-RNN training analysis]] | 2.188 +- 0.034 | 2.187 [2.115, 2.262] | 2.160 | 1.556 |
| pqn-rnn (control) | [[PQN-RNN training analysis]] | 1.356 +- 0.161 | 1.465 [1.032, 1.721] | 1.268 | 1.265 |
| count-bonus-ppo-rnn | [[Count-bonus PPO-RNN training analysis]] | 2.274 +- 0.037 | 2.278 [2.178, 2.369] | 2.244 | 1.533 |
| count-bonus-pqn-rnn | [[Count-bonus PQN-RNN training analysis]] | 1.481 +- 0.119 | 1.558 [1.261, 1.734] | 1.469 | 1.471 |
| sil | [[SIL training analysis]] | 2.137 +- 0.015 | 2.138 [2.103, 2.171] | 2.131 | 1.377 |
| combo | [[Combo training analysis]] | 2.132 +- 0.029 | 2.121 [2.067, 2.188] | 2.144 | 1.375 |

Uses hand-written skills block (controls as above):

| Method | Analysis | Mean +- s.e. | IQM [95% CI] | Eval stochastic | Eval greedy |
|---|---|---|---|---|---|
| dyna-q | [[Dyna-Q training analysis]] | 1.077 +- 0.196 | 1.200 [0.661, 1.536] | 1.081 | 1.070 |
| smdp-skills | [[SMDP skills training analysis]] | 0.831 +- 0.206 | 0.907 [0.386, 1.353] | 0.819 | 0.800 |

- Design notes: [[PPO design]], [[PPO-RNN design]], [[PQN-RNN design]], [[Count-Based Progress Bonus design]], [[Self-Imitation Learning design]], [[SMDP Q-Learning over Hand-Defined Skills design]], [[Dyna-Q over a Skill-Level Progress Model design]].
- Greedy eval: PPO 0.909 against 1.556 for PPO-RNN, although their stochastic evals are level (2.164 and 2.160).

### PoI and Holm p against the controls

PoI [95% CI] of the method over each control, with the Holm-adjusted p in brackets; blank for a control against itself (a control's own Holm p is not reported). Source: `main.csv`.

| Method | vs PPO | vs PPO-RNN | vs PQN-RNN |
|---|---|---|---|
| ppo | - | 0.46 [0.21, 0.72] | 1.00 [1.00, 1.00] |
| ppo-rnn | 0.54 [0.27, 0.80] | - | 1.00 [1.00, 1.00] |
| pqn-rnn | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| count-bonus-ppo-rnn | 0.74 [0.49, 0.93] (0.530) | 0.68 [0.42, 0.89] (1.000) | 1.00 [1.00, 1.00] (0.002) |
| count-bonus-pqn-rnn | 0.00 [0.00, 0.00] (0.002) | 0.00 [0.00, 0.00] (0.002) | 0.56 [0.30, 0.82] (1.000) |
| sil | 0.35 [0.11, 0.62] (1.000) | 0.33 [0.10, 0.60] (1.000) | 1.00 [1.00, 1.00] (0.002) |
| combo | 0.37 [0.13, 0.63] (1.000) | 0.33 [0.10, 0.59] (1.000) | 1.00 [1.00, 1.00] (0.002) |
| dyna-q (skills) | 0.00 [0.00, 0.00] (0.001) | 0.00 [0.00, 0.00] (0.001) | 0.35 [0.12, 0.61] (0.273) |
| smdp-skills (skills) | 0.00 [0.00, 0.00] (0.001) | 0.00 [0.00, 0.00] (0.001) | 0.24 [0.06, 0.48] (0.108) |

- Count-bonus on PQN-RNN does not improve on PQN-RNN (PoI 0.56 [0.30, 0.82], Holm p 1.000); count-bonus PPO-RNN is the only non-control method whose PoI over both PPO controls exceeds 0.5, without a significant Holm p.
- Ten seeds make the PoI intervals wide (about 0.45 to 0.55 wide across the PPO-family rows).

## Reproduction check

Each control against its published Craftax-1M value ([[Craftax Baselines]], [[Existing Implementations]]), band half-width h of `evaluation.reproduction.band` computed with our 10 seeds. Source: `reproduction_gate.csv`.

| Control | Ours (mean) | Published | Gap (ours - published) | h | Result |
|---|---|---|---|---|---|
| ppo | 2.173 | 2.2 | -0.027 | 0.128 | pass |
| ppo-rnn | 2.188 | 2.3 | -0.112 | 0.135 | pass |
| pqn-rnn | 1.356 | none at 1M | n/a | n/a | pass (reference-code comparison: original 1.509, ours 1.27 in `tune-coarse`, 3 seeds; band 1.005) |

- All three controls pass at 1M, so no issue note was opened in `knowledge/issues/`; the comparison continues with our controls as the base.
- PPO-RNN passes the gate while its 95% CI [2.125, 2.249] lies below the published 2.3 (`literature.csv`, `ours_below`); the gate band h (0.135) is wider than the CI half-width of the mean (0.062).
- The published values are given to one decimal without a spread.
- The 1B gate is pending.

## T2: literature

Our 1M training scores beside block D of [[Existing Implementations]]; training-episode return, our methods in T1. "Above", "within" and "below" mean our 95% CI lies above, around or below the published value; only same-protocol rows are classed. Source: `literature.csv`.

| Published entry | Protocol | Seeds | Score | Spread | Our methods above / within / below |
|---|---|---|---|---|---|
| PPO-RNN (Matthews 2024) | training-episode return, same | 10 | 2.3 | none | none / count-bonus-ppo-rnn / ppo, ppo-rnn, pqn-rnn, count-bonus-pqn-rnn, sil, combo, dyna-q, smdp-skills |
| RND, PPO, ICM, E3B (Matthews 2024, one row each) | training-episode return, same | 10 | 2.2 each | none | count-bonus-ppo-rnn / ppo, ppo-rnn / pqn-rnn, count-bonus-pqn-rnn, sil, combo, dyna-q, smdp-skills |
| Simulus (Cohen 2025) | mean of 100 test episodes at the end of training, different protocol | 5 | 6.59 | sd 0.49 | not classed |
| EASimulus (Peng 2026) | not reported | 5 (repo) | 7.23 | sd 0.19 | not classed |

- Our best (count-bonus PPO-RNN, 2.274) sits at the level of the published PPO-family methods (2.2 to 2.3); the two model-based entries (6.59, 7.23) use another or unreported protocol and are context only.
- Blocks A and A2 (1B) are pending.

## Ablations (R77)

Each ablation against its method's tuned runs; difference = ablation mean minus tuned mean with a bootstrap 95% CI; PoI = probability that an ablation run beats a tuned run; Holm over the block's ablations. Tuned means from `main.csv`. Source: `ablation.csv`.

| Method | Ablation | Mean +- s.e. | Tuned mean | Difference [95% CI] | PoI | Holm p |
|---|---|---|---|---|---|---|
| count-bonus-ppo-rnn | simhash | 1.783 +- 0.039 | 2.274 | -0.491 [-0.591, -0.386] | 0.00 | 0.0004 |
| sil | single-buffer | 2.138 +- 0.030 | 2.137 | +0.001 [-0.061, 0.062] | 0.48 | 0.910 |
| dyna-q | kappa0 | 0.692 +- 0.187 | 1.077 | -0.385 [-0.888, 0.132] | 0.30 | 0.421 |
| smdp-skills | monte-carlo | 0.478 +- 0.153 | 0.831 | -0.353 [-0.823, 0.118] | 0.32 | 0.421 |
| smdp-skills | no-flags | 0.446 +- 0.168 | 0.831 | -0.385 [-0.878, 0.113] | 0.34 | 0.421 |
| smdp-skills | random-top | 0.274 +- 0.126 | 0.831 | -0.557 [-0.991, -0.112] | 0.26 | 0.303 |

- The SimHash key (same counts over a 64-bit hash of the full observation, [[Count-Based Progress Bonus design]]) loses 0.491 points: the matched observation-novelty control is clearly worse than the tuned progress-based counts.
- The SIL single-buffer ablation equals the tuned config (difference +0.001).
- All four skill ablations have a negative difference; only `random-top` has a CI that excludes 0, and its Holm p is 0.303.

## Tuning optimism (R102)

Tuning mean over the tuned config's runs (`tune-fine`, else `tune-coarse`, 3 seeds unless noted) against the mean over fresh final seeds not reused from tuning; optimism = tuning minus fresh. Source: `tuning_optimism.csv`.

| Method | Tuning mean | Fresh seeds | Fresh mean | Optimism |
|---|---|---|---|---|
| pqn-rnn (tune-coarse) | 1.270 | 7 | 1.393 | -0.122 |
| count-bonus-ppo-rnn | 2.333 | 7 | 2.249 | 0.084 |
| count-bonus-pqn-rnn | 1.721 | 7 | 1.378 | 0.344 |
| sil | 2.140 | 7 | 2.136 | 0.004 |
| dyna-q | 1.364 | 7 | 0.954 | 0.410 |
| smdp-skills | 1.415 | 7 | 0.581 | 0.835 |

- PPO and PPO-RNN read n/a (selected on all their final seeds); the combination reads n/a (never tuned).
- The tuning mean of the skill methods overstates the fresh result by 0.410 (Dyna-Q) and 0.835 (SMDP skills); SIL and the count bonus on PPO-RNN lose 0.004 and 0.084 (point estimates, 3 tuning vs 7 fresh seeds, no interval).

## T3: stability

Across the 10 final seeds; drawdown floor at 1M is max(0.1, 0.25 x the run's best smoothed score) (R79, R92). Source: `stability.csv`.

| Method | Seed sd | CV | Median max drawdown | Runs with drops | Runs with collapse |
|---|---|---|---|---|---|
| ppo | 0.091 | 0.042 | 0.229 | 0 | 0 |
| ppo-rnn | 0.107 | 0.049 | 0.280 | 1 | 0 |
| pqn-rnn | 0.508 | 0.374 | 1.300 | 10 | 4 |
| count-bonus-ppo-rnn | 0.117 | 0.051 | 0.241 | 1 | 0 |
| count-bonus-pqn-rnn | 0.375 | 0.254 | 1.304 | 10 | 3 |
| sil | 0.046 | 0.022 | 0.256 | 2 | 1 |
| combo | 0.093 | 0.043 | 0.241 | 0 | 0 |
| dyna-q | 0.619 | 0.575 | 1.628 | 9 | 5 |
| smdp-skills | 0.651 | 0.784 | 1.809 | 10 | 8 |

## T6: in-game summary

Final-checkpoint stochastic eval episodes, each seed weighted equally; race wins = achievements a method wins in its block by median first-unlock env steps (the wall-clock variant is context only, R97). "n/l" = not logged (R78: the controls PPO and PPO-RNN are 10 of 10 reused from tuning). Source: `game_summary.csv`.

| Method | Race wins | Deepest floor (share) | Top death cause (share) | Coverage | Wasted share | Adoption |
|---|---|---|---|---|---|---|
| ppo | n/l | 0 (1.0) | zombie (0.611) | 0.112 | 0.665 | 0.0006 |
| ppo-rnn | n/l | 0 (1.0) | zombie (0.566) | 0.126 | 0.658 | 0.0004 |
| pqn-rnn | 3 (main), 9 (skills) | 0 (1.0) | zombie (0.466) | 0.108 | 0.831 | 0.0002 |
| count-bonus-ppo-rnn | 6 | 0 (1.0) | zombie (0.589) | 0.130 | 0.616 | 0.0002 |
| count-bonus-pqn-rnn | 5 | 0 (1.0) | zombie (0.502) | 0.093 | 0.756 | 0.0011 |
| sil | 6 | 0 (1.0) | zombie (0.552) | 0.115 | 0.667 | 0.0008 |
| combo | 3 | 0 (1.0) | zombie (0.575) | 0.120 | 0.657 | 0.0009 |
| dyna-q | 4 | 0 (1.0) | zombie (0.525) | 0.110 | 0.786 | 0.0003 |
| smdp-skills | 4 | 0 (1.0) | zombie (0.495) | 0.089 | 0.857 | 0.0009 |

- Coverage is the mean over visited floors; wasted share and coverage are medians of seed medians; adoption is the share of episodes using an advanced mechanic, mean over mechanics (definitions: `game_summary_table` in [[src/wut_murl_craftax/analysis/compare.py]]).
- Per-method detail (achievements, deaths, stalls, obstacles) is in the training-analysis notes of T1; every one opens with a 5-bullet summary.

## T7: sensitivity

Per method and tuning phase: configs and runs tried, best config (mean), robust share (configs with r_c of at least 0.9), noise floor, and the two most sensitive hyperparameters with `S_h`; "insensitive" = `S_h` below the noise floor. Combo was never tuned. Source: `sensitivity.csv`.

- ppo: coarse 7 configs, 28 runs, best 845b73b4 (2.261), robust 0.857, floor 0.031: envs 0.310, num_steps 0.310 (both sensitive); fine 4 configs, 12 runs, best c043716c (2.224), robust 1.000, floor 0.025: gae_lambda 0.052, lr 0.028 (both sensitive).
- ppo-rnn: coarse 8, 31, best 04e6b687 (2.261), robust 0.750, floor 0.044: envs 0.361, lr 0.139 (both sensitive); fine 4, 12, best efda7c6f (2.307), robust 0.750, floor 0.027: ent 0.100, lr 0.040 (both sensitive).
- pqn-rnn: coarse 10, 30, best 606530ad (1.718), robust 0.200, floor 0.248: envs 1.175, num_steps 1.175 (both sensitive); fine 4, 12, best dcb8a7ef (1.661), robust 0.750, floor 0.204: lam 0.159, lr 0.115 (both insensitive).
- count-bonus-ppo-rnn: coarse 24, 72, best b0b1c535 (2.333), robust 0.958, floor 0.040: beta 0.034, granularity 0.011 (both insensitive); fine 6, 18, same best, robust 1.000, floor 0.028: granularity 0.028, beta 0.022 (both insensitive).
- count-bonus-pqn-rnn: coarse 24, 72, best 3c5f5200 (1.590), robust 0.375, floor 0.220: beta 0.246 (sensitive), decay 0.069 (insensitive); fine 6, 18, best e6e1fafa (1.721), robust 0.333, floor 0.275: decay 0.144, beta 0.037 (both insensitive).
- sil: coarse 18, 54, best 6f8690ad (2.140), robust 0.889, floor 0.040: sil_steps 0.074 (sensitive), lambda_sil 0.030 (insensitive); fine 6, 18, same best, robust 1.000, floor 0.026: lambda_sil 0.046, sil_steps 0.033 (both sensitive).
- dyna-q: coarse 21, 63, best 670e2f80 (0.949), robust 0.143, floor 0.598: plan_kappa 0.317, plan_n 0.244 (both insensitive); fine 9, 27, best 74c22d85 (1.364), robust 0.111, floor 0.323: model_slots 0.353 (sensitive), plan_kappa 0.234 (insensitive).
- smdp-skills: coarse 48, 144, best 0b84c247 (1.415), robust 0.021, floor 0.333: alpha 0.167, timeout_mult 0.122 (both insensitive); fine 9, 27, same best, robust 0.111, floor 0.202: timeout_mult 0.405, alpha 0.256 (both sensitive).

## Hardware and timing (context only, R97)

Every laptop run trained with two concurrent queue workers, so these figures are never ranked; PPO and PPO-RNN are fully reused from tuning, the others partly, the combination not at all (`reused_runs` in the file). Source: `hardware.csv`.

| Method | Steady SPS | Wall-clock (s) |
|---|---|---|
| ppo | 4344 | 230 |
| ppo-rnn | 3071 | 341 |
| pqn-rnn | 2510 | 418 |
| count-bonus-ppo-rnn | 1962 | 529 |
| count-bonus-pqn-rnn | 2183 | 470 |
| sil | 1591 | 651 |
| combo | 1546 | 668 |
| dyna-q | 2733 | 377 |
| smdp-skills | 3034 | 341 |

## Figure index

Files in `outputs/results/figures/1m/` (`.png` only; a `_video` variant beside each, `_double` for the order graphs and `legend_methods`). One panel per file; the description of each file is in [[Figures 1M]], the tables in [[Tables 1M]]. "Descriptive" = no table-backed headline; read the figure.

| ID | Files | Takeaway |
|---|---|---|
| F1 | `curves_score_main`, `curves_score_skills` | Learning curves per method (mean, 95% band; controls dashed); drawdowns in T3 |
| F2 | `aggregate` | IQM and mean with CIs: PPO, PPO-RNN, count-bonus PPO-RNN, SIL and the combination between 2.07 and 2.37; PQN-RNN family and skills below 1.74 |
| F3 | `poi_matrix_main`, `poi_matrix_skills` | Per-block PoI matrix: only the PQN-RNN family and the skill methods separate from PPO and PPO-RNN (Holm p 0.002 or lower) |
| F4 | `seed_spread_main`, `seed_spread_skills`, `seed_spread_published` | Seed sd from 0.046 (SIL) to 0.651 (SMDP skills); published marks on the same axis |
| F5 | `success_heatmap_main`, `success_heatmap_skills` | 47 to 54 of 67 achievements never unlocked by any seed, per method |
| F6 | `race_steps_<block>`, `race_wall_<block>` | Race wins in steps: count-bonus PPO-RNN and SIL 6 each in the main block; wall variant context only (R97) |
| F7 | `order_<method>`, `order_agreement` | Descriptive: achievement order graphs and Kendall tau agreement |
| F8 | `funnel_tech`, `funnel_floor`, `funnel_training_depth`, `funnel_training_floor` | Every method's deepest floor is 0 (share 1.0) |
| F9 | `deaths`, `stalls_ccdf`, `stalls_escape` | Zombie is the top death cause for every method (0.466 to 0.611); stalls per eval episode 0.084 (combo) to 0.164 (PQN-RNN) |
| F10 | `splits` | Descriptive: steps to each milestone |
| F11 | `coverage`, `new_tiles` | Coverage 0.089 (SMDP skills) to 0.130 (count-bonus PPO-RNN) |
| F12 | `survival_low`, `survival_zero`, `survival_level` | Descriptive: low and zero stats, stat value at eat, drink and sleep |
| F13 | `combat_kills`, `combat_damage`, `combat_fights` | Descriptive: kills per creature, damage dealt and taken |
| F14 | `actions_share`, `actions_entropy`, `actions_wasted` | Wasted share 0.616 (count-bonus PPO-RNN) to 0.857 (SMDP skills) |
| F15 | `resources_use`, `resources_time` | Descriptive: gathered vs used, steps per achievement |
| F16 | `advanced_mean`, `advanced_final` | Advanced-mechanics adoption at most 0.0011 for every method |
| F17 | `radar_<method>` | Descriptive: playstyle radars, shared axes |
| F18 | `config_scores_<phase>`, `sensitivity_<phase>` | Robust share from 0.021 (SMDP skills, coarse) to 1.000 (PPO, count-bonus PPO-RNN and SIL at `tune-fine`); see T7 |
| F19 | `stability_drawdown`, `stability_events` | Collapses concentrate in PQN-RNN family and skills (T3) |
| F20 | `hardware_<metric>` | Context only (R97); never ranked |
| LEG | `legend_methods` | Colour, line style and marker of every method; no other figure carries a method legend |

## Interpretation (decision trees)

Full write-up: [[Decision-tree interpretation]] (90 policies, 9 methods x 10 seeds, from `outputs/interpretation/1m/complexity.csv` and `policies.csv`).

- No method's median policy is distillable at 90%: the L80, L90 and L95 medians are "> 4096" leaves for every method (R106); the one policy that reaches 90% is SMDP skills seed 8 (2 leaves), a collapsed one-action policy; median test fidelity at 256 leaves is 0.385 (Dyna-Q) to 0.512 (SIL), and the median tree of every method is 0.108 to 0.166 above the majority-action share; fidelity falls for larger trees.
- Score vs fidelity at 256 leaves: Spearman 0.308 (p 0.003, base), between methods; it is accompanied by the majority share, not by extra structure (score vs gain over majority -0.027, derived).
- Every recurrent policy's tree leans most on the previous action, a memory proxy (importance 0.117 to 0.186); without it fidelity at 256 leaves drops by 0.036 to 0.071.
- A 256-leaf tree keeps 0.183 (count-bonus PPO-RNN) to 0.490 (SMDP skills) of its network's greedy score in closed loop; one DAgger round raises this for all 9 methods.
- SMDP skills and Dyna-Q: the lowest-scoring seeds have skill networks whose greedy majority action cannot succeed given the policy's inventory (`READ_BOOK` with no book, `MAKE_DIAMOND_PICKAXE` and `MAKE_IRON_ARMOUR` with no diamond or iron ever collected); their top levels are lookup tables over 47 to 355 visited keys, and the median-score seed's 10 highest values lie within 0.08 of the optimistic start 10.
- Figure: `fidelity_vs_leaves` (PNG, `figures/1m/`); F27 `distil_score` and F28 `distil_features` come from the figure-angles branch.

## 1B (pending)

- Pending: T1 and the PoI table at 1B (mean, median, 95% CI; indicative, 3 seeds), from `final-1b` and `combo-1b`.
- Pending: the 1B reproduction gate (the 3-seed band is wider, so indicative) and blocks A and A2 of T2.
- Pending: the re-check of the combination pair with the 1B means (2026-10-24 decision, [[Experiment decisions]]).
- Pending: the 1B panels of F1 and F2, the 1B radar and funnel, and the 1B rows of T3 and T6.
- Deferred (R107): the obstacles table of the tuning runs (`tune-coarse`, `tune-fine`, 494 runs).
