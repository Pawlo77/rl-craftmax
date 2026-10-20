---
title: "Experiment decisions"
type: decision-log
domain: training
parent: "[[wut_murl_craftax]]"
tags: [decision-log, experiments, training]
created: 2026-10-20
updated: 2026-10-20
---
# Experiment decisions

Execution plan: [[2026-10-20-craftax-project]]. Video and figure decisions: [[Visualisation decisions]].

## 2026-10-20 - Benchmarks, budgets and seeds

Decision: experiments target Craftax-Symbolic at 1M and 1B environment steps, the two challenges of the Craftax scoreboard; Craftax-Classic is used for debugging and the video warm-up only.

Why: both budgets are directly comparable with [[Existing Implementations]] (blocks for 1M and 1B) and [[Craftax Baselines results]].

Consequences: tuning runs at 1M; finals at 1M and 1B. Rejected: 1M only; an extra Craftax-Classic 1B comparison.

Decision: 3 seeds per configuration while tuning, 10 seeds per final configuration.

Why: the scoreboard reports 10 seeds; 3 seeds are enough to rank configurations.

Consequences: final error bars are comparable with the published baselines ([[Craftax Baselines benchmark protocol]]).

## 2026-10-20 - Hyperparameter search

Decision: each method is tuned in two stages at Craftax-1M (3 seeds): a coarse grid over the method's own knobs, then a finer grid around the best coarse configuration; shared PPO settings start from the tuned control.

Why: a coarse-then-fine search finds a good configuration with fewer runs than one dense grid.

Consequences: every method plan file lists both grids; [[2026-10-20-craftax-compute-budget]] counts them. Rejected: one small grid; a medium grid with shared knobs.

Decision: the 1B finals reuse the method knobs of the best 1M configuration and the paper's tuned 1B settings for the shared PPO part; no tuning at 1B.

Why: the paper tuned PPO at 1B already ([[Craftax Baselines algorithms and hyperparameters]]); tuning at 1B would multiply the cost of the most expensive runs.

Consequences: a method can look worse at 1B than its best possible setting; the report says so. Rejected: a 100M-step check of the top 2 configurations; a separate 1B grid.

Decision: after the individual finals, the two best compatible methods are combined and run at 1M and 1B.

Why: the candidates attack different difficulties, so their gains may add up.

Consequences: one extra plan stage and one extra row in the comparison. Rejected: no combination; combining every method that helps.

## 2026-10-20 - Metrics, evaluation and interpretation

Decision: the headline metric is the scoreboard one (training-episode reward as % of the maximum 226); every run also reports fixed-seed evaluation (64 seeds, stochastic and greedy policy), learning stability, hardware usage and twelve in-game metrics.

Why: the user wants results analysed beyond the goal: stability, hardware, and in-game behaviour.

Consequences: the in-game metrics are first-unlock race, achievement order graph, progress funnel, deaths and stalls, speedrun splits, exploration coverage, survival management, combat profile, action usage, resource efficiency, advanced mechanics, playstyle radar; definitions in [[2026-10-20-craftax-evaluation-and-metrics]]; they read achievements from `env_state.achievements`, because `info` reports them only at episode end ([[Craftax scoring and evaluation]]).

Decision: results are compared with interquartile means and stratified bootstrap 95% confidence intervals, plus the probability of improvement over each control.

Why: 10 seeds are few; these statistics are robust to outlier seeds and are the RL standard for small seed counts.

Consequences: [[2026-10-20-craftax-comparison-and-results]].

Decision: the complexity of each final policy's decision rules is measured by decision-tree distillation: trees of growing size fitted to the policy's actions on fixed-seed rollouts, reported as fidelity against number of leaves and the leaves needed for 90% fidelity.

Why: a simple, comparable measure of how complex the learned behaviour is, and readable trees for the interpretation.

Consequences: [[2026-10-20-craftax-interpretation-and-ruleset]]. Rejected: rule lists; counting behaviour clusters.

## 2026-10-20 - Running, stopping and resuming

Decision: every (method, configuration, budget, seed) is one MLflow run with a deterministic name; a queue runner skips finished runs and resumes interrupted ones from their last checkpoint.

Why: the user wants an incremental plan that can be started and stopped at any time.

Consequences: [[2026-10-20-craftax-infrastructure]]; the SQLite store `mlruns.db` is the record of truth and is published at the end.

Decision: the same code and run names work on the laptop and on clusters; machine profiles change only runtime settings (CPU shards, device roles), never hyperparameters.

Why: the user wants to run any phase on either machine.

Consequences: results from different machines merge into one store by run name ([[Craftax runtime setup]]).

## 2026-10-20 - Report and publication

Decision: a report of at most 4 pages in the Interspeech LaTeX template, then, as the last step, a public Hugging Face Space `Pawlo77/wut-murl-craftax` and a public dataset repo `Pawlo77/wut-murl-craftax-results` with `mlruns.db`, metric exports, final checkpoints and the videos.

Why: the user wants all results published after the report.

Consequences: [[2026-10-20-craftax-report]] and [[2026-10-20-craftax-huggingface-release]]; publishing waits for the user's confirmation. Rejected: private repos; dataset repo only.

## 2026-10-20 - Method selection

Decision: pending: 3-4 methods besides the PPO and PPO-RNN controls, mixing different algorithm families with PPO add-ons ([[Candidate Algorithms]]).

Why: the first candidate search returned PPO add-ons only; the user wants truly different, textbook-level algorithms.

Consequences: each chosen method gets a design note in `knowledge/methods/` and a plan file in [[2026-10-20-craftax-project]].
