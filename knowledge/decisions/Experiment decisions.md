---
title: "Experiment decisions"
type: decision-log
domain: training
parent: "[[wut_murl_craftax]]"
tags: [decision-log, experiments, training]
created: 2026-10-20
updated: 2026-10-21
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

Decision: three controls (PPO, PPO-RNN, PQN-RNN) and four methods: Count-Based Progress Bonus on PPO-RNN and on PQN-RNN, Self-Imitation Learning on PPO-RNN, SMDP Q-Learning over Hand-Defined Skills, and Dyna-Q over a Skill-Level Progress Model ([[Methods]]).

Why: the first candidate search returned PPO add-ons only; the user asked for truly different, textbook-level algorithms, so a second search over algorithm families added hierarchical RL and model-based planning; the mix answers policy gradient vs value-based, base vs add-on, flat vs hierarchical and model-free vs planning ([[Candidate Algorithms]]).

Consequences: one plan file and one design note per method; PQN-RNN joins the controls because the value-based comparison needs it; the skill methods use prior knowledge (hand-written skills) and are reported in their own block. Rejected: the five PPO add-ons alone; Actor-Critic with Eligibility Traces (kept as reserve); thirteen other families (verdicts in [[Candidate Algorithms]]).

Decision: every add-on builds on the recurrent controls (PPO-RNN, PQN-RNN) through one hook contract: an intrinsic reward per step, an observe callback per rollout and an extra loss after the base update; empty hooks must leave the control bit-identical.

Why: gains are measured against the stronger published bases, on the same code, settings and seeds.

Consequences: `training/hooks.py`, shared by [[2026-10-20-craftax-method-ppo-rnn]] and [[2026-10-20-craftax-method-pqn-rnn]].

## 2026-10-20 - Resets, report length, licences and final score

Decision: every control and method uses optimistic resets: `auto_reset=False` envs inside Craftax_Baselines' `OptimisticResetVecEnvWrapper` with reset ratio 16 (`min(16, envs)`), as the original baselines; PQN-RNN also keeps the repo's random warm-up rollout.

Why: the published PPO, PPO-RNN and PQN-RNN numbers were produced with them, and they are about 2x faster because only one new world is generated per 16 envs per step ([[Craftax performance]], [[Craftax API and wrappers]]).

Consequences: the wrapper is vendored once at 7ce36fa (MIT notice kept, includes the fix for Baselines issue #11) into `src/wut_murl_craftax/training/wrappers.py` ([[2026-10-20-craftax-method-ppo]] Task 2) and every plan references it; the state returned for a done step is a new episode's start state, never the terminal state, which the count bonus masks ([[Count-Based Progress Bonus design]]); a comparison against auto-reset envs is a debug check, not a step to turn resets on. Rejected: auto-reset envs for all runs (one wrapper fewer, but slower and a deviation from every published control).

Decision: the report has 4 pages of content; references, the generative-AI disclosure and acknowledgements go on page 5.

Why: the Interspeech rules for regular papers allow page 5 for exactly these parts, so the content keeps all 4 pages.

Consequences: [[2026-10-20-craftax-report]] budgets 4.00 pages for the sections and checks that the content ends on page 4. Rejected: 4 pages in total, references included.

Decision: MIT for everything we release: data, metrics, figures, tables, GIFs, videos, checkpoints, the Space code and this repository (`LICENSE`: Copyright (c) 2026 Paweł Pozorski).

Why: one permissive licence for code and results is simple to apply and matches Craftax and Craftax_Baselines (both MIT).

Consequences: `LICENSE`, `README.md`, `CITATION.cff`, `pyproject.toml` and the Hugging Face cards (`license: mit`) say MIT ([[2026-10-20-craftax-huggingface-release]]); third-party code keeps its own licence header (`training/batch_renorm.py` from purejaxql stays Apache-2.0, the vendored Craftax_Baselines wrappers keep their MIT notice); a video with music is uploaded only if the music licence allows redistribution. Rejected: CC-BY-4.0 for data and media with Apache-2.0 for code; CC-BY-NC-4.0.

Decision: the final score of a run is the mean training-episode return, as % of the maximum 226, over the episodes that finished in the last 2% of the budget, widened backwards to at least 100 episodes; the fixed-seed evaluation of the final checkpoint is reported alongside it.

Why: it is the scoreboard metric, so our numbers compare with [[Craftax Baselines results]] and [[Existing Implementations]]; averaging over many episodes is less noisy than one evaluation of the last checkpoint.

Consequences: tuning, reproduction bands, statistics and the report use this score ([[2026-10-20-craftax-evaluation-and-metrics]], [[2026-10-20-craftax-comparison-and-results]], [[2026-10-20-craftax-report]]); T1 shows the stochastic and greedy fixed-seed evaluation beside it. Rejected: the final fixed-seed evaluation only.

## 2026-10-21 - Run identity, resume state, evaluation, step events, skills, licences, combination and gates

Decision: a run is named `<registry-name>-<budget>-<confighash8>-s<seed>` and looked up across the whole MLflow store, not per experiment; a run belongs to every phase that lists it, recorded in the MLflow tag `phases` (a comma list, extended when a later phase reuses the run); each phase is still an MLflow experiment for browsing.

Why: a phase-local lookup created two runs for one (config, budget, seed), for example PPO's published config in `tune-coarse` and again in `final-1m`.

Consequences: `make status` and the exports filter by the tag `phases`; `make merge-mlflow` merges the tags; a reused finished run only gets the evaluation its new phase needs ([[2026-10-20-craftax-infrastructure]] Tasks 1, 2, 6). Rejected: phase name inside the run name (duplicated compute).

Decision: resuming restores the whole trainer state (params, optimizer state, env states, RNG keys, method state, BatchRenorm statistics, run-log and episode accumulators) from `state/`, written every `checkpoint_every` updates and on SIGINT/SIGTERM; `ckpt/` keeps only the log-spaced params for eval and video; the run directory is the run's MLflow artifact directory.

Why: params-only checkpoints cannot resume a run to the same result; a resumed run must equal an uninterrupted one.

Consequences: `save_state` / `load_state` in `training/checkpoints.py`; every method's state is one pytree and its resume test compares against an uninterrupted run ([[2026-10-20-craftax-infrastructure]] Task 2). Rejected: restarting interrupted runs from scratch.

Decision: fixed-seed evaluation stores seed, actions and step events only (no observations), runs each checkpoint until every eval episode ends, evaluates tuning runs at the final checkpoint only and final runs at every log-spaced checkpoint; the eval time is part of the compute budget.

Why: storing observations of 64 seeds x 2 modes per checkpoint cost about 106 GB per checkpoint and mode, and evaluating every checkpoint of every tuning run cost about 280M eval steps per 1M run; replaying seed + actions rebuilds the observations exactly.

Consequences: one eval module, `visualisation/eval_rollouts.py`, builds every method's policy from the registry ([[2026-10-20-craftax-infrastructure]] Task 5); [[2026-10-20-craftax-compute-budget]] adds about 42 laptop hours at 1M and 45 hours at 1B (3 seeds, counted at the laptop rate) for evaluation. Rejected: fixed-length eval rollouts (cut episodes); evaluating every checkpoint of tuning runs.

Decision: the wrapper step (`training/wrappers.py`) computes `StepEvents` (unlocks, floor change, damage, kills, deaths, ...) from the pre-reset next state, before the optimistic reset swaps in a new episode, and returns them with the transition; nobody compares states across a done step.

Why: with optimistic resets the state returned on a done step is a new episode's start, so an unlock or death on the last step was invisible to evaluation, SIL, skill tests and the count bonus.

Consequences: training metrics, evaluation, SIL, skill success tests and the count bonus all read these events; `evaluation/events.py` defines the fields ([[2026-10-20-craftax-method-ppo]] Task 2, [[2026-10-20-craftax-infrastructure]] Task 4). This refines the 2026-10-20 resets decision: the events of a done step come from the terminal state, while the state returned for it is still the new episode's start. Rejected: masking done steps (loses the last step's events).

Decision: skill methods act without privileged state: skill start, termination and success use only the observation and the reward the agent receives (`unlock_any` succeeds when the achievement part of the reward, reward minus 0.1 x health change read from the observation, is above 0; `fight` succeeds when an adjacent hostile creature disappears after the agent's attack or a kill achievement's reward arrives); privileged state is used for logging only. SIL's slot index uses `events.unlocked`, privileged but only for selecting training data.

Why: a policy that reads hidden game state at decision time is not comparable with the controls, which see the observation only.

Consequences: [[2026-10-20-craftax-method-smdp-skills]], [[2026-10-20-craftax-method-dyna-q]] and [[2026-10-20-craftax-method-sil]] document the rule; the skill block still marks the prior knowledge of hand-written skills. Rejected: success tests on the full `EnvState`.

Decision: the three YouTube uploads use Creative Commons BY; everything published on Hugging Face and this repository stay MIT.

Why: YouTube offers only the standard YouTube licence and Creative Commons BY, no MIT option (decided by the user).

Consequences: [[2026-10-20-craftax-visualisation-production]] and [[2026-10-20-craftax-huggingface-release]] state CC BY for the uploads and MIT for the same videos in the dataset repo. Rejected: the standard YouTube licence.

Decision: the combination's pair is chosen after both `final-1m` and `final-1b`, from both budgets, as the best two compatible methods (same base control, no hook field set by both); it runs as `combo-1m` (10 seeds) and `combo-1b` (3 seeds, see the next section), without retuning; if the pair contains a skill method, the combination is reported in the "uses hand-written skills" block.

Why: choosing from 1M alone could pick a pair that does not hold at 1B, and a combination using hand-written skills inherits their prior knowledge.

Consequences: Phase 3 task in [[2026-10-20-craftax-project]]; every analysis reads the four final phases `final-1m final-1b combo-1m combo-1b`. Rejected: choosing the pair at 1M and running `combo-1b` in parallel with `final-1b`.

Decision: a reproduction gate reports the gap between our control and the published number (band h from `evaluation.reproduction`) and the comparison continues with our control; it never stops the comparison.

Why: every method is compared against its base control on the same code, so a gap to the published number changes how the controls are read, not whether the methods can be compared.

Consequences: the gap is reported in [[2026-10-20-craftax-comparison-and-results]] and [[2026-10-20-craftax-report]]; the control plans keep their diagnostic steps for a failed gate. Rejected: blocking the comparison until every control reproduces.

## 2026-10-21 - Seeds at 1B

Decision: `final-1m` and `combo-1m` keep 10 seeds per configuration; `final-1b` and `combo-1b` use 3 seeds. At 1B the results are reported as mean and median with 95% confidence intervals (stratified bootstrap and t-based) and read as indicative; probability of improvement and Holm-corrected significance claims are made at 1M only. This replaces "10 seeds per final configuration" of the 2026-10-20 seeds decision for the 1B finals.

Why: the 1B runs are about 99% of all environment steps; the user chose to cut compute (runs 888 -> 825; the 1B training and evaluation hours fall to 3/10, about 83 GPU hours in [[2026-10-20-craftax-compute-budget]]).

Consequences: 1B results are indicative; significance is tested at 1M (with 3 against 3 seeds a two-sided Mann-Whitney U cannot reach 0.05); the IQM of 3 seeds equals the median, so 1B tables report mean and median; the 1B reproduction band of each control is computed with n = 3 on our side and is wider ([[2026-10-20-craftax-method-ppo]] Task 5, `evaluation.reproduction`); the combination pair is ranked by the 1M IQM and the 1B mean ([[2026-10-20-craftax-project]] Phase 3); run counts and hours in [[2026-10-20-craftax-compute-budget]], statistics in [[2026-10-20-craftax-evaluation-and-metrics]] and [[2026-10-20-craftax-comparison-and-results]]. Rejected: 10 seeds at 1B; adaptive 5 + 5 seeds at 1B; 5 seeds everywhere.
