---
title: "Experiment decisions"
type: decision-log
domain: training
parent: "[[wut_murl_craftax]]"
tags: [decision-log, experiments, training]
created: 2026-10-20
updated: 2026-10-29
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

## 2026-10-21 - Phase 0 implementation rulings

Why: each came up while implementing [[2026-10-20-craftax-infrastructure]] and the controls; code in [[orchestration]], [[training]] and [[evaluation]].

- The run budget is injected as `steps` into the config given to `init` and is not part of the config hash.
- Trainers declare `loss_columns`; only those columns are logged and abort-checked.
- Method-metric NaN aborts are opt-in: a metric aborts a run only if the method YAML lists it under `abort.finite`.
- The epsilon is stored in the `ckpt/` params payload (`{"variables", "eps"}` for PQN-RNN), because evaluation builds one policy from `jax.eval_shape(init)` ([[Recording training data]]).
- Eval files carry the provenance keys `method` and `devices` next to `seeds`, `actions` and `length`.
- Tests run on a temporary MLflow store and never touch the repo store; the template's cleanup fixture deleted live checkpoints and was removed.
- Craftax is imported eagerly after `configure_runtime` (`import_craftax`): a first import inside a trace leaks tracers.
- The laptop profile sets `JAX_PLATFORMS=metal,cpu` (`select_platforms`) so a metal learner is found, unless the variable is already set.
- Tuning never overlaps heavy test runs on the laptop: the memory ceiling is shared.

## 2026-10-22 - Control tuning results

Decision: all three controls keep their published or base 1M config as `tuned`; no stage-1 or stage-2 config beat it by more than h.

Why: PPO and PPO-RNN: `tuned` = the best config only if its 3-seed mean beats the published config's 10-seed mean by more than h = 0.05 + 2 sqrt(SE^2 + (s_pub / sqrt(10))^2) (s_pub 0.083 for both); PQN-RNN: ties within h of the base keep the base, with h = 2 sqrt(SE_base^2 + SE_best^2) over 3 seeds each (no 1M published score). Means are of `final_score`; sd is the sample sd over seeds. For PPO, `fine-lr=0.0003` is the published config, so its stage-2 runs are seeds 0 to 2 of the 10.

Consequences: `tuned: {}` in `configs/methods/{ppo,ppo-rnn,pqn-rnn}.yaml` (equal to the base), `best=true` on the 10, 10 and 3 base runs. Methods with `base_from: <control>.tuned` start from the published or base settings. Stage 2 showed small gains only. Rejected: taking the best stage-2 config of PPO-RNN (+0.12) or PQN-RNN (+0.45, base sd 0.85 over 3 seeds).

PPO (s_pub 0.083, h 0.1280 with the 10-seed SE; 0.1351 with the best config's SE):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| e06ac6db published-1m (tuned) | none | 10 | 2.17342 | 0.09122 |
| 845b73b4 lr-2e-4 (best stage 1) | lr 2e-4 | 3 | 2.26138 | 0.05806 |
| c043716c fine-gae_lambda=0.75 (best stage 2) | lr 2e-4, gae_lambda 0.75 | 3 | 2.22396 | 0.01782 |

Best minus published: 0.0880 < h, so tuned = published (e06ac6db).

PPO-RNN (s_pub 0.083, h 0.1354 with the 10-seed SE; 0.1264 with the best config's SE):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| b3ce95c4 published-1m (tuned) | none (ReLU) | 10 | 2.18755 | 0.10654 |
| 04e6b687 shape-64x64 (best stage 1) | envs 64, num_steps 64 | 3 | 2.26074 | 0.07585 |
| efda7c6f fine-ent=0.005 (best stage 2) | envs 64, num_steps 64, ent 0.005 | 3 | 2.30750 | 0.04803 |

Best minus published: 0.1200 < h, so tuned = published (b3ce95c4); the margin to h is small.

PQN-RNN (no 1M published score, h 0.9884 over 3 seeds each):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| 10cbfc16 base-1m (tuned) | none | 3 | 1.27030 | 0.84897 |
| 606530ad lambda-0.8 (best stage 1) | lam 0.8 | 3 | 1.71833 | 0.10930 |
| dcb8a7ef fine-lr=0.000225 (best stage 2) | lam 0.8, lr 2.25e-4 | 3 | 1.66113 | 0.11714 |

Best minus base: 0.4480 < h, so a tie and tuned = base (10cbfc16). At 1M the batch-renorm hidden normalisation (1.44072, sd 0.30554) beat the base (layer-norm hidden, 1.27030), and layer-norm input (0.84326) lost; both within h of the base. The base spread (sd 0.85) is the reason h is wide.

## 2026-10-23 - Phase 1 tuning results

Decision: each Phase 1 method takes the config with the highest 3-seed mean `final_score` over `tune-coarse` and `tune-fine` (ties to the coarser key, the smaller buffer, the smaller timeout multiplier or the cheaper planning variant; no tie occurred); count-bonus is chosen per variant. The tuned block is written with `--mark-tuned`, which also sets `best=true` on the config's tuning runs.

Why: the method briefs fix this rule; unlike the controls, a method is not held to a band around its base control. Means are of `final_score`, sd is the sample sd over the 3 seeds, n is seeds. "Best stage 2" is the best config new in stage 2; a reused stage-1 centre counts as stage 1.

Consequences: `tuned:` is set in `configs/methods/{smdp-skills,count-bonus,sil}.yaml`; dyna-q starts from `smdp-skills.tuned` (`base_from`), and its own `tuned:` was written 2026-10-23 20:35 by `--mark-best dyna-q` (7dbe10f). Rejected: choosing the base control's config for any method.

smdp-skills (base control pqn-rnn 10cbfc16, 1.27030, n 3):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| 0b84c247 a0.05-h200-e0.3-t2-recipe (best stage 1, chosen, stage-2 centre) | alpha 0.05, horizon 200, eps decay 0.3, timeout 2 | 3 | 1.41543 | 0.19584 |
| cbb15b50 fine-alpha=0.1-timeout_mult=4.0 (best stage 2) | alpha 0.1, timeout 4 | 3 | 0.91567 | 0.44200 |

Chosen 0b84c247: +0.145 over the pqn-rnn base. It is the centre of the stage-2 grid and its four direct neighbours are lower (0.28 to 0.77; the other stage-2 configs score -0.30 to 0.92), so the stage-1 win is an isolated, possibly lucky cell; the best new stage-2 config sits at the grid corner (alpha 0.1, timeout 4).

count-bonus, ppo-rnn variant (base control ppo-rnn b3ce95c4, 2.18755, n 10):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| b0b1c535 b0.1-recipe-constant (best stage 1, chosen, stage-2 centre) | beta 0.1, recipe, constant (the base) | 3 | 2.33276 | 0.05290 |
| afe0d0f2 fine-beta=0.2-granularity=tiers (best stage 2) | beta 0.2, tiers | 3 | 2.18712 | 0.12751 |

Chosen b0b1c535: +0.145 over the control; the stage-2 grid found nothing better (new configs 2.16 to 2.19, `tuned: {ppo-rnn: {}}`).

count-bonus, pqn-rnn variant (base control pqn-rnn 10cbfc16, 1.27030, n 3):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| 3c5f5200 b0.03-log2-linear (best stage 1) | beta 0.03, log2, linear | 3 | 1.59011 | 0.61100 |
| e6e1fafa fine-beta=0.015-decay=linear (best stage 2, chosen) | beta 0.015, log2, linear | 3 | 1.72111 | 0.13798 |

Chosen e6e1fafa: +0.451 over the base, +0.131 over the stage-1 best (within its sd 0.61). Its beta is the lower edge of the stage-2 grid, and beta 0.01 was erratic in stage 1 (0.62 to 1.49 over its six configs), so the optimum may sit near unstable values.

sil (base control ppo-rnn b3ce95c4, 2.18755, n 10):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| 6f8690ad l0.3-s2-k1 (best stage 1, chosen, stage-2 centre) | lambda_sil 0.3, S 2, K 1 | 3 | 2.13982 | 0.04932 |
| 4b57f12c fine-sil_steps=1-lambda_sil=0.15 (best stage 2) | lambda_sil 0.15, S 2, K 1 | 3 | 2.10891 | 0.04546 |

Chosen 6f8690ad: -0.048 against the ppo-rnn control (inside the control's sd 0.107), so SIL does not beat PPO-RNN at 1M. The best K is the lower bound of the grid (K 1), so K below 1 was never available; stage 2 tried only K 1 and 2.

dyna-q (base smdp-skills 0b84c247, 1.41543, n 3; base control pqn-rnn 10cbfc16, 1.27030, n 3):

| Config | Knobs | n | Mean | sd |
|---|---|---|---|---|
| 670e2f80 uniform-n1-k0.003 (best stage 1, stage-2 centre) | uniform, n 1, kappa 0.003, M 8 | 3 | 0.94895 | 0.34508 |
| 74c22d85 fine-plan_kappa=0.006-model_slots=4 (best stage 2, chosen) | uniform, n 1, kappa 0.006, M 4 | 3 | 1.36364 | 0.40171 |

Chosen 74c22d85: -0.052 against smdp-skills (inside both sds) and +0.093 over the pqn-rnn base, so planning does not beat the model-free skill table at 1M. It is a corner of the stage-2 grid (largest kappa, smallest M); kappa 0.03 with n 1 scored 0.454 in stage 1. Its kappa is above 0, so the `final-1m` kappa ablation runs kappa 0.

## 2026-10-24 - Phase 3 combination pair

Decision: the combination is `count-bonus-ppo-rnn` + `sil` on PPO-RNN (`combo-1m` config fd6c4333), chosen by `select_pair` from `final-1m` alone (ruling R71, 1dc1b40). Methods are ranked by the IQM of `final_score` over 10 seeds, ties by PoI over the base control (none occurred); the pair is the best one combinable through hooks (ruling R73).

Why: `combo-1m` was to run right after `final-1m`, before `final-1b`, and the plan's rank sum needs the 1B means. With skill methods included, the plan-rule pair is the same (ranks 1 and 2; the PQN-RNN pair is ranks 3 and 4), so there is no R73 deviation; `select_pair` logged no PARTIAL or NOTE line. IQM and PoI come from `evaluation/stats.py`; PoI is P(method > control) over the 100 seed pairs.

Consequences: this departs from the 2026-10-21 decision (pair from both budgets). The ranking is in the header of `configs/methods/combo.yaml`. The 1B re-check is pending: once `final-1b` has run, the plan rule (1M IQM rank plus 1B mean rank, ties by 1M PoI) is recomputed; if it gives another pair, that is recorded here as a deviation and any change for `combo-1b` is a new dated decision, never a silent swap.

| Rank | Method | Base control | n | IQM | PoI vs control |
|---|---|---|---|---|---|
| - | ppo-rnn (control) | - | 10 | 2.18722 | - |
| - | pqn-rnn (control) | - | 10 | 1.46508 | - |
| 1 | count-bonus-ppo-rnn | ppo-rnn | 10 | 2.27844 | 0.680 |
| 2 | sil | ppo-rnn | 10 | 2.13793 | 0.330 |
| 3 | count-bonus-pqn-rnn | pqn-rnn | 10 | 1.55828 | 0.560 |
| 4 | dyna-q | pqn-rnn | 10 | 1.19965 | 0.350 |
| 5 | smdp-skills | pqn-rnn | 10 | 0.90668 | 0.240 |

Outcome (`combo-1m`, 10 seeds, 2026-10-24 07:24 to 08:35): mean 2.13164, sd 0.09252, IQM 2.12066. It is below `count-bonus-ppo-rnn` alone (PoI of the combination over it 0.170) and the PPO-RNN control (0.330), and level with `sil` (0.450), so at 1M the two components do not add up.

## 2026-10-28 - Phase 4 analysis rulings at 1M

Decision: Phase 4 (analysis) runs on the 1M finals only (`final-1m`, `combo-1m`); every step that needs `final-1b` or `combo-1b` is built and tested and reruns with the same commands once 1B exists (ruling R75). The rulings below change how results are reported:

- R77: runs belong to a phase by the `phases` tag, never the experiment name; statistics are per (method, `config_ref`), and the six ablations get their own table, never pooled with tuned runs.
- R78: the 26 runs reused from tuning (ppo 10, ppo-rnn 10, pqn-rnn 3, count-bonus-ppo-rnn 3) lack the online metric families; detectors and figures report them as "not logged", never as 0 or "not detected".
- R79 and R92: detector thresholds that assume 1B are relative to the run's own scale at 1M or reported "uninformative at 1M"; the drawdown floor at 1M is max(0.1, 0.25 x the run's max smoothed score).
- R97: every laptop run trained with two queue workers, so wall-clock time, SPS and machine-wide hardware series are context only, never ranked.
- R98: hoarding (`capped_steps`) counts only materials whose progress-key cap is above 1.
- R102: tuning optimism compares the 3-seed tune-fine mean with fresh final seeds only (not reused from tuning); controls, selected on all their final seeds, read n/a.
- R106: decision-tree fidelity at 1M is low and genuine (the pipeline is aligned: greedy labels match the stored greedy actions on every step checked); L80, L90 and L95 are reported as "> 4096", each fidelity next to the majority-class baseline, and the all-censored score-vs-L90 Spearman is replaced by score vs fidelity at 256 leaves.
- R107: the obstacles table for the tuning runs (`tune-coarse`, `tune-fine`, 494 runs) is deferred at 1M; the training-analysis notes say so.

Why: the analysis was requested before the 1B finals (which need the cluster). The full ruling log with costs-if-wrong is the Phase 4 part of the project's SDD ledger (`.superpowers/sdd/2026-10-20-craftax-project/progress.md`, git-ignored).

Consequences: the tuning-runs obstacles table remains open; the 1M results notes in `knowledge/results/` state which numbers are 1M-only.

## 2026-10-29 - Abort guards end a run for good

Decision: a metric under a method's `abort.permanent` (a guard: full count table, SIL dominance, diverged Q values) or outside an `abort.limits` bound (the recompute ratio of the RNN controls) ends the run for good: tag `abort/guard`, and the queue skips it (`queue.permanently_ended`), as it does for `abort/memory`. A non-finite loss keeps one retry from `state/`, then `abort/bug`. A bound a profile has not measured on its devices (`unmeasured_limits`) is retried like a loss abort (tag `abort/unmeasured`).

Why: a guard reads a state that a retry from the same checkpoint reaches again, so a second run only repeats the abort.

Consequences: a guard-aborted run ends FAILED; `coarse_results` takes finished runs only, so such a config is left out of the ranking (with `--min-complete` when enough configs are complete, or `--allow-partial`), not scored as a failure; without either, the fine-grid selection fails naming its missing runs. The chunk that aborted is not logged and the tag holds the metric and update, not its value. The abort rules are in the method YAML files; `abort.py` and [[orchestration]] describe them.

## 2026-10-29 - 1B launch readiness

Decision: `final-1b` runs every entry's `tuned` config, as `final-1m` does; the controls' tuned blocks are empty, so their run names are unchanged. The skill tables are sized for 1B in `budgets.1b` blocks only: smdp-skills `log2_rows: 22` (dyna-q inherits it through `base_from`) and dyna-q `model_pairs: 2^22`; every 1M config, name and result is unchanged.

Why: the tuned 1M runs of smdp-skills and dyna-q decide at 47-362 table rows and visit 247-1,645 (key, skill) pairs, and both counts are still growing at 1M. Extrapolating the last quarter of each seed to 1B gives up to 0.73 million rows and 2.75 million pairs (a power-law fit up to 0.36 million rows and 1.65 million pairs). The 1M shapes (2^20 rows with `smdp/table_full` at a quarter of them, 2^16 pairs with `dyna/model_full` at the last one) would end such runs for good. The larger shapes only change hash collisions and capacity, about 0.35 GB more for the model; the method is the same.

Consequences: the 1B skill-method runs differ from the 1M ones in table size only; the report says so. The tuned count-bonus tables (2^20 slots, a quarter full) extrapolate to at most 0.2 million keys at 1B and are unchanged. The abort flags of the PQN-RNN controls at 1B use the dead-input share (`pqn/bn_dead_share`), a bound that does not depend on the update count.
