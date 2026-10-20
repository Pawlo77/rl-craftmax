---
title: "Craftax training analysis implementation plan"
type: plan
domain: analysis
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, analysis, training, failure-modes]
---
# Craftax training analysis implementation plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** for each method in [[2026-10-20-craftax-project]] (the controls and the combined method included) and each benchmark, a written account of which scenarios were hard, how the hardest ones were eventually solved, and which obstacles the method met, every claim backed by an MLflow number or a replayable episode.

**Architecture:** a new package `src/wut_murl_craftax/analysis/` reads the exported tables of [[2026-10-20-craftax-evaluation-and-metrics]] (`outputs/metrics/<phase>/`); `analysis/hard.py` finds hard scenarios, `analysis/cases.py` finds and exports case episodes (replay, 16 px master, GIF), `analysis/obstacles.py` runs automatic obstacle detectors and fills the obstacles table. Outputs go to `outputs/analysis/<budget>/<method>/` and are logged as artifacts of run `training-analysis-<budget>-<method>` in MLflow experiment `analysis`. The hand-written account is one note per method, `knowledge/results/<Method> training analysis.md`.

**Tech Stack:** pandas, numpy, matplotlib, MLflow, `wut_murl_craftax.evaluation` (records, stability), `wut_murl_craftax.visualisation` (`eval_rollouts.as_rollout`, `render.render_master`, `cuts.gif`, `overlay.render_overlay`).

## Questions answered per method

- Which floors were never reached, and which were reached in fewer than 5% of final eval episodes ([[Craftax floors and progression]])?
- Which achievements were never unlocked by any seed at any checkpoint, and which stayed rare (success rate < 5% at the final checkpoint, [[Craftax achievements]])?
- What killed the agent: cause-of-death shares at the final checkpoint and how they moved over training?
- Where did the agent stall: stalls per episode, median stall length, share of stalls escaped, the floor and achievement where stalls start?
- How were long stalls broken: what changed in behaviour just before the unlock that ended the stall?
- How was the deepest floor reached: route, inventory, fights and survival on the way?
- Which obstacles appeared during training (table below), in how many seeds, at which env steps?

## Thresholds

- Rare achievement: success rate < 5% in final stochastic eval episodes.
- Never unlocked: 0 unlocks in training episodes and in every eval of every seed.
- Floor not reached: max floor below `f` in every final eval episode of every seed.
- Stall: >= 300 steps without a new unlock (definition in [[2026-10-20-craftax-evaluation-and-metrics]]).
- Long stall for a case study: the 5 longest escaped stalls per method and benchmark in final-checkpoint eval episodes, plus the longest one at each of 3 earlier checkpoints.
- Deep-floor case: the 3 episodes per method and benchmark with the deepest floor, ties broken by achievements unlocked.
- Rare-unlock case: one episode per rare achievement that unlocked it.
- Death case: one representative episode for each of the 3 most frequent causes of death.

## Obstacle detectors

| Obstacle | Evidence rule | Source |
|---|---|---|
| Instability | >= 1 drop in the run | `evaluation.stability.drops` |
| Collapse | collapse flag set | `evaluation.stability.collapsed` |
| Plateau | final score minus score at 50% of the budget < 1 (% of max) | per-update score |
| Entropy collapse | mean policy entropy < 0.2 nats before 10% of the budget | `train/action_entropy`, logged policy entropy |
| Value failure | explained variance < 0 for >= 5% of the budget, where the method logs it | losses |
| Numerical failure | any NaN or inf in a logged loss, or gradient norm > 10 x its running median | losses, `grad_norm` |
| Reward hacking | non-achievement share of return > 20% (health term or a method's auxiliary reward), or auxiliary reward rising for >= 10% of the budget while the score is flat; confirmed by a repeated action cycle of >= 50 steps in eval replays | achievement reward share, method metrics, eval actions |
| Exploration failure | no new first unlock in the last 50% of the budget and fewer floors reached than the better control | first-unlock race, funnel |
| Survival failure | > 50% of final-checkpoint deaths caused by needs drain | cause of death |
| Skill forgetting | an achievement's success rate falls from >= 20% to < 5% later in the same run | `train/ach/<name>` |

## Obstacles table template

One table per method in its note, one row per detected obstacle; a row may also come from manual inspection (evidence column says so).

| Obstacle | Benchmark | Runs affected | When (env steps) | Evidence | Mitigation tried | Outcome | Replay or plot |
|---|---|---|---|---|---|---|---|
| e.g. collapse | 1B | 3 / 10 | 420M to 510M | drawdown 0.62, no recovery | lower learning rate in tuning phase 2 | 0 / 10 after the change | `training-analysis-1b-<method>/plots/drawdown.png` |

- "Runs affected" counts final runs; tuning runs go in a second table with the same columns.
- "Mitigation tried" names the config change; each such change appends a dated section to [[Experiment decisions]].
- Replays and plots are referenced by MLflow run name and artifact path in backticks; `outputs/` is not in git, so no wikilinks to it.

---

### Task 1: Package and loaders

**Files:**
- Create: `src/wut_murl_craftax/analysis/__init__.py`, `src/wut_murl_craftax/analysis/load.py`
- Test: `tests/analysis/conftest.py` (a small synthetic export: 2 methods x 2 budgets x 3 seeds), `tests/analysis/test_load.py`

- [ ] **Step 1: Write the failing tests**: `load_phase(dir)` returns `runs`, `curves` and `episodes` frames with the export schema; `select(frames, method=, budget=)` filters every frame consistently; a missing column raises a clear error.
- [ ] **Step 2: Implement** `load_phase`, `select` and `final_checkpoint(episodes, run)`.
- [ ] **Step 3: Run** `uv run pytest tests/analysis/test_load.py -q`.

### Task 2: Hard scenarios

**Files:**
- Create: `src/wut_murl_craftax/analysis/hard.py`
- Test: `tests/analysis/test_hard.py`

- [ ] **Step 1: Write the failing tests** on the fixture: floor reach rates with bootstrap CIs over seeds; never-unlocked and rare lists follow the thresholds; cause-of-death shares sum to 1; stall statistics (stalls per episode, median length, escape rate, start floor histogram).
- [ ] **Step 2: Implement** `floor_reach`, `achievement_status` (final rate, best rate over training, k of n seeds ever), `death_causes` (final and per eval checkpoint), `stall_stats`, each returning a DataFrame.
- [ ] **Step 3: Implement** the plots `floor_reach.png`, `achievement_status.png` (rows grouped by tier and floor), `death_causes_over_training.png`, `stall_lengths.png` (CCDF), same dark style as `visualisation.charts`.
- [ ] **Step 4: Run** `uv run pytest tests/analysis/test_hard.py -q`.

### Task 3: Case finder and replay export

**Files:**
- Create: `src/wut_murl_craftax/analysis/cases.py`
- Test: `tests/analysis/test_cases.py`

- [ ] **Step 1: Write the failing tests**: in a fixture with a 400-step stall ended by an unlock, `find_cases` ranks it first among stall cases; deep-floor, rare-unlock and death cases follow the thresholds; an exported case rollout replays to the same final achievements as the stored eval episode; its GIF lasts at most 20 s.
- [ ] **Step 2: Implement** `find_cases(frames, method, budget) -> cases DataFrame` (id, run, update, mode, seed, kind, start, end, peak, caption).
- [ ] **Step 3: Implement** `export_case(case, out_dir)`: rollout file via `visualisation.eval_rollouts.as_rollout`, 16 px master via `visualisation.render.render_master`, GIF of the window around the peak via `visualisation.cuts.gif`, and for stall cases an agent-internals clip of the last 100 steps before the unlock via `visualisation.overlay.render_overlay`.
- [ ] **Step 4: Implement** `describe_case(case) -> dict` with the facts the write-up needs: floor, stats and inventory at the stall start and at the unlock; action histogram inside the stall vs the whole episode; coverage gained during the stall; the last 50 actions before the unlock; damage taken and close calls inside the window.
- [ ] **Step 5: Run** `uv run pytest tests/analysis/test_cases.py -q`.

### Task 4: Obstacle detectors and table

**Files:**
- Create: `src/wut_murl_craftax/analysis/obstacles.py`
- Test: `tests/analysis/test_obstacles.py`

- [ ] **Step 1: Write the failing tests**: one synthetic curve or metric series per detector that fires it and one that does not; a method without a logged metric (no explained variance) yields "not logged", not "absent"; `obstacles_table(rows)` renders exactly the template columns in Markdown.
- [ ] **Step 2: Implement** one function per detector returning `Obstacle(name, run, start, end, evidence)` and `detect_all(frames, method, budget)`.
- [ ] **Step 3: Implement** the repeated-cycle check for reward hacking: the shortest action period of >= 50 consecutive steps in eval actions (autocorrelation of the action sequence).
- [ ] **Step 4: Run** `uv run pytest tests/analysis/test_obstacles.py -q`.

### Task 5: Run the analysis for every method

**Files:**
- Create: `src/wut_murl_craftax/analysis/training.py` (CLI)

- [ ] **Step 1: Write the failing test** in `tests/analysis/test_training_cli.py`: the CLI on the fixture writes `hard/`, `cases/`, `obstacles.md` and `plots/` for each method and logs them to a temporary MLflow store under `training-analysis-<budget>-<method>`.
- [ ] **Step 2: Implement** `uv run python -m wut_murl_craftax.analysis.training --phase <final phase> --method <method|all> --out outputs/analysis/<budget>`.
- [ ] **Step 3: Run** it for both final phases (names in [[2026-10-20-craftax-compute-budget]]) and `--method all`; check the MLflow runs in `make mlflow`.
- [ ] **Step 4: Copy** every case rollout into `outputs/visualisation/runs/cases/` so [[2026-10-20-craftax-visualisation-production]] picks them up as highlight candidates.

### Task 6: Per-method write-up

**Files:**
- Create: `knowledge/results/<Method> training analysis.md` for each method and control (template `knowledge/_templates/concept.md`, `domain: analysis`, `parent: "[[2026-10-20-craftax-project]]"`)
- Modify: `knowledge/AGENTS.md` (row for `results/`: one note per method plus the cross-method results notes)

- [ ] **Step 1: Write** section "Hard scenarios": floors not reached, never and rare achievements, death causes, stall statistics, each bullet with its number and the MLflow run that holds it.
- [ ] **Step 2: Write** section "How complex scenarios were solved": one paragraph per case (situation, behaviour during the stall, what broke it, outcome), facts from `describe_case`, the GIF and replay referenced by run name and artifact path.
- [ ] **Step 3: Write** section "Obstacles" with the filled template table, one table per benchmark.
- [ ] **Step 4: Write** section "Comparison with the controls": the same three questions answered for PPO and PPO-RNN in one line each, linking their notes.
- [ ] **Step 5: Check** every number against the MLflow UI for 3 random bullets per note; numbers come from executed output only.
- [ ] **Step 6: Run** `uv run python scripts/check_vault.py` and `make check`.

### Task 7: Hand-off to results, report and video

**Files:**
- Create: `knowledge/architecture/analysis.md` (template `knowledge/_templates/module.md`)

- [ ] **Step 1: Write** a 5-bullet summary per method at the top of its note; [[2026-10-20-craftax-report]] (Discussion) and [[2026-10-20-craftax-comparison-and-results]] quote these bullets.
- [ ] **Step 2: Mark** in `outputs/analysis/<budget>/cases.csv` the cases worth showing on video (column `video` = yes) for the reviewer of [[2026-10-20-craftax-visualisation-production]].
- [ ] **Step 3: Write** the module note (responsibility, public surface, CLI) for the modules of this plan; run `make knowledge`.
