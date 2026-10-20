---
title: "Craftax experiment infrastructure implementation plan"
type: plan
domain: orchestration
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, infrastructure, mlflow]
---
# Craftax experiment infrastructure implementation plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** one command runs any slice of the experiment grid on the laptop or a cluster node, logs everything to MLflow, and can be stopped at any moment and resumed later without losing or repeating work.

**Architecture:** a run is one (method, config, budget, seed). YAML files declare each method's base config and grid (`configs/methods/<method>.yaml`) and each phase's slice (`configs/phases/<phase>.yaml`); `orchestration/queue.py` expands a phase into runs with deterministic names; `orchestration/runner.py` takes the next unfinished run, resumes it from its last checkpoint or starts it, and logs to the MLflow store. Design decisions: [[Experiment decisions]].

**Tech Stack:** JAX 0.11, craftax 1.6.1, flax, optax, MLflow (SQLite store), mlflow-sysmetrics, PyYAML, existing `wut_murl_craftax.orchestration` (runtime, tracking, reproducibility).

---

### Task 1: Run identity and queue expansion

**Files:**
- Create: `src/wut_murl_craftax/orchestration/queue.py`, `configs/methods/ppo.yaml`, `configs/phases/smoke.yaml`
- Test: `tests/orchestration/test_queue.py`

- [ ] **Step 1: Write the failing tests**: a phase YAML with 2 configs x 2 seeds expands to 4 runs; run names are stable across processes (`<method>-<budget>-<config hash 8>-s<seed>`); a grid entry overrides only its keys; unknown keys fail loudly.
- [ ] **Step 2: Implement** `RunSpec` (frozen dataclass: method, budget, env, seed, config dict, config hash) and `expand_phase(path) -> list[RunSpec]`.
- [ ] **Step 3: Run** `make tests`; expected: the new tests pass.

### Task 2: Resumable runner with MLflow bookkeeping

**Files:**
- Create: `src/wut_murl_craftax/orchestration/runner.py`
- Modify: `src/wut_murl_craftax/orchestration/tracking.py` (tags and resume lookup if missing)
- Test: `tests/orchestration/test_runner.py`

- [ ] **Step 1: Write the failing tests** with a fake 3-update trainer: a FINISHED run is skipped; a run interrupted after update 2 resumes at update 3 with identical final parameters to an uninterrupted run; SIGINT writes a checkpoint and marks the MLflow run `KILLED`; two workers with `--worker-index 0/1 --num-workers 2` never take the same run.
- [ ] **Step 2: Implement** the loop: look up the run by name in MLflow; status FINISHED -> skip; RUNNING/KILLED with checkpoint -> resume (`training/checkpoints.py`); else start. Checkpoint every `checkpoint_every` updates and on SIGINT/SIGTERM.
- [ ] **Step 3: Implement** the CLI `uv run python -m wut_murl_craftax.orchestration.runner --phase configs/phases/<phase>.yaml [--method M] [--max-runs N] [--worker-index i --num-workers k]`.
- [ ] **Step 4: Run** `make tests`.

### Task 3: Trainer interface shared by every method

**Files:**
- Create: `src/wut_murl_craftax/training/interface.py`
- Modify: `src/wut_murl_craftax/training/ppo.py` (from the visualisation port)
- Test: `tests/training/test_interface.py`

- [ ] **Step 1: Write the failing test**: every registered method exposes `init(config, key) -> state`, `train_chunk(state, n_updates) -> (state, metrics)`, `policy(state) -> Policy` and runs 2 updates on Craftax-Classic.
- [ ] **Step 2: Implement** a registry `METHODS: dict[str, TrainerFactory]`; PPO is the first entry; each method file of [[2026-10-20-craftax-project]] adds one.
- [ ] **Step 3: Use** `orchestration.configure_runtime(host_devices=...)` and `device_roles()` inside every trainer (CPU env shards, metal learner on Apple GPU, all-GPU on CUDA; [[Craftax runtime setup]]).

### Task 4: Metrics logged during training

**Files:**
- Create: `src/wut_murl_craftax/evaluation/episode_metrics.py`, `src/wut_murl_craftax/evaluation/__init__.py`
- Test: `tests/evaluation/test_episode_metrics.py`

- [ ] **Step 1: Write the failing tests** on a scripted 200-step Craftax-Classic episode with known events: achievement unlock steps, floor reached, cause of death, steps with each stat at 0, kills per creature type, invalid actions.
- [ ] **Step 2: Implement** per-step extraction from `env_state` (achievements from `env_state.achievements`, not `info`, which is non-zero only at episode end) and per-episode aggregation, all inside `jit`; the metric list is [[2026-10-20-craftax-evaluation-and-metrics]].
- [ ] **Step 3: Log** per update to MLflow: return (% of 226 or of 22), per-achievement success, the in-game metric summaries, losses, entropy, SPS, wall-clock, compile time; per run: peak memory, device roles, host name; system metrics via `mlflow-sysmetrics`.

### Task 5: Fixed-seed evaluation and replayable rollouts

**Files:**
- Create: `src/wut_murl_craftax/evaluation/eval_rollouts.py` (or reuse `visualisation/eval_rollouts.py` from the port)
- Test: `tests/evaluation/test_eval_rollouts.py`

- [ ] **Step 1: Write the failing test**: evaluating the same checkpoint twice gives identical returns and identical stored actions.
- [ ] **Step 2: Implement** evaluation every `eval_every` updates and at the end: 64 fixed seeds, stochastic policy (the scoreboard metric) and greedy policy; store seed + actions per episode as an MLflow artifact (replayable for [[2026-10-20-craftax-visualisation-production]]).

### Task 6: Laptop and cluster profiles

**Files:**
- Create: `configs/profiles/laptop.yaml`, `configs/profiles/cuda.yaml`, `scripts/cluster/run_array.sh`
- Modify: `Makefile` (targets `queue`, `status`, `merge-mlflow`), `.env.example`
- Test: `tests/orchestration/test_profiles.py`

- [ ] **Step 1: Write the failing test**: the laptop profile selects 8 CPU env shards and the metal learner; the CUDA profile puts actor and learner on the GPU; both resolve the same run names.
- [ ] **Step 2: Implement** profiles as runtime settings only (never hyperparameters), so a run is identical wherever it executes; `run_array.sh` maps a scheduler array index to `--worker-index` (Slurm example, scheduler-agnostic flags).
- [ ] **Step 3: Implement** `make status PHASE=...` (pending / running / finished / killed per run) and `make merge-mlflow SRC=<store>` copying runs from a node's store into the main `mlruns.db` without duplicates (run names are the key).

### Task 7: Smoke phase end to end

**Files:**
- Create: `configs/phases/smoke.yaml` (PPO, Craftax-Classic, 2 configs x 2 seeds, 100k steps)

- [ ] **Step 1: Run** `make queue PHASE=smoke`, stop it with Ctrl-C in the middle of a run, run it again; expected: the interrupted run resumes, finished runs are skipped, `make status PHASE=smoke` shows 4 finished.
- [ ] **Step 2: Check** in `make mlflow` that every metric of Task 4 and the eval artifacts of Task 5 are present.
- [ ] **Step 3: Record** measured SPS per method family in [[2026-10-20-craftax-compute-budget]].
