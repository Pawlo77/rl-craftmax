---
title: "Craftax PPO control implementation plan"
type: plan
domain: training
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, method, control, ppo]
---
# Craftax PPO control implementation plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the feed-forward PPO control reproduces the published Craftax-Symbolic scores (2.2% at 1M, 11.9% at 1B, [[Craftax Baselines results]]) inside an acceptance band, is lightly tuned at 1M, and its tuned shared settings become the starting point of every PPO-based method in [[2026-10-20-craftax-project]].

**Architecture:** the trainer already exists: `src/wut_murl_craftax/training/ppo.py` (the MLP PPO ported from Craftax_Baselines `ppo.py`; adapted to the trainer interface by Task 3 of [[2026-10-20-craftax-infrastructure]]). This plan adds the paper configurations as `configs/methods/ppo.yaml`, a port audit against the original, the published reference scores (`configs/reference/published_scores.yaml`) with a shared reproduction check (`evaluation/reproduction.py`, also used by PPO-RNN and PQN-RNN), and the five phases. Design decisions: [[PPO design]].

**Tech Stack:** JAX 0.11, craftax 1.6.1, flax, optax, MLflow, PyYAML, existing `wut_murl_craftax.training` (ppo, records, checkpoints) and `wut_murl_craftax.orchestration` (queue, runner, runtime).

---

## Design

- Clipped PPO with GAE, separate actor and critic MLPs (3 x Dense(512), tanh), linear learning-rate annealing; the losses of the existing port are kept bit-for-bit.
- Two published configurations: Craftax-1M (paper Table 11) and Craftax-1B (paper Table 7); both are set explicitly, never taken from code defaults (the port's defaults are a Classic debugging setup).
- Reproduction check at 1M (published config, 10 seeds) before any method tuning starts, and at 1B (the final-1b phase itself); acceptance band and debug order in Task 5.
- Light tuning: Stage 1 checks only published alternatives one knob at a time; the tuned config replaces the published one only if it wins by more than the band half-width.
- Optimistic resets, as the original: `auto_reset=False` envs inside Craftax_Baselines' `OptimisticResetVecEnvWrapper` with reset ratio 16 (`min(16, envs)`), vendored into `src/wut_murl_craftax/training/wrappers.py` (Task 2); every control and method uses the same reset setup ([[Experiment decisions]]); everything else matches `ppo.py` at commit 7ce36fa as well.
- Decisions and their reasons: [[PPO design]].

## Hyperparameter grid

Stage 1 (coarse, Craftax-1M, 3 seeds; the published config gets 10 seeds because it is the reproduction check): one knob at a time away from Table 11, so the grid is 7 configs, not a Cartesian product.

| Knob | Values (published first) | Why |
|---|---|---|
| Rollout shape (envs x steps) | 256 x 16 (Table 11); 1024 x 64 with lr 2e-4 (Table 7 at 1M, 15 updates) | the paper tuned the two budgets separately; checks that the 1M table is needed |
| Learning rate | 3e-4; 2e-4; 4e-4 | the 1M values the paper tried (Table 11) |
| GAE lambda | 0.8; 0.7; 0.9 | tried at 1M {0.7, 0.8, 0.9, 0.95}; the neighbours of the chosen value |
| Entropy coefficient | 0.01; 0 | tried at 1B {0, 0.01}; a third-party rerun found entropy matters ([[PPO (Matthews 2024)]], RePPO row) |

Configs: published (1) + shape (1) + lr (2) + lambda (2) + entropy (1) = 7; runs = 6 x 3 + 10 = 28.

Stage 2 (fine, Craftax-1M, 3 seeds): take the best Stage 1 config; pick the 1-2 numeric knobs with the largest Stage 1 effect (range of 3-seed mean final scores over that knob's values); try 2 neighbours each; the rollout shape is categorical and never refined.

| Knob | Fine values around the best coarse value b |
|---|---|
| Learning rate | b x 0.75, b x 1.5 |
| GAE lambda | b - 0.05, b + 0.05 |
| Entropy coefficient | b x 0.5, b x 2 (if b = 0: 0.001, 0.003) |

Expected: 2 knobs x 2 values = 4 configs, 12 runs. Selection: the best Stage 1 or 2 config becomes `tuned` only if its 3-seed mean beats the published config's 10-seed mean by more than the reproduction band half-width h (Task 5); otherwise `tuned` = published.

Fixed settings (shared PPO settings for every PPO-based method start from the `tuned` column; [[Craftax Baselines algorithms and hyperparameters]]):

| Setting | Craftax-1M (Table 11) | Craftax-1B (Table 7) | Note |
|---|---|---|---|
| Environment | `Craftax-Symbolic-v1` | `Craftax-Symbolic-v1` | Classic only in `smoke` |
| Envs x steps per rollout | 256 x 16 (4,096) | 1024 x 64 (65,536) | updates 244 and 15,258 |
| Update epochs, minibatches | 4, 8 | 4, 8 | minibatch 512 and 8,192 steps |
| Learning rate | 3e-4, annealed to 0 | 2e-4, annealed to 0 | Adam eps 1e-5 |
| Gamma, GAE lambda | 0.99, 0.8 | 0.99, 0.8 | |
| Clip epsilon, value coef, entropy coef | 0.2, 0.5, 0.01 | 0.2, 0.5, 0.01 | value loss clipped |
| Max grad norm | 1.0 | 1.0 | code default, not in the tables |
| Width, activation | 512, tanh | 512, tanh | `--activation` is a no-op in the original; tanh is the class default |
| Resets | optimistic, ratio 16 (16 new worlds per step) | optimistic, ratio 16 (64 new worlds per step) | as the original: `auto_reset=False` + `OptimisticResetVecEnvWrapper` |

- 1B: no tuning; final-1b runs Table 7 exactly (PPO has no method knobs), which makes final-1b the 1B reproduction check.

## Runs and laptop time estimate

Throughput (estimate): PPO on Craftax-Symbolic 3,000 steps/s with 8 CPU shards, Craftax-Classic 10,000 steps/s; overhead factor 1.0 (PPO is the reference of the shared throughput assumptions; estimate); +1 min compile per run.

| Phase (MLflow experiment) | Configs | Seeds | Steps per run | Runs | Est. steps/s | Est. hours (laptop, sequential) |
|---|---|---|---|---|---|---|
| `smoke` (Craftax-Classic) | 2 | 1 | 200k | 2 | 10,000 | 0.04 |
| `tune-coarse` | 7 | 3 (published: 10) | 1M | 28 | 3,000 | 3.06 |
| `tune-fine` | 4 | 3 | 1M | 12 | 3,000 | 1.31 |
| `final-1m` | 1 | 10 | 1M | 10 | 3,000 | 1.09 |
| `final-1b` | 1 | 10 | 1B | 10 | 3,000 | 926.09 |
| Total | | | | 62 | | 931.6 |

- One 1M run: 1e6 / 3,000 + 60 s = 393 s; one 1B run: 333,393 s = 92.6 h.
- Optimistic resets should raise the measured throughput above these estimates (the authors report about 2x, [[Craftax performance]]); Task 3 records the measurement.
- final-1b on a CUDA cluster GPU (authors' hardware, estimate): 1e9 / 266,961 steps/s x 1.0 = 1.04 h per run, 10.4 h for 10 seeds.
- Run counts feed [[2026-10-20-craftax-compute-budget]].

## Tasks

### Task 1: Paper configurations and reference scores

**Files:**
- Modify: `configs/methods/ppo.yaml` (created by infrastructure Task 1), `src/wut_murl_craftax/orchestration/queue.py` (key `base_from`)
- Create: `configs/reference/published_scores.yaml`
- Test: `tests/training/test_ppo_configs.py`, `tests/orchestration/test_queue.py`

- [ ] **Step 1: Write the failing tests**: the resolved `ppo` base config equals Table 11 field by field and its `budgets.1b` override equals Table 7 (table in Fixed settings); the 7 Stage 1 entries differ from the base in exactly the keys listed in the Stage 1 table; a method config with `base_from: ppo.tuned` resolves to the `tuned` block of `ppo.yaml` plus its own keys; `published_scores.yaml` holds PPO 2.2 / 11.9, PPO-RNN 2.3 / 15.3, PQN-RNN 16.0 (1B), each with source URL and s_pub.
- [ ] **Step 2: Write** `ppo.yaml`: `base` (Table 11), `budgets.1b` (Table 7 overrides), `grid.coarse` (7 named entries, `published-1m` with `seeds: 10`), `grid.fine` (the rule as data: knobs and multipliers), `tuned: null` (filled in Task 6).
- [ ] **Step 3: Implement** `base_from` in `queue.py` (a method config may start from another method's `tuned` block; unknown keys still fail loudly).
- [ ] **Step 4: Run** `uv run pytest tests/training/test_ppo_configs.py tests/orchestration/test_queue.py -q`; expected: pass.

### Task 2: Port audit against Craftax_Baselines and PPO metrics

**Files:**
- Create: `src/wut_murl_craftax/training/wrappers.py` (vendored from Craftax_Baselines `wrappers.py` at 7ce36fa: `GymnaxWrapper`, `LogWrapper`, `OptimisticResetVecEnvWrapper`; its MIT notice kept in the file header; the one copy every control and method imports)
- Modify: `src/wut_murl_craftax/training/ppo.py`
- Test: `tests/training/test_ppo.py`, `tests/training/test_wrappers.py`

- [ ] **Step 1: Write the failing tests** for the wrapper (Craftax-Classic, 32 envs, ratio 16): a non-done env's returned state equals the plain step; every done env's returned state is the start state of a new episode (timestep 0, fresh inventory), never the terminal state; `num_envs % reset_ratio != 0` raises; with fewer done envs than resets (the issue #11 case) only done envs change; the ratio used is `min(16, envs)`.
- [ ] **Step 2: Vendor** the wrappers unchanged in behaviour and build the training env as the original: `make_craftax_env_from_name(name, auto_reset=False)`, then `LogWrapper`, then `OptimisticResetVecEnvWrapper(envs, reset_ratio=min(16, envs))`; config keys `resets: optimistic` (default; `auto` only for the debug comparison of Task 5) and `reset_ratio: 16`.
- [ ] **Step 3: Write the failing tests** for the port: GAE on a hand-computed 4-step, 2-env example with a done at step 2 (gamma 0.9, lambda 0.8; expected advantages written in the test); the clipped policy and value losses on a 2-sample batch against values computed by hand; the learning rate is `lr` at the first minibatch and `lr / n_updates` at the last update; `n_updates` drops the last partial update (1e6 // 4096 = 244); two runs with the same seed on Classic give identical parameters after 2 updates; `clip_frac` and `grad_norm` appear in the loss summary.
- [ ] **Step 4: Add** `ppo_terms(logits, value, batch, cfg)` (the loss math, so PPO-RNN reuses it) and the extra summaries `clip_frac` (share of ratios outside 1 +- clip) and `grad_norm` (before clipping); outputs of the existing loss stay identical (the old test values must not change).
- [ ] **Step 5: Check** the registry entry `METHODS["ppo"]` from infrastructure Task 3 accepts every key of `ppo.yaml` (`width`, `envs`, `num_steps`, `lr`, `gae_lambda`, `ent`, ...).
- [ ] **Step 6: Run** `uv run pytest tests/training/test_ppo.py tests/training/test_wrappers.py -q`, then `make tests`.

### Task 3: Smoke run

**Files:**
- Modify: `configs/phases/smoke.yaml` (add PPO: Craftax-Classic, 2 configs `published-1m` and `lr-4e-4`, 1 seed, 200k steps)

- [ ] **Step 1: Run** `make queue PHASE=smoke METHOD=ppo`; expected: 2 finished runs in MLflow experiment `smoke`.
- [ ] **Step 2: Check** in `make mlflow`: Classic return rises above the uniform-random return measured in the same task (`uv run python -m wut_murl_craftax.visualisation.rollout --game classic --policy random --seeds 1 2 3 4 5 6 7 8 --max-steps 10000`), no NaN, every metric of Task 7 present.
- [ ] **Step 3: Record** the measured steps/s in [[2026-10-20-craftax-compute-budget]].

### Task 4: Reproduction check helper

**Files:**
- Create: `src/wut_murl_craftax/evaluation/reproduction.py`
- Test: `tests/evaluation/test_reproduction.py`

- [ ] **Step 1: Write the failing tests**: `band(ours, published, s_pub)` returns h and pass/fail; with 10 seeds of sd 0.083 around 2.2 the planning h is 0.124 (+- 0.001); a mean of 2.40 fails, 2.25 passes.
- [ ] **Step 2: Implement** `band` and the CLI `uv run python -m wut_murl_craftax.evaluation.reproduction --method <slug> --budget 1m|1b --phase <phase>` that reads the final scores from MLflow, prints mean, SE_ours, h, pass/fail, and logs them as tags `repro/*` on the phase.
- [ ] **Step 3: Run** `uv run pytest tests/evaluation/test_reproduction.py -q`.

### Task 5: Stage 1 and the 1M reproduction check

**Acceptance band** (shared by the three controls; numbers per control in their plans):

- Published reference: README scoreboard, training-episode reward as % of 226, 10 seeds; no spread is published (curves show 1 standard error only).
- Spread proxy s_pub (per-seed sd, percentage points): the largest third-party sd at the same budget; 1M: 0.083 (PPO-RNN 1M, mean of the last 5 updates per seed, derived from the Simulus per-run data, `results/data/craftax/craftax_baselines.json`); 1B: 0.91 (SimBa rerun, raw sd 2.06 / 2.26; [[PPO (Matthews 2024)]]).
- Our score: the final score of [[2026-10-20-craftax-evaluation-and-metrics]] (training episodes in the last 2% of the budget), 10 seeds, standard error SE_ours.
- Band half-width: h = 0.05 + 2 sqrt(SE_ours^2 + (s_pub / sqrt(10))^2); 0.05 covers the one-decimal rounding of the published number; the rest is a two-sample 95% interval.
- Planning bands (assuming our sd equals s_pub): 1M 2.2 +- 0.12 (2.08 to 2.32); 1B 11.9 +- 0.86 (11.04 to 12.76).
- Pass: |our mean - published| <= h. Fail: follow the debug order; after the last step, record the gap and continue with our control as the comparison base.

```mermaid
flowchart TD
  A["published config, 10 seeds, 1M"] --> B{"inside the band?"}
  B -->|yes| C["pass: Stage 1 and 2 tuning"]
  B -->|no| D["1 config diff vs Table 11"]
  D --> E["2 metric definition and final window"]
  E --> F["3 CPU-only learner (device numerics)"]
  F --> G["4 check: compare against auto-reset"]
  G --> H["5 original ppo.py in an isolated env"]
  H --> I{"original also outside?"}
  I -->|yes| J["record: environment or version gap"]
  I -->|no| K["bisect our port against the original"]
  J --> C
  K --> A
```

**Files:**
- Modify: `configs/phases/tune-coarse.yaml` (add `ppo`, all `grid.coarse` entries)

- [ ] **Step 1: Run** `make queue PHASE=tune-coarse METHOD=ppo` (resumable; stop and restart at will); `make status PHASE=tune-coarse` shows 28 finished.
- [ ] **Step 2: Run** `uv run python -m wut_murl_craftax.evaluation.reproduction --method ppo --budget 1m --phase tune-coarse`; pass: continue; fail: work the debug order in the diagram, one step at a time, 3 seeds per diagnostic run (MLflow experiment `tune-coarse`, config names `debug-<step>`), and note each result in [[PPO design]].
- [ ] **Step 3: Run** the debug step 5 command only if needed: `WANDB_MODE=offline uv run --isolated --no-project --python 3.12 --with craftax==1.6.1 --with "jax==0.11.*" --with flax --with optax --with distrax --with wandb python ppo.py --total_timesteps 1e6 --num_envs 256 --num_steps 16 --lr 3e-4 --seed <s>` inside a checkout of Craftax_Baselines at 7ce36fa (outside this repo).

### Task 6: Stage 2 and the tuned control config

**Files:**
- Modify: `configs/methods/ppo.yaml` (`grid.fine` expanded, `tuned`), `configs/phases/tune-fine.yaml`

- [ ] **Step 1: Generate** the 4 fine configs from the rule: `uv run python -m wut_murl_craftax.orchestration.queue --expand-fine ppo` (writes the entries into `ppo.yaml`, reviewed in the diff).
- [ ] **Step 2: Run** `make queue PHASE=tune-fine METHOD=ppo`; 12 finished.
- [ ] **Step 3: Apply** the selection rule (Hyperparameter grid) and write `tuned` into `ppo.yaml`; add a dated section to [[Experiment decisions]] with the 3-seed means, h and the choice.

### Task 7: PPO metrics logged per update

**Files:**
- Modify: `src/wut_murl_craftax/training/ppo.py`
- Test: `tests/training/test_ppo.py`

- [ ] **Step 1: Write the failing test**: one `train_chunk` returns `ppo/policy_loss`, `ppo/value_loss`, `ppo/entropy`, `ppo/approx_kl`, `ppo/clip_frac`, `ppo/explained_variance`, `ppo/grad_norm`, `ppo/adv_mean`, `ppo/adv_std`, `ppo/lr`, all finite.
- [ ] **Step 2: Implement** the names on top of the shared metrics of [[2026-10-20-craftax-evaluation-and-metrics]].

### Task 8: Finals at 1M and 1B

**Files:**
- Modify: `configs/phases/final-1m.yaml`, `configs/phases/final-1b.yaml` (add `ppo`: `tuned` at 1M; Table 7 at 1B)

- [ ] **Step 1: Run** `make queue PHASE=final-1m METHOD=ppo`; 10 finished.
- [ ] **Step 2: Run** `make queue PHASE=final-1b METHOD=ppo` (laptop or `scripts/cluster/run_array.sh`; merge with `make merge-mlflow SRC=<store>`); 10 finished.
- [ ] **Step 3: Run** `uv run python -m wut_murl_craftax.evaluation.reproduction --method ppo --budget 1b --phase final-1b`; a fail is reported in [[2026-10-20-craftax-comparison-and-results]] with the band, not rerun.

### Task 9: Analysis hooks and stop criteria

**Files:**
- Modify: `src/wut_murl_craftax/orchestration/runner.py` (abort rules per method), `configs/methods/ppo.yaml` (`abort:` block)
- Test: `tests/orchestration/test_runner.py`

- [ ] **Step 1: Write the failing test**: a fake run with a NaN `ppo/value_loss` at update 3 ends `FAILED` with its last checkpoint kept; a run with `ppo/approx_kl` > 0.1 for 3 updates in a row is flagged, not stopped.
- [ ] **Step 2: Implement** the rules: abort on any NaN or inf in a logged loss; rerun once from the last checkpoint with the same seed, and a second NaN marks a bug (open an issue note); flags (tags `flag/*`) for approx KL > 0.1 for 3 updates, entropy < 0.2 nats before 10% of the budget, explained variance < 0 after 20% of the budget; no early stopping on score.
- [ ] **Step 3: Hand** to [[2026-10-20-craftax-training-analysis]] the obstacles to watch for PPO: plateau at 1M (the paper: only basic achievements at 1M), entropy collapse, value failure, skill forgetting of `defeat_zombie` at 1B, exploration failure (rare dungeon entry), survival failure.
- [ ] **Step 4: Run** `make tests` and `uv run python scripts/check_vault.py`.

## Risks

- Craftax 1.6.1 vs the 2024 version behind the scoreboard: reward-affecting errata (v1.5.0 `eat_plant`, v1.6.0 Classic health) may shift scores; debug step 5 separates code from environment ([[Craftax Baselines reproduction notes]]).
- Optimistic resets hand duplicate start worlds to envs when more envs finish in one step than worlds are generated (16 per step at 256 envs); the paper reports "essentially no effect" on learning, and debug step 4 (a comparison against auto-reset envs) checks it on our code. The vendored copy is taken at 7ce36fa, which includes the fix for Baselines issue #11 (fewer finished envs than resets).
- Metal learner numerics differ from CUDA float32; debug step 3 runs the learner on CPU ([[Craftax on Apple GPU]]).
- The published number has no spread; s_pub comes from third-party reruns with other protocols, so the band is an estimate.
- Third-party PPO 1M spread is not published; the PPO-RNN 1M spread is the proxy.
- The 1B table at 1M gives only 15 updates; a poor score there is expected and informative, not a bug.
