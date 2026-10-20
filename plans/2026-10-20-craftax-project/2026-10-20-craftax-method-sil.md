---
title: "Self-Imitation Learning implementation plan"
type: plan
domain: training
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, method, self-imitation, replay, ppo-rnn]
---
# Self-Imitation Learning implementation plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add self-imitation of better-than-expected actions from a per-achievement replay buffer to PPO-RNN, tune it at Craftax-1M, and run it at 1M and 1B against our PPO-RNN control, with a one-buffer ablation.

**Architecture:** `training/sil.py` keeps a per-env staging ring of the last 128 steps and a buffer of windows, one FIFO slot per achievement; after each rollout it copies the window that ends at every new unlock into that achievement's slot; after the PPO epochs the trainer runs `K` optimizer steps on the SIL loss through the `extra_loss_fn` hook of `PPORNNConfig` ([[2026-10-20-craftax-method-ppo-rnn]]). Registry entry `sil`. Design decisions: [[Self-Imitation Learning design]].

**Tech Stack:** JAX 0.11, craftax 1.6.1, flax, optax, MLflow, the runner and queue of [[2026-10-20-craftax-infrastructure]].

---

## Design

- Unit of replay: a 128-step window ending at a new achievement unlock, with returns truncated at the unlock and the GRU state at the window start.
- Layout: 67 slots (one per achievement) of `S` windows each, FIFO inside a slot; sampling uniform over non-empty slots, then by priority $(R - V)_+$.
- Loss: $\lambda_{\text{sil}} [-\log\pi(a|s)(R - V)_+ + 0.01 \cdot \tfrac12 (R - V)_+^2]$ on 32 windows, `K` steps per PPO update, same optimizer.
- Memory: 8,432 B per stored step (`uint8` map, `float32` tail); staging 1.21 MB per env; buffer 145 MB to 1.16 GB over the grid.
- Ablation at `final-1m`: one shared buffer of the same capacity (`layout: single`).
- Decisions, memory table and loss details: [[Self-Imitation Learning design]].

```mermaid
flowchart LR
  subgraph PPO["PPO-RNN update (unchanged)"]
    RO["rollout"] --> GAE["GAE"] --> EP["4 epochs x 8 minibatches"]
  end
  subgraph SILB["SIL (sil.py)"]
    STG["staging ring, 128 steps per env"] --> INS["insert window at each new unlock"]
    INS --> SLOT["67 achievement slots, FIFO"]
    SLOT --> SAMP["32 windows by priority"]
    SAMP --> LS["SIL loss where R > V"]
  end
  RO --> STG
  EP --> LS
  LS -->|"K optimizer steps"| NEXT["next rollout"]
```

## Hyperparameter grid

Stage 1 (coarse, Craftax-1M, 3 seeds):

| Knob | Values | Why |
|---|---|---|
| SIL loss weight $\lambda_{\text{sil}}$ | 0.1, 0.3, 1.0 | 1.0 is the paper's unweighted loss; smaller keeps the off-policy step small (candidate note risk) |
| Windows per slot `S` | 2, 8 | 145 MB and 579 MB buffers; few recent successes against more varied ones |
| SIL steps per PPO update `K` | 1, 2, 4 | the candidate note's 1-2 plus one stronger setting; a 1M update has 32 PPO minibatch steps |

- Configs: 3 x 2 x 3 = 18; 54 runs.

Stage 2 (fine, 3 seeds):

- Knobs: the two of $\lambda_{\text{sil}}$, `S`, `K` whose stage-1 config means spread most.
- Values: best stage-1 value x {0.5, 1, 2}, rounded to integers >= 1 for `S` and `K` (largest point `S` = 16, 1.16 GB).
- Configs: 3 x 3 = 9, the centre reused by run name; 8 new configs, 24 runs.
- Selection: highest mean final score over 3 seeds ([[2026-10-20-craftax-evaluation-and-metrics]]); ties go to the smaller buffer.

Fixed settings, PPO-RNN part (the tuned PPO-RNN control; Craftax paper Table 11 at 1M, Table 7 at 1B, [[Craftax Baselines algorithms and hyperparameters]]):

| Setting | 1M | 1B |
|---|---|---|
| Envs x steps per rollout | 256 x 16 | 1024 x 64 |
| Learning rate (linear anneal) | 3e-4 | 2e-4 |
| Epochs / minibatches | 4 / 8 | 4 / 8 |
| γ / GAE λ | 0.99 / 0.8 | 0.99 / 0.8 |
| Clip / value coef / entropy | 0.2 / 0.5 / 0.01 | 0.2 / 0.5 / 0.01 |
| Max grad norm | 1.0 | 1.0 |
| Network | Dense 512, GRU 512, 2 x Dense 512 heads (ReLU) | same |

Fixed SIL settings:

| Setting | Value |
|---|---|
| Window length $L_w$ | 128 steps |
| Value-loss weight $\beta_v$ | 0.01 |
| SIL batch | 32 windows (4,096 steps) |
| Return | discounted with the control γ, truncated at the unlock, no bootstrap |
| Priority | mean $(R - V)_+$ + 0.01, exponent 0.6, new windows at max priority |
| Insertion cap | 64 windows per update |
| Layout | per-achievement slots (ablation: single buffer) |
| Storage | map `uint8`, tail `float32`, staging GRU state `float16` |
| 1B finals | best 1M SIL knobs on the 1B PPO-RNN settings |

## Runs and laptop time estimate

- Throughput: PPO-RNN 2,400 steps/s (shared assumption), Craftax-Classic RNN 8,000 steps/s (0.8 x 10,000).
- Overhead factor 1.3 (estimate): one SIL step is a GRU forward and backward pass over 4,096 steps, about a quarter of the 4 PPO epochs; with learning at roughly 40% of an update, `K` = 1, 2, 4 cost about +10%, +20%, +40%, plus about 5% for staging and insertion; 1.3 is the mean over the stage-1 grid.
- Method throughput: 1,846 steps/s on Craftax, 6,154 steps/s on Craftax-Classic; +1 min compile per run.

| Phase | Configs | Seeds | Steps per run | Runs | Est. steps/s | Est. hours (laptop, sequential) |
|---|---|---|---|---|---|---|
| `smoke` (Classic) | 2 | 1 | 200k | 2 | 6,154 | 0.05 |
| `tune-coarse` | 18 | 3 | 1M | 54 | 1,846 | 9.02 |
| `tune-fine` | 8 new | 3 | 1M | 24 | 1,846 | 4.01 |
| `final-1m` (best + single-buffer ablation) | 2 | 10 | 1M | 20 | 1,846 | 3.34 |
| `final-1b` | 1 | 10 | 1B | 10 | 1,846 | 1,504.8 |
| Total | | | | 110 | | 1,521.2 |

- One 1M run: 0.167 h; one 1B run: 150.5 h.
- 1B on a CUDA cluster GPU (authors' hardware, estimate): 266,961 x 0.8 (RNN) / 1.3 = 164,284 steps/s, 1.71 h per run, 17.1 h for 10 seeds.
- Peak extra memory per run: 1M up to 1.47 GB (staging 310 MB + `S` = 16 buffer); 1B up to 2.40 GB.
- Measured throughput replaces these numbers in [[2026-10-20-craftax-compute-budget]] after the smoke phase.

### Task 1: Window buffer

**Files:**
- Create: `src/wut_murl_craftax/training/sil.py` (buffer part)
- Test: `tests/training/test_sil_buffer.py`

- [ ] **Step 1: Write the failing tests**: `compress_obs` / `decompress_obs` round-trip a real Craftax and a real Craftax-Classic observation exactly; a scripted unlock at step 130 of env 1 copies steps 3-130 of that env into the slot of that achievement; steps before a done inside the window get mask 0; returns of a window with rewards `[0, 0, 1]` and γ 0.9 are `[0.81, 0.9, 1.0]`; a slot with `S` = 2 keeps the two newest windows (FIFO); 70 unlocks in one update insert 64 and report 6 overflowed; `buffer_bytes(cfg)` equals the memory table of [[Self-Imitation Learning design]] for 256 and 1024 envs.
- [ ] **Step 2: Implement** `SILConfig` (lambda_sil, slots_windows `S`, sil_steps `K`, window 128, beta_v, batch 32, alpha 0.6, insert_cap 64, layout `slots|single`), `init(cfg, n_envs, n_ach) -> SILState`, `observe(state, traj) -> state` (staging write once per rollout, unlock detection from the inner `EnvState.achievements`, window copy, returns), all `jit`-compatible with static shapes.
- [ ] **Step 3: Run** `uv run pytest tests/training/test_sil_buffer.py -q`.

### Task 2: Sampling and loss

**Files:**
- Modify: `src/wut_murl_craftax/training/sil.py` (sampling and loss)
- Test: `tests/training/test_sil_loss.py`

- [ ] **Step 1: Write the failing tests**: with slots holding 1 and 3 windows, slot frequencies over 10,000 draws are 0.5 / 0.5 within 0.02 (slots layout) and 0.25 / 0.75 at equal priority (single layout); window frequencies inside a slot follow priority^0.6 within 0.02; where $V > R$ on every step the loss and its gradient are 0; on a 2-step example (log-probs, values, returns given) the loss equals the hand-computed value; replaying a window from its stored GRU state gives the same logits as the original rollout when the parameters are unchanged (tolerance from the `float16` state); sampled priorities are rewritten after a step.
- [ ] **Step 2: Implement** `sample(state, key) -> windows`, `sil_loss(params, windows) -> (loss, aux)` and `extra_loss(params, state, key) -> (loss, (state, metrics))` matching the `extra_loss_fn` contract of [[2026-10-20-craftax-method-ppo-rnn]].
- [ ] **Step 3: Run** `uv run pytest tests/training/test_sil_loss.py -q`.

### Task 3: Trainer registry and config

**Files:**
- Modify: `src/wut_murl_craftax/training/interface.py` (entry `sil`)
- Create: `configs/methods/sil.yaml` (base config, `grid_stage1`, `grid_stage2` rule, ablation `layout: single`)
- Test: `tests/training/test_sil_trainer.py`

- [ ] **Step 1: Write the failing tests**: with $\lambda_{\text{sil}}$ = 0 the parameters after 2 updates on Craftax-Classic equal the PPO-RNN control's bit for bit (same seed); with an empty buffer the SIL step is skipped and logs `sil/active = 0`; the buffer and staging ring are in the checkpoint and a resumed run continues with the same buffer contents; the entry passes the interface test of [[2026-10-20-craftax-infrastructure]] Task 3.
- [ ] **Step 2: Implement** the factory: `PPORNNConfig(extra_loss_fn=sil.extra_loss, extra_steps=K)` with `sil.observe` called on each rollout (the trainer hook named in [[2026-10-20-craftax-method-ppo-rnn]]), SIL state in the train state.
- [ ] **Step 3: Write** `configs/methods/sil.yaml`; `uv run python -m wut_murl_craftax.orchestration.queue --phase configs/phases/tune-coarse.yaml --method sil --dry-run` lists 54 runs.
- [ ] **Step 4: Run** `make tests`.

### Task 4: Method metrics

**Files:**
- Modify: `src/wut_murl_craftax/training/sil.py`
- Test: `tests/training/test_sil_loss.py`

- [ ] **Step 1: Write the failing test**: one update logs every metric below; slot fill never exceeds `S`.
- [ ] **Step 2: Implement** per-update logging beyond [[2026-10-20-craftax-evaluation-and-metrics]]: `sil/slot_fill/<achievement>`, `sil/nonempty_slots`, `sil/inserted`, `sil/insert_overflow`, `sil/active` (share of sampled steps with $R > V$), `sil/adv_pos_mean`, `sil/policy_loss`, `sil/value_loss`, `sil/grad_norm`, `sil/kl_after` (KL between the policy before and after the SIL steps on the latest PPO batch), `sil/sample_age` (updates since insertion, mean and max), `sil/sample_share/<achievement>`, `sil/value_bias` (mean $V - R$ on sampled steps); per run `sil/buffer_bytes` and `sil/staging_bytes`.
- [ ] **Step 3: Run** `uv run pytest tests/training/test_sil_loss.py -q`.

### Task 5: Smoke run

**Files:**
- Modify: `configs/phases/smoke.yaml` (sil: config A slots, `S` 2, `K` 1, $\lambda$ 0.3; config B single, `S` 2, `K` 2, $\lambda$ 1.0; 1 seed, 200k steps, Craftax-Classic)

- [ ] **Step 1: Run** `make queue PHASE=smoke METHOD=sil`; expected: 2 runs finish, `sil/nonempty_slots` > 5, `sil/active` > 0 early, no NaN.
- [ ] **Step 2: Interrupt** one run and rerun; expected: it resumes with the same `sil/slot_fill` values.
- [ ] **Step 3: Record** measured steps/s and peak memory in [[2026-10-20-craftax-compute-budget]].

### Task 6: Coarse tuning

- [ ] **Step 1: Run** `make queue PHASE=tune-coarse METHOD=sil` (54 runs, resumable).
- [ ] **Step 2: Check** `make status PHASE=tune-coarse`; expected: 54 finished for `sil`.
- [ ] **Step 3: Write** the stage-1 table (config, mean final score, sd, `sil/active` at 1M, non-empty slots) into this plan under a new section `## Results`, and pick the two stage-2 knobs by the rule above.

### Task 7: Fine tuning

- [ ] **Step 1: Fill** `grid_stage2` in `configs/methods/sil.yaml` from Task 6.
- [ ] **Step 2: Run** `make queue PHASE=tune-fine METHOD=sil` (24 runs).
- [ ] **Step 3: Tag** the best config `best=true` in MLflow and record it in `## Results` and [[Experiment decisions]].

### Task 8: Final 1M

- [ ] **Step 1: Run** `make queue PHASE=final-1m METHOD=sil` (best config x 10 seeds and the same config with `layout: single` x the same 10 seeds: 20 runs).
- [ ] **Step 2: Check** that the PPO-RNN control final exists with the same seeds; comparisons and statistics follow [[2026-10-20-craftax-comparison-and-results]].

### Task 9: Final 1B

- [ ] **Step 1: Run** `make queue PHASE=final-1b METHOD=sil` (best 1M knobs on the 1B PPO-RNN settings, 10 seeds; laptop or `scripts/cluster/run_array.sh`).
- [ ] **Step 2: Merge** node stores with `make merge-mlflow SRC=<store>`; `make status PHASE=final-1b` shows 10 finished for `sil`.

### Task 10: Analysis hooks and abort rules

**Files:**
- Modify: `src/wut_murl_craftax/training/sil.py` (guards), `src/wut_murl_craftax/analysis/obstacles.py` (method detectors, from [[2026-10-20-craftax-training-analysis]])
- Test: `tests/training/test_sil_trainer.py`, `tests/analysis/test_obstacles.py`

- [ ] **Step 1: Write the failing tests**: a NaN SIL loss aborts the run with status `FAILED` and `abort=nan`; a planned buffer above the profile's memory limit refuses to start with `abort=memory` before compiling; each detector below fires on a synthetic series and stays silent on a clean one.
- [ ] **Step 2: Implement** the abort rules: NaN or inf in loss or parameters; `buffer_bytes + staging_bytes` above `memory_limit_bytes` of the machine profile (checked at init); `sil/kl_after` > 0.5 for 20 consecutive updates (the SIL step dominates the policy).
- [ ] **Step 3: Implement** the detectors for [[2026-10-20-craftax-training-analysis]]:
  - Overfitting to early lucky windows: `sil/sample_age` max above 50% of the updates so far while `sil/active` > 0.5 on that slot, and entropy below the control's at the same step.
  - Forgetting: `DEFEAT_ZOMBIE` and other basic success rates falling below the control's by more than 10 points.
  - Imitated unsafe play: a cause-of-death share rising above the control's by more than 10 points after SIL starts.
  - Value overestimation: `sil/value_bias` > 0 and explained variance of the PPO critic below the control's.
  - Dead SIL: `sil/active` < 0.01 for 20% of the budget (the critic caught up; SIL is a no-op).
  - Insertion overflow in > 10% of updates (cap too small for the env count).
- [ ] **Step 4: Run** `make tests`.

## Risks

- Memory: the buffer is computed and checked at init, stored compressed, and the staging ring is written once per rollout; the grid tops at 2.4 GB at 1B.
- Overfitting to early lucky episodes: FIFO slots replace old windows when the achievement repeats; for windows that never repeat, $(R - V)_+$ falls to 0 as the critic learns; the age detector flags the rest; mitigation if it fires: an age cap per window, logged in [[Experiment decisions]].
- Off-policy drift inside PPO: $\lambda_{\text{sil}}$ and `K` are tuned, `sil/kl_after` is logged and bounded by an abort rule.
- Stale GRU start states: replay from stored states without burn-in; the replay test bounds the error for unchanged parameters; burn-in is the fallback.
- Truncated returns ignore rewards after the unlock, so SIL fires only when the unlock beats the critic's expectation; this makes it conservative, not wrong.
- Hook mismatch with the base plan: SIL needs `extra_loss_fn`, a number of extra optimizer steps per update and a per-rollout `observe` call; Task 3 tests the contract, and any change in [[2026-10-20-craftax-method-ppo-rnn]] updates this plan in the same change.
- 1M budget: about 250 updates; SIL needs unlocks to start, so early updates have few windows; `sil/nonempty_slots` over time shows how much of the budget SIL was active.
