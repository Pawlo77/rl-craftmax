---
title: "Craftax EDEN 1B execution implementation plan"
type: plan
domain: orchestration
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-31
updated: 2026-10-31
tags: [plan, eden, compute, 1b]
---
# Craftax EDEN 1B execution implementation plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `final-1b` and `combo-1b` on the EDEN cluster ([[EDEN compute]]) with the cluster scripts already in the repository, from measurements taken there on 2026-10-31.

**Architecture:** One Slurm array per method entry, one A100 per run, through `scripts/cluster/run_array.sh` ([[2026-10-20-craftax-compute-budget]] Request lines); only the EDEN flags differ (`-A <group>`, partition `short`, an A100 gres). Real account, user and host names stay out of the vault.

**Tech Stack:** Slurm on EDEN, uv, JAX CUDA wheels, MLflow SQLite node stores.

---
## Measured on EDEN (2026-10-31, one A100 40 GB on `short`)

| Fact | Value |
|---|---|
| Job start from submit | seconds (queue nearly empty; H100 and H200 queues were days) |
| `uv sync --frozen --all-groups` on a node | 195 s; JAX sees `CudaDevice(id=0)` |
| `check_recompute.py` (1B config, 3 updates) | exit 0; worst gap 5.5e-5 (bound 1e-4, guard 1e-3) |
| 1M `ppo-rnn` run, 256 envs | 332 s in all; training 166 s at 16,800 steps/s; peak device 0.97 GB |
| `smoke` phase, 16 runs | all finished in 21 min |

Throughput at the 1B shape (1,024 envs x 64 steps, 7 minutes per entry, killed by SIGTERM at a chunk end):

| Entry | steps/s | h per run (training) | peak device GB | peak host GB |
|---|---|---|---|---|
| ppo | 78,200 | 3.6 | 8.4 | 4.5 |
| ppo-rnn | 61,400 | 4.5 | 6.6 | 3.8 |
| pqn-rnn | 62,900 | 4.4 | 12.6 | 5.6 |
| count-bonus-ppo-rnn | 59,800 | 4.6 | 7.0 | 4.6 |
| count-bonus-pqn-rnn | 59,600 | 4.7 | 12.6 | 5.6 |
| sil | 56,700 | 4.9 | 11.9 | 6.1 |
| smdp-skills | 61,900 | 4.5 | 12.2 | 5.2 |
| dyna-q | 50,300 | 5.5 | 18.3 | 11.5 |

- Training takes about 110 GPU hours for the 24 `final-1b` runs plus about 15 for `combo-1b`: 3.3 times the 33 and 5.1 of the compute budget, which assumed 267,000 steps/s. Evaluation is not measured at 1B (Task 3 Step 3).
- Every entry fits a 40 GB A100 with room (Dyna-Q 18.3 of 40 GB). Host RAM peaks at 11.5 GB (Dyna-Q), so `--mem=32G` holds for all.
- The A100 drew 28% of its power in the 1M run: the 1B shape fills it better, but nothing was measured; do not plan two jobs per GPU.

Cluster facts used by the plan ([[EDEN Slurm basics and partitions]], [[EDEN GPU jobs]]):

- `short` allows 24 h, `long` 5 days; both reach the DGX A100 nodes (4 x 8 GPUs). H100 (`hopper`, 8) and H200 (`hopper-2`) were not reachable inside a day.
- A job needs `-A <group>`; without it submission fails with `AssocMaxSubmitJobLimit`. No per-user GPU limit showed in the QOS table.
- The login node cannot import numpy from the project venv (its CPU lacks x86-64-v2): read or merge stores on a compute node or on the laptop after the copy back.
- Home has no quota (22 GB used by two test jobs); a run adds 0.3 to 0.65 GB and a live `state/` up to 6.4 GB (Dyna-Q).

### Task 1: Before the first array

**Files:**
- Modify: `configs/profiles/cuda.yaml` (`unmeasured_limits`), [[PPO-RNN design]] (Device verification table)

- [ ] **Step 1:** record the three update lines of `check_recompute.py` above in the Device verification table of [[PPO-RNN design]] and remove `ppo/ratio_first_mb` from `unmeasured_limits` in `configs/profiles/cuda.yaml` (exit 0 rule of [[2026-10-20-craftax-infrastructure]] Task 6 Step 5).
- [ ] **Step 2:** `combo-1b` waits for `select_pair --rank-sum-with final-1m` after `final-1b` ([[2026-10-20-craftax-project]] Step 1b); do not submit it first.
- [ ] **Step 3:** copy the code to the cluster (`git archive HEAD | ssh eden 'mkdir -p ~/craftax-eden && tar -x -C ~/craftax-eden'`) and keep the uv cache warm: the first node pays the 195 s sync, later nodes less.
- [ ] **Step 4:** fix the cap on concurrent GPUs (suggested 12 of the 32 A100s; `sfree` first) and the time limit per entry: `short` allows 24 h, a run is 4 to 6 h of training plus evaluation, so request `--time=14:00:00` (`--time=18:00:00` for Dyna-Q and SIL) and let `--signal=B:TERM@600` resume what the limit cuts.

### Task 2: Submit

- [ ] **Step 1:** one submission per entry, 3 seeds each, from `~/craftax-eden`; `<group>` is the research account (`sacctmgr show assoc user=$USER` lists it):

```
STORE_DIR=node-stores/ppo sbatch -A <group> -p short --gres=gpu:a100:1 --array=0-2%3 --cpus-per-task=8 --mem=32G --time=14:00:00 scripts/cluster/run_array.sh final-1b --method ppo
```

  The other 7 entries are the same line with their `--method` and `STORE_DIR`; `sbatch --test-only` first shows the planned node and start.
- [ ] **Step 2:** start without `SEED_FROM` (no 1B run exists), as the compute-budget Task 2 Step 1 says.

### Task 3: Watch

- [ ] **Step 1:** `squeue -u $USER`, `sacct -j <id> -o JobID,State,Elapsed,MaxRSS`, and `make status PHASE=final-1b` per `STORE_DIR` through a node or after a copy back.
- [ ] **Step 2:** once a guarded entry (Dyna-Q, SMDP, count-bonus) passes 5 to 10 % of its budget run `scripts/cluster/project_guards.py` ([[2026-10-20-craftax-compute-budget]] Task 2 Step 1b); a resize restarts all 3 seeds of the entry.
- [ ] **Step 3:** time the first checkpoint evaluation and the longest 1B episode from the first finished run and replace the 0.43 h per run of the compute budget; resize the time limits if they miss by more than 1.5 x.

### Task 4: Copy back

- [ ] **Step 1:** only finished runs, as in [[2026-10-20-craftax-compute-budget]] Task 2 Step 2: `rsync -a --exclude state/ --exclude '*.tmp' --exclude owner.json eden:craftax-eden/node-stores/ node-stores/`, then `make merge-mlflow SRC=node-stores/<entry>/mlruns-node-<i>.db` per node.
- [ ] **Step 2:** replace the estimates of [[2026-10-20-craftax-compute-budget]] with the table above and the measured evaluation times; mark this plan `done` when `make status PHASE=final-1b` shows 24 finished and `combo-1b` 3.

### Risks

- Fair share: the group's recent usage decides how soon 12 GPUs start; the dry run (`sbatch --test-only`) estimated hours when the cluster had free A100s, so check `squeue --start` after submitting.
- Preemption and the 24 h `short` limit: runs resume from `state/` when the same command is submitted again.
- The 1B evaluation (26 passes) is unmeasured; it may add up to 1 h per run.
