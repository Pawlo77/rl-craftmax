---
title: "Craftax EDEN 1B execution implementation plan"
type: plan
domain: orchestration
parent: "[[2026-10-20-craftax-project]]"
status: in progress
created: 2026-10-31
updated: 2026-10-31
tags: [plan, eden, compute, 1b]
---
# Craftax EDEN 1B execution implementation plan

Status: in progress

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

- [x] **Step 1:** (done 2026-10-31, `ec56d91`) record the three update lines of `check_recompute.py` above in the Device verification table of [[PPO-RNN design]] and remove `ppo/ratio_first_mb` from `unmeasured_limits` in `configs/profiles/cuda.yaml` (exit 0 rule of [[2026-10-20-craftax-infrastructure]] Task 6 Step 5).
- [x] **Step 2:** `combo-1b` waits for `select_pair --rank-sum-with final-1m` after `final-1b` ([[2026-10-20-craftax-project]] Step 1b); do not submit it first.
- [x] **Step 3:** (done 2026-10-31) `git clone https://github.com/Pawlo77/rl-craftmax.git ~/craftax-eden-1b` and `git checkout <pushed commit>`: a clone, not a `git archive`, so each run's reproducibility report records a clean commit. Then `uv sync --frozen --all-groups` once. `run_array.sh` calls `uv run`, which installs only the default groups: the first launch skipped this and every run stopped in seconds with `JAX has no GPU backend (platforms ['cpu'])`. Submit with `UV_NO_SYNC=1` so concurrent jobs do not race on the environment.
- [x] **Step 4:** (done) cap of 12 concurrent A100s (`%2` on ppo, ppo-rnn, pqn-rnn and count-bonus-ppo-rnn, `%1` on the other four) and the time limit per entry: `short` allows 24 h, a run is 4 to 6 h of training plus evaluation, so `--time=14:00:00` (`--time=18:00:00` for Dyna-Q and SIL); `--signal=B:TERM@600` resumes what the limit cuts. `sfree` showed free GPUs but few free CPUs: request `--cpus-per-task=2` (measured: PPO-RNN on 2 CPUs trains at 61,343 steps/s, the 8-CPU probe's rate).

### Task 2: Submit

- [x] **Step 1:** (done 2026-10-31, jobs 1936132 and 1936134 to 1936140) one submission per entry, 3 seeds each, from `~/craftax-eden-1b`; `<group>` is the research account (`sacctmgr show assoc user=$USER` lists it):

```
STORE_DIR=node-stores/ppo sbatch -A <group> -p short --gres=gpu:a100:1 --array=0-2%2 --cpus-per-task=2 --mem=8G -J cx-ppo --time=14:00:00 scripts/cluster/run_array.sh final-1b --method ppo
```

  The other 7 entries are the same line with their `--method`, `STORE_DIR`, cap and memory (12 GB ppo-rnn and count-bonus-ppo-rnn, 16 GB pqn-rnn, count-bonus-pqn-rnn, sil and smdp-skills, 24 GB dyna-q); `sbatch --test-only` first shows the planned node and start.
- [x] **Step 2:** start without `SEED_FROM` (no 1B run exists), as the compute-budget Task 2 Step 1 says.

### Task 3: Watch

- [ ] **Step 1:** `squeue -u $USER`, `sacct -j <id> -o JobID,State,Elapsed,MaxRSS`, and `make status PHASE=final-1b` per `STORE_DIR` through a node or after a copy back.
- [ ] **Step 2:** once a guarded entry (Dyna-Q, SMDP, count-bonus) passes 5 to 10 % of its budget run `scripts/cluster/project_guards.py` ([[2026-10-20-craftax-compute-budget]] Task 2 Step 1b); a resize restarts all 3 seeds of the entry.
- [ ] **Step 3:** time the first checkpoint evaluation and the longest 1B episode from the first finished run and replace the 0.43 h per run of the compute budget; resize the time limits if they miss by more than 1.5 x.

### Task 4: Copy back

- [ ] **Step 1:** only finished runs, as in [[2026-10-20-craftax-compute-budget]] Task 2 Step 2: `rsync -a --exclude state/ --exclude '*.tmp' --exclude owner.json eden:craftax-eden/node-stores/ node-stores/`, then `make merge-mlflow SRC=node-stores/<entry>/mlruns-node-<i>.db` per node.
- [ ] **Step 2:** replace the estimates of [[2026-10-20-craftax-compute-budget]] with the table above and the measured evaluation times; mark this plan `done` when `make status PHASE=final-1b` shows 24 finished and `combo-1b` 3.

### Launch note (2026-10-31)

- First launch failed in seconds on every run (no `cuda` group in the venv, see Task 1 Step 3); nothing trained, stores and logs were cleared.
- Second launch (jobs 1936132 and 1936134 to 1936140): PPO seeds 0 and 1 train on `dgx-4` at 79,100 steps/s steady (probe 78,200; 3.5 h of training), 8.2 GB device memory, 95 % GPU utilisation.
- The other arrays wait: other users' `long` jobs held 126 of 128 CPUs on `dgx-2` and `dgx-3` (5 free A100s each, stranded) and `dgx-4` had 11 GB of free RAM. Lowering `--mem` to the measured host peaks plus margin (8 GB for the PPO family, 10 GB for pqn-rnn, count-bonus-pqn-rnn, sil and smdp-skills, 16 GB for dyna-q; `scontrol update JobId=<array> MinMemoryNode=<MB>`) let seed 1 start at once. The scheduler's own estimates for the rest ranged from hours to days: check `squeue --start`.

- Two CPUs per job let the arrays use the A100s other users' CPU-only jobs strand (126 of 128 CPUs allocated, 5 GPUs free): PPO-RNN seeds started on `dgx-2` and `dgx-3` within seconds. The arrays of the other six entries were resubmitted with `--cpus-per-task=2` (jobs 1936153 and 1936158 to 1936163).
- A task on `dgx-2` sat 11 minutes with no child process, 0 % GPU and no database (node load 29, probably I/O): `scontrol requeue <jobid>` restarted it. Check `nvidia-smi` through `srun --jobid=<id> --overlap` when a log stays empty.
- A store records its array size: never resubmit single seeds of an array under a different size.

### Risks

- Fair share: the group's recent usage decides how soon 12 GPUs start; the dry run (`sbatch --test-only`) estimated hours when the cluster had free A100s, so check `squeue --start` after submitting.
- Preemption and the 24 h `short` limit: runs resume from `state/` when the same command is submitted again.
- The 1B evaluation (26 passes) is unmeasured; it may add up to 1 h per run.
