---
title: PLGrid Slurm
type: handbook
domain: plgrid
tags: [handbook, plgrid, slurm]
created: 2026-10-11
updated: 2026-10-11
parent: "[[PLGrid compute]]"
---
# PLGrid Slurm

## How jobs run

- Users log in to a shared login node, write a batch script that names the resources, and submit it with `sbatch`. The Slurm queue starts it on compute nodes when resources are free; the job then has exclusive access to what it requested until it ends or hits the time limit.
- A job waits as *pending* and then runs. To use several cores or GPUs the program itself must be parallel (threads, MPI, OpenMP, CUDA).
- The grant to charge goes in `-A`, and its name carries a suffix for the resource type.

## Account suffixes

| Cluster | Suffix |
|---------|--------|
| Ares CPU, CPU big memory, GPU | `-cpu`, `-cpu-bigmem`, `-gpu` |
| Athena GPU | `-gpu-a100` |
| Helios GPU | `-gpu-gh200` |

So the account for grant `<grant>` on the Helios GPU partition is `<grant>-gpu-gh200`.

## Partitions

| Cluster | Partition | Limit | Access |
|---------|-----------|-------|--------|
| Helios | `plgrid` | 72 h | CPU, open |
| Helios | `plgrid-long` | 168 h | CPU, grant with a 168 h maximum runtime |
| Helios | `plgrid-bigmem` | 72 h | CPU, grant with big-memory resources |
| Helios | `plgrid-gpu-gh200` | 48 h | Grant with GPGPU resources |
| Athena | `plgrid-gpu-a100` | 48 h | GPU |
| Ares | `plgrid` / `plgrid-testing` / `plgrid-now` / `plgrid-long` | 72 h / 1 h / 12 h / 168 h | CPU; testing is high priority and limited to 3 running jobs, now to 1 job |
| Ares | `plgrid-gpu-v100` | 48 h | Grant with GPGPU resources |
| Lem | `plgrid-short` / `plgrid` / `plgrid-long` | 24 h / 72 h / 168 h | CPU and GPU |
| Tryton Plus | `plgrid-testing` / `plgrid` / `plgrid-long` | 1 h / 72 h / 168 h | CPU |

## Batch and interactive

A batch script starts with `#!/bin/bash -l` and `#SBATCH` lines: job name `-J`, nodes `-N`, tasks per node `--ntasks-per-node`, memory `--mem-per-cpu` (the default is 5 GB per core), `--time=HH:MM:SS`, account `-A`, partition `-p`, and `--output` and `--error` files. It then changes to `$SLURM_SUBMIT_DIR`, loads modules ([[PLGrid software]]) and runs the program. Submit with `sbatch script.sh`.

An interactive shell: `srun -p <partition> -N 1 --ntasks-per-node=1 -n 1 -A <account> --pty /bin/bash -l`.

## Waiting time

- `sbatch --test-only script.sh` checks the script and prints a pessimistic start estimate; `squeue -t pd -o %i,%S` shows the estimate of already scheduled jobs. Estimates are not guarantees.
- Priority depends on the resources requested, the declared time, the recent activity of the user and other grant members, and the allocation in the grant. Requesting less time and fewer resources usually starts a job sooner.
