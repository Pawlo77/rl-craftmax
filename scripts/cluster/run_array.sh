#!/usr/bin/env bash
# One node of a job array: runs every N-th run of a phase on a CUDA profile and writes
# its own MLflow store, merged into the main one later with `make merge-mlflow`.
#
# Slurm:  sbatch --array=0-3 --gres=gpu:1 scripts/cluster/run_array.sh tune-coarse
# Other schedulers: set ARRAY_INDEX (0-based) and ARRAY_SIZE yourself, or export the
# scheduler's variables as SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT.
# Preempted or killed jobs resume: rerun the same command with the same store.
#
# Environment: PROFILE (default cuda), STORE_DIR (default node-stores), extra queue
# flags after the phase, e.g. `... tune-coarse --method ppo`.
#SBATCH --job-name=craftax
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

phase="${1:?usage: run_array.sh <phase> [queue flags]}"
shift
index="${ARRAY_INDEX:-${SLURM_ARRAY_TASK_ID:?no array index}}"
size="${ARRAY_SIZE:-${SLURM_ARRAY_TASK_COUNT:?no array size}}"
store_dir="${STORE_DIR:-node-stores}"
mkdir -p "$store_dir"
store_dir="$(cd "$store_dir" && pwd)"

# A node's own SQLite file: nodes never write one database concurrently. The artifacts
# go to mlruns/ beside it (see orchestration/tracking.py).
store="sqlite:///$store_dir/mlruns-node-$index.db"

exec uv run python -m wut_murl_craftax.orchestration.queue \
    --phase "configs/phases/$phase.yaml" \
    --profile "configs/profiles/${PROFILE:-cuda}.yaml" \
    --worker-index "$index" --num-workers "$size" --tracking-uri "$store" "$@"
