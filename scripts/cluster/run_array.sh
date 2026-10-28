#!/usr/bin/env bash
# One node of a job array: runs every N-th run of a phase on a CUDA profile and writes
# its own MLflow store, merged into the main one later with `make merge-mlflow`.
#
# Slurm:  sbatch --array=0-3 --gres=gpu:1 scripts/cluster/run_array.sh tune-coarse
# Other schedulers: set ARRAY_INDEX (0-based) and ARRAY_SIZE yourself, or export the
# scheduler's variables as SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT.
# Preempted or killed jobs resume: rerun the same command with the same store. A node
# store records the array size (<db>.array.json); another size is refused until the
# store is merged, as its runs move to other workers.
#
# Environment: PROFILE (default cuda), STORE_DIR (default node-stores), SEED_FROM (a
# read-only copy of the main mlruns.db with its mlruns/ beside it: the node first takes
# this worker's runs from it, so it reuses finished runs instead of retraining them),
# extra queue flags after the phase, e.g. `... tune-coarse --method ppo`.
# The 1B phases (final-1b, combo-1b) start without SEED_FROM: the main store holds no
# 1B run. Copy back finished runs only, without state/ (cuda.yaml deletes it once a run
# is FINISHED and evaluated):
#   rsync -a --exclude state/ --exclude '*.tmp' --exclude owner.json node-stores/ <laptop>:node-stores/
# then `make merge-mlflow SRC=node-stores/mlruns-node-<i>.db` per node.
#
# Stop grace (review M12): --signal=B:TERM@600 sends SIGTERM to this shell, exec'd into
# uv, which forwards it to the queue, 10 minutes before the time limit. The queue ends
# its chunk (at most checkpoint_every = 50 updates on cuda.yaml: about 15 s for PPO-RNN
# and 40 s for PQN-RNN at the 1B estimates, plus the compile in a first chunk), saves
# state/ and closes the run KILLED before the SIGKILL at the limit. Preemption uses
# the partition's GraceTime instead: check it is a few minutes.
#SBATCH --job-name=craftax
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@600
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
seed=()
if [[ -n "${SEED_FROM:-}" ]]; then
    seed=(--seed-from "$SEED_FROM")
fi

exec uv run python -m wut_murl_craftax.orchestration.queue \
    --phase "configs/phases/$phase.yaml" \
    --profile "configs/profiles/${PROFILE:-cuda}.yaml" \
    --worker-index "$index" --num-workers "$size" --tracking-uri "$store" --node \
    ${seed[@]+"${seed[@]}"} "$@"
