---
title: "training"
type: module
domain: training
parent: "[[wut_murl_craftax]]"
tags: [architecture, training]
created: 2026-10-21
updated: 2026-10-28
related-code:
  - "[[src/wut_murl_craftax/training/__init__.py]]"
  - "[[src/wut_murl_craftax/training/ppo.py]]"
  - "[[src/wut_murl_craftax/training/checkpoints.py]]"
  - "[[src/wut_murl_craftax/training/records.py]]"
  - "[[src/wut_murl_craftax/training/interface.py]]"
  - "[[src/wut_murl_craftax/training/hooks.py]]"
  - "[[src/wut_murl_craftax/training/ppo_rnn.py]]"
  - "[[src/wut_murl_craftax/training/pqn_rnn.py]]"
  - "[[src/wut_murl_craftax/training/batch_renorm.py]]"
  - "[[src/wut_murl_craftax/training/wrappers.py]]"
---
# training

## Responsibility

- Train agents on Craftax and Craftax-Classic and record what the analysis and the videos need: checkpoints, per-update metrics, per-episode achievements.
- Today: the three controls PPO, PPO-RNN and PQN-RNN (adapted from Craftax_Baselines); the methods of [[Methods]] join through the trainer registry of [[2026-10-20-craftax-infrastructure]].

## Public surface

- `METHODS`, `Policy`: the trainer registry and the policy evaluation builds from it; `run_trainer(config)`: the trainer of a saved `config.json` (a combo run gets its own pair via `combo_pair`, never the live combo.yaml); `PPORNNConfig`, `PQNRNNConfig`, `learn`: the recurrent controls and the PQN-RNN learner other methods call.
- `PPOConfig`, `make_network`, `train(config, out)`: the baseline PPO; CLI `python -m wut_murl_craftax.training.ppo` (`make train-ppo`).
- `checkpoint_schedule`, `save_checkpoint`, `load_params`, `load_index`: log-spaced checkpoints (msgpack params + `index.json`).
- `mark_pruned`, `pruned_updates`, `PRUNED_FILE` (`ckpt/pruned.json`: `updates`, `date`, `reason`), `PrunedCheckpointError`: checkpoints deleted on purpose by storage retention.
- `RunLog`, `load_metrics`, `TrainingMetrics`, `load_episodes`, `Episodes`, `first_unlocks`, `achievement_names`, `achievement_weights`: training records ([[Recording training data]]).

## Modules

One line each; signatures and details: [[Modules]].

- `interface`: `METHODS` registry (name to `Trainer`) and the `Policy` protocol evaluation uses.
- `ppo`, `ppo_rnn`, `pqn_rnn`: the controls and the learners later methods reuse (`pqn_rnn.learn`).
- `hooks`: `HookFields`, the extension points of the recurrent controls (`None` hooks cost nothing).
- `batch_renorm`: BatchRenorm of the PQN-RNN input.
- `wrappers`: Craftax_Baselines env wrappers; the wrapper step also computes `StepEvents`.
- `checkpoints`: log-spaced params-only checkpoints, the full resume state (`state/`) and the prune marker (`ckpt/pruned.json`).
- `records`: per-update metrics, per-episode log and achievement tables.

## Data flow

- In: a config and the device roles from [[orchestration]] (CPU rollouts with a metal learner on Apple GPU, all-GPU on CUDA).
- Out: `metrics.npz`, `episodes.npz`, checkpoints, `index.json` and `state/` in the run folder (layout: [[Recording training data]]), read by [[visualisation]] and the analysis.

## Invariants

- Same seed, same config, same backend: same parameters and records.
- Score is training-episode return as % of the game's maximum (226 or 22).
- Per-checkpoint parquet files (`smdp_greedy_table.parquet`, `dyna_model_graph.parquet`) type their text columns explicitly, so a checkpoint without stored keys still joins later rows; `smdp.rebuild_greedy_table` rewrites one from `ckpt/` (ruling R81).
- A pruned checkpoint (listed in `ckpt/pruned.json`) is known-absent, never lost: `missing_checkpoints` skips it, `drop_missing_checkpoints` keeps it in `index.json` (readers take `env_steps` of its eval files from there), `load_params` raises `PrunedCheckpointError` (a `FileNotFoundError`) and `smdp.rebuild_greedy_table` keeps its old rows without reading it (with every checkpoint pruned and no old file it raises `FileNotFoundError`).
- A saved combo run replays the pair it trained with: its `combine` block, else the one registry pair its keys fit; zero or two fitting pairs raise (ruling R80a).

## Known issues

- None.

## Tests

- [[tests/training/test_interface.py]], [[tests/training/test_hooks.py]], [[tests/training/test_ppo.py]], [[tests/training/test_ppo_rnn.py]], [[tests/training/test_pqn_rnn.py]], [[tests/training/test_batch_renorm.py]], [[tests/training/test_wrappers.py]], [[tests/training/test_checkpoints.py]], [[tests/training/test_records.py]]
