---
title: "training"
type: module
domain: training
parent: "[[wut_murl_craftax]]"
tags: [architecture, training]
created: 2026-10-21
updated: 2026-10-21
related-code:
  - "[[src/wut_murl_craftax/training/__init__.py]]"
  - "[[src/wut_murl_craftax/training/ppo.py]]"
  - "[[src/wut_murl_craftax/training/checkpoints.py]]"
  - "[[src/wut_murl_craftax/training/records.py]]"
---
# training

## Responsibility

- Train agents on Craftax and Craftax-Classic and record what the analysis and the videos need: checkpoints, per-update metrics, per-episode achievements.
- Today: the baseline PPO (adapted from Craftax_Baselines); the controls and methods of [[Methods]] join through the trainer interface of [[2026-10-20-craftax-infrastructure]].

## Public surface

- `PPOConfig`, `make_network`, `train(config, out)`: the baseline PPO; CLI `python -m wut_murl_craftax.training.ppo` (`make train-ppo`).
- `checkpoint_schedule`, `save_checkpoint`, `load_params`, `load_index`: log-spaced checkpoints (msgpack params + `index.json`).
- `RunLog`, `load_metrics`, `TrainingMetrics`, `load_episodes`, `Episodes`, `first_unlocks`, `achievement_names`, `achievement_weights`: training records ([[Recording training data]]).

## Data flow

- In: a config and the device roles from [[orchestration]] (CPU rollouts with a metal learner on Apple GPU, all-GPU on CUDA).
- Out: `metrics.npz`, `episodes.npz`, checkpoints and `index.json` in the run folder, read by [[visualisation]] and the analysis.

## Invariants

- Same seed, same config, same backend: same parameters and records.
- Score is training-episode return as % of the game's maximum (226 or 22).

## Known issues

- None.

## Tests

- [[tests/training/test_ppo.py]], [[tests/training/test_checkpoints.py]], [[tests/training/test_records.py]]
