---
title: "Craftax Baselines reproduction notes"
type: benchmark
parent: "[[Craftax Baselines]]"
tags:
  - craftax
  - reproducibility
  - dependencies
  - pitfalls
source: "https://github.com/MichaelTMatthews/Craftax_Baselines/issues"
created: 2026-10-20
---
# Craftax Baselines reproduction notes

> [!abstract] Summary
> Dependencies are almost all unpinned, the published ICM numbers are wrong, the RND numbers predate a bug fix, and the README E3B command does not use the paper's coefficient. Pin versions, set every paper hyperparameter explicitly and record seeds when comparing against the scoreboard ([[Craftax Baselines results]]).

## Dependencies and pins

| Item | Value | Source |
|---|---|---|
| `requirements.txt` | `jax[cuda12_pip]`, distrax, optax, flax, numpy, black, pre-commit, argparse, wandb, `orbax-checkpoint==0.5.0`, pygame, gymnax, chex, matplotlib, imageio, craftax | [requirements.txt](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/requirements.txt) |
| Only pin | `orbax-checkpoint==0.5.0` (released 2024-01-16), added 2025-01-08 "until I can be bothered to migrate" | [commit a2a029d](https://github.com/MichaelTMatthews/Craftax_Baselines/commit/a2a029d) |
| JAX wheels | installed with `-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` | README, Installation |
| `craftax` | unpinned; latest on PyPI is 1.6.1, requires Python >= 3.9 | PyPI |
| Docker base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`, installs Python 3.8 from deadsnakes, then `python3 -m pip install $REQS` | [Dockerfile](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/Dockerfile) |
| Formatter | black 22.3.0 via pre-commit | `.pre-commit-config.yaml` |
| Lock file | none; the package versions behind the paper are not recorded | repo tree |

- Current JAX (0.12.0 on PyPI) offers extras `cuda12`, `cuda13` and others but not `cuda12_pip`, so the requirement no longer selects a CUDA build.
- The Craftax README now recommends `pip install -U "jax[cuda12]"` for NVIDIA GPUs ([README, GPU-Enabled JAX](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/README.md)).
- Open [PR #10](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/10) (2026-03-11) migrates checkpointing to the current Orbax API and fixes deprecations; unmerged.
- The `jax.tree_map` deprecation crash ([issue #7](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/7)) was fixed on 2026-06-16 (commit 6b0385f).

## Issues and PRs in the baselines repo

| # | Title | State | Outcome |
|---|---|---|---|
| [1](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/1) | reward (% of max) | closed | Max reward is 226 for Craftax, 22 for Craftax-Classic |
| [2](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/2) | Problem with policy save file in view_ppo_agent.py | open | `--save_policy` raises `ValueError: Unrecognized name format` (M1 Mac) and saves a step folder with a `.0` suffix; RNN policies cannot be viewed (`KeyError: 'Dense_7'`); maintainer plans an Orbax refactor |
| [4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4) | Can't reproduce ICM results (Craftax-1M) | closed | Paper ICM sweep ran on an old commit with an inverted done mask; ICM results are wrong |
| [7](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/7) | This does not compile atm | closed | `jax.tree_map` removal; fixed; commenter warns that W&B connects to the cloud by default |
| [11](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/11) | `OptimisticResetVecEnvWrapper` silently corrupts states when `sum(done) < num_resets` | open | Relied on undefined `jax.random.choice` behaviour; maintainer says invalid indices hit only non-done workers in practice; fix in commit 7ce36fa |
| [PR 5](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/5) | Fix RND loss | merged 2025-05-12 | Intrinsic GAE used the extrinsic last value |
| [PR 8](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/8) | Add: Algorithm DQN | closed, unmerged | No DQN baseline in the repo |
| [PR 10](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/10) | Fix deprecation crashes and update to current JAX/Orbax APIs | open | See above |

## Relevant Craftax repo issues

| # | Point | State |
|---|---|---|
| [10](https://github.com/MichaelTMatthews/Craftax/issues/10) | No resume-from-checkpoint; `--save_policy` exists only for visualisation; user reports reproducing PPO-RNN results | closed |
| [19](https://github.com/MichaelTMatthews/Craftax/issues/19) | Pixel tile size was 7 instead of the paper's 10; fixed; paper experiments were all symbolic | closed |
| [28](https://github.com/MichaelTMatthews/Craftax/issues/28) | 60 GB of an H100 taken on import: default JAX preallocation of 75% | closed |
| [34](https://github.com/MichaelTMatthews/Craftax/issues/34) | PPO-RNN was never run on pixels; pixels about 10k steps per second | closed |
| [37](https://github.com/MichaelTMatthews/Craftax/issues/37) | All achievements are always active; single-task reward needs code changes | closed |
| [55](https://github.com/MichaelTMatthews/Craftax/issues/55) | `pip install craftax` fails on Python 3.14 (pygame hard dependency) | open |
| [57](https://github.com/MichaelTMatthews/Craftax/issues/57) | Symbolic observation drops visible mobs; 87 of 25,600 Classic observations (0.340%) disagree with state on 1.6.1 | open |
| [60](https://github.com/MichaelTMatthews/Craftax/issues/60) | Craftax-Classic `DO` action revives despawned and unused mobs | open |

- Issue #53 (swapped `ENTER_DUNGEON`/`ENTER_GNOMISH_MINES` enum values) was closed by its author as a misreading.

## Pitfalls in the code

- W&B is on by default: runs need a W&B login, or `--no-use_wandb`, which also turns off all metric logging and policy saving.
- `--seed` defaults to a random draw; pass it explicitly to make runs repeatable.
- `--activation` is parsed but never used; PPO is tanh (class default), PPO-RNN is ReLU.
- README E3B command leaves `--e3b_reward_coeff` at 1.0; paper Table 9 uses 0.001 (1B) and Table 13 uses 0.01 (1M).
- `--rnd_output_size` defaults to 512; paper Table 10 uses 256.
- Craftax-1M needs `--total_timesteps 1e6 --num_envs 256 --num_steps 16 --lr 3e-4`; defaults are the 1B config ([[Craftax Baselines algorithms and hyperparameters]]).
- `ppo_rnd.py` raises `ValueError` on pixel environments; ICM and E3B assert 1-D observations; `ppo_rnn.py` has no CNN encoder.
- Pixel PPO's actor squeezes features through a ReLU layer of width `n_actions` before the logits ([models/actor_critic.py L102-L114](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/models/actor_critic.py#L102-L114)).
- `--no-use_optimistic_resets` builds an auto-resetting env and also wraps it in `AutoResetEnvWrapper`, so a reset is computed twice per step.
- `*-AutoReset-v1` env names fail with optimistic resets on (the default).
- Optimistic resets reuse one fresh world for several workers when more than `num_envs / 16` episodes end in the same step (paper Appendix C).
- Logged returns are training-episode returns; updates without a finished episode give NaN and are skipped.
- `--save_policy` saves only the final train state; nothing supports resuming.
- `analysis/view_ppo_agent.py` loads only MLP or CNN PPO policies and prints achievement counts out of a hard-coded 22, also for Craftax.
- JAX grabs 75% of GPU memory at import unless configured otherwise (Craftax issue #28).

## Paper inconsistencies to keep in mind

- Section 3.3 says Craftax has 65 achievements; Table 6 lists 67 (IDs 0-66), which sum to the stated maximum 226.
- Appendix D gives a 100,000-step episode limit; Appendix G.1 calls the full length 10000 when cutting UED episodes to 4096.
- The Introduction points to Appendix F.3 for PPO solving Classic in 51 minutes, but F.3 shows PPO-RNN results.
- Section 4.1 cites ICM as Sukhbaatar et al. (2018); the README links the ICM paper arXiv 1705.05363.

## Not included in the repo

- No DQN, PQN, transformer, model-based or UED baselines (UED used JaxUED).
- No evaluation script with a greedy policy or held-out levels.
- No results, logs, W&B links, checkpoints or seed lists.
- No hyperparameter sweep configs or the scripts behind paper figures.
- No pixel results and no PPO-RNN, ICM, E3B or RND support for pixels.
- No PPO-RNN policy viewer or checkpoint resume.
- No tests or CI.
- The human-play offline dataset is linked from the Craftax README, not this repo, and targets Craftax v1.1.0 or earlier.

## Could not verify

- The scoreboard numbers themselves: no public runs or logs exist to recompute them.
- How the scoreboard number is aggregated over seeds and time.
- Whether the paper runs used `--num_repeats` or separate processes per seed.
- That the current code runs end to end with today's JAX, Flax and Orbax; these notes come from reading, not executing, the code.
