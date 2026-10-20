---
title: "Craftax Baselines"
type: index
parent: "[[wut_murl_craftax]]"
tags:
  - craftax
  - baselines
  - benchmark
  - reinforcement-learning
  - jax
source: "https://github.com/MichaelTMatthews/Craftax_Baselines"
created: 2026-10-20
---
# Craftax Baselines

> [!abstract] Scope
> The official baseline code for the Craftax paper (Matthews et al., ICML 2024, [arXiv 2402.16801](https://arxiv.org/abs/2402.16801)): PureJaxRL-style PPO, PPO-RNN, ICM, E3B and RND on Craftax and Craftax-Classic. These notes document the repository, its configs, the benchmark protocol and the published numbers, so our agents can be compared with them. Per-method deep dives of these and later methods (PPO-GTrXL, PQN, model-based) live in [[Existing Implementations]].

## Notes

| Note | Content |
|---|---|
| [[Craftax Baselines algorithms and hyperparameters]] | Per-algorithm networks, losses, code defaults vs paper tables, commands for the paper configs |
| [[Craftax Baselines benchmark protocol]] | Environments, Craftax-1B and Craftax-1M, seeds, metric, achievement success rates |
| [[Craftax Baselines results]] | Published scores per algorithm, budget and environment, with caveats |
| [[Craftax Baselines compute and wall-clock]] | Hardware, steps per second, run times |
| [[Craftax Baselines reproduction notes]] | Dependency pins, issues, pitfalls, what the repo does not include |

## Snapshot read

| Source | Version read |
|---|---|
| Baselines repo | `main` at [7ce36fa](https://github.com/MichaelTMatthews/Craftax_Baselines/commit/7ce36fa05b84a2c9e758012f1e6da402e1e3a891) (2026-06-16), 20 commits since 2024-03-27, MIT license |
| Craftax repo | `main` at [c3c2e0d](https://github.com/MichaelTMatthews/Craftax/commit/c3c2e0d038c4e641f9481320c158f457f30c28f3) (v1.6.1, 2026-06-20) |
| Paper | [arXiv v2](https://arxiv.org/pdf/2402.16801v2) (3 Jun 2024, latest version), published in PMLR 235:35104-35137 |
| Issues and PRs | all 5 issues and 6 PRs of the baselines repo; relevant Craftax issues (read 2026-10-20) |

- The baselines repo README says it holds the code for the paper baselines and is separate from the main repo "for packaging reasons" ([README](https://github.com/MichaelTMatthews/Craftax_Baselines#craftax-baselines)).
- The code is adapted from Chris Lu's PureJaxRL ([ppo.py header](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L36-L37); paper Section 4.1).
- No Weights & Biases project, run link or result files are published in either repo or on the [project page](https://craftaxenv.github.io/).

## Repository contents

| Path | Purpose |
|---|---|
| `ppo.py` | PPO with MLP (symbolic) or CNN (pixels); also ICM (`--train_icm`) and E3B (`--use_e3b`) |
| `ppo_rnn.py` | PPO with a GRU (PPO-RNN), network defined in the script |
| `ppo_rnd.py` | PPO with Random Network Distillation and a two-head critic; symbolic only |
| `models/actor_critic.py` | `ActorCritic` (MLP), `ActorCriticConv` (pixels), two unused variants |
| `models/icm.py` | ICM encoder, forward and inverse models |
| `models/rnd.py` | RND network and the two-value-head actor-critic |
| `wrappers.py` | `LogWrapper`, `BatchEnvWrapper`, `AutoResetEnvWrapper`, `OptimisticResetVecEnvWrapper` |
| `logz/batch_logging.py` | Aggregates per-seed metrics and sends them to W&B |
| `analysis/view_ppo_agent.py` | Replays a saved PPO (not PPO-RNN) policy in the pygame renderer |
| `requirements.txt`, `Dockerfile`, `build.sh`, `run_docker.sh` | Environment setup, one Docker container per GPU |

- No tests, CI, sweep configs, plotting scripts, logs or checkpoints are in the repo.

## Install

```bash
git clone https://github.com/MichaelTMatthews/Craftax_Baselines.git
cd Craftax_Baselines
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

- Source: [README, Installation](https://github.com/MichaelTMatthews/Craftax_Baselines#installation).
- The `jax[cuda12_pip]` extra in `requirements.txt` no longer exists in current JAX; see [[Craftax Baselines reproduction notes]].

## Run each baseline

| Baseline | Command (README) | Script and network |
|---|---|---|
| PPO | `python ppo.py` | `ppo.py`, `ActorCritic` MLP |
| PPO-RNN | `python ppo_rnn.py` | `ppo_rnn.py`, `ActorCriticRNN` (GRU) |
| ICM | `python ppo.py --train_icm` | `ppo.py` + `models/icm.py` |
| E3B | `python ppo.py --train_icm --use_e3b --icm_reward_coeff 0` | `ppo.py`, ICM encoder as the embedding |
| RND | `python ppo_rnd.py` | `ppo_rnd.py` + `models/rnd.py` |

- Source: [README, Run Experiments](https://github.com/MichaelTMatthews/Craftax_Baselines#run-experiments).
- Defaults reproduce the Craftax-1B PPO configuration on `Craftax-Symbolic-v1` with 1e9 steps; Craftax-1M needs explicit flags ([[Craftax Baselines algorithms and hyperparameters]]).
- The README E3B command keeps the code default `--e3b_reward_coeff 1.0`, not the paper's 0.001 ([[Craftax Baselines reproduction notes]]).
- Saving a policy: add `--save_policy`; it is written only when W&B is on, under `<wandb run dir>/policies` ([ppo.py L647-L665](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L647-L665)).
- Viewing a policy: `python analysis/view_ppo_agent.py --path <wandb run>/files` (README, Visualisation); reads `config.yaml` and `policies/` from that folder.
- Docker: `./build.sh`, then `./run_docker.sh <gpu id | all> <command>`; `all` launches one container on each of GPUs 0-7 and reads the W&B key from `./wandb_key` ([run_docker.sh](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/run_docker.sh)).

## Common flags

All three scripts parse these (defaults from `ppo.py` [L669-L707](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L669-L707); `ppo_rnn.py` and `ppo_rnd.py` match).

| Flag | Default | Meaning |
|---|---|---|
| `--env_name` | `Craftax-Symbolic-v1` | Craftax environment name |
| `--total_timesteps` | `1e9` | Env steps summed over all parallel envs (scientific notation accepted) |
| `--num_envs` | 1024 | Parallel environments |
| `--num_steps` | 64 | Rollout length per env per update |
| `--num_minibatches` | 8 | Minibatches per epoch |
| `--update_epochs` | 4 | PPO epochs per update |
| `--lr` | 2e-4 | Adam learning rate (linearly annealed to 0 by default) |
| `--anneal_lr` / `--no-anneal_lr` | on | Linear LR decay |
| `--gamma` | 0.99 | Discount |
| `--gae_lambda` | 0.8 | GAE lambda |
| `--clip_eps` | 0.2 | PPO ratio and value clip |
| `--ent_coef` | 0.01 | Entropy bonus |
| `--vf_coef` | 0.5 | Value loss weight |
| `--max_grad_norm` | 1.0 | Global gradient-norm clip |
| `--layer_size` | 512 | Hidden width (also GRU size in PPO-RNN) |
| `--activation` | `tanh` | Parsed but never passed to a network (no effect) |
| `--seed` | random | Drawn with `np.random.randint(2**31)` when omitted |
| `--num_repeats` | 1 | Seeds trained in one process via `jax.vmap` |
| `--use_optimistic_resets` | on | Optimistic resets (Craftax paper Appendix C) |
| `--optimistic_reset_ratio` | 16 | Envs per freshly generated world |
| `--use_wandb` / `--no-use_wandb` | on | W&B logging and policy saving |
| `--wandb_project`, `--wandb_entity` | none | W&B destination |
| `--debug` | on | Enables the per-update logging callback |
| `--jit` | on | `--no-jit` runs under `jax.disable_jit()` |
| `--save_policy` | off | Save final train state with Orbax |

- Algorithm-specific flags (ICM, E3B, RND) are in [[Craftax Baselines algorithms and hyperparameters]].

## Environment names

Accepted by `craftax.craftax_env.make_craftax_env_from_name` ([craftax_env.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d038c4e641f9481320c158f457f30c28f3/craftax/craftax_env.py)):

| Name | Observation | Baselines that run on it |
|---|---|---|
| `Craftax-Symbolic-v1` | flat vector, 8268 | PPO, PPO-RNN, ICM, E3B, RND |
| `Craftax-Pixels-v1` | image, 110x130x3 | PPO (CNN) only |
| `Craftax-Classic-Symbolic-v1` | flat vector, 1345 | PPO, PPO-RNN, ICM, E3B, RND |
| `Craftax-Classic-Pixels-v1` | image, 63x63x3 | PPO (CNN) only |

- Observation sizes from paper Section 3.3.
- `*-AutoReset-v1` aliases exist only for `auto_reset=True`, which the scripts request only with `--no-use_optimistic_resets`.
- `ppo_rnd.py` raises `ValueError` for non-symbolic names; ICM and E3B assert a 1-D observation; `ppo_rnn.py` has no convolutional encoder.

## Logging

- Metrics go to W&B only, through a `jax.debug.callback` that fires each update when both `--debug` and `--use_wandb` are on ([ppo.py L581-L592](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L581-L592)).
- With `--no-use_wandb` nothing is logged; the script only prints total time and `SPS` at the end (time includes JIT compilation).
- Run name: `<env>-<steps in M>M`, with `-PPO_RNN-` or `-PPO_RND-` inserted for those scripts.
- Logged keys ([logz/batch_logging.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/logz/batch_logging.py)):

| Key | Meaning |
|---|---|
| `episode_return` | Mean return of episodes that ended in this update's rollout (training policy, stochastic) |
| `episode_length` | Mean length of those episodes |
| `Achievements/<name>` | % of those episodes that unlocked the achievement (one key per achievement) |
| `achievements` | Sum of the above / 100, the mean number of achievements per episode |
| `intrinsic_reward`, `extrinsic_reward` | Mean per-step rewards (ICM, E3B, RND) |
| `icm_inverse_loss`, `icm_forward_loss` | ICM losses |
| `rnd_loss` | RND distillation loss |
| `sps` | Steps per second between two logging calls |

- With `--num_repeats > 1`, returns, lengths and `rnd_loss` are averaged over seeds; other keys are logged as per-seed arrays.
- Updates with no finished episode give NaN and are dropped from the aggregate.
