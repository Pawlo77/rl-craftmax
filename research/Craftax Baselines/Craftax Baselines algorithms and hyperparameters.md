---
title: "Craftax Baselines algorithms and hyperparameters"
type: benchmark
parent: "[[Craftax Baselines]]"
tags:
  - craftax
  - baselines
  - hyperparameters
  - ppo
  - exploration
source: "https://github.com/MichaelTMatthews/Craftax_Baselines"
created: 2026-10-20
---
# Craftax Baselines algorithms and hyperparameters

> [!abstract] Summary
> All five baselines share one PureJaxRL PPO loop; ICM, E3B and RND add an intrinsic reward. Paper hyperparameters are in Appendix E, Tables 7-13 of [arXiv v2](https://arxiv.org/pdf/2402.16801v2); code defaults equal the Craftax-1B PPO and ICM tables but differ for E3B and RND. Method background and comparisons with other implementations: [[Existing Implementations]].

Code references point to commit [7ce36fa](https://github.com/MichaelTMatthews/Craftax_Baselines/tree/7ce36fa05b84a2c9e758012f1e6da402e1e3a891).

## Shared PPO core

- Rollout: `num_steps` steps in `num_envs` envs under `jax.lax.scan`, whole training jitted and `vmap`ped over `num_repeats` seeds ([ppo.py L621-L645](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L621-L645)).
- Updates: `total_timesteps // num_steps // num_envs`, so the last partial update is dropped.
- Advantages: GAE with `gamma` and `gae_lambda`, normalised per minibatch.
- Policy loss: clipped ratio with `clip_eps`.
- Value loss: clipped value loss (`clip_eps`), `0.5 * max(unclipped, clipped)`, weight `vf_coef`.
- Entropy bonus: `ent_coef`.
- Optimizer: Adam, eps 1e-5, global-norm clip `max_grad_norm`, learning rate linearly annealed to 0 over all updates when `anneal_lr` is on.
- Minibatches: flattened steps x envs shuffled into `num_minibatches` (PPO, ICM, E3B, RND); PPO-RNN shuffles whole env sequences instead.
- No observation normalisation, reward normalisation or reward scaling in any script.

## Networks

| Baseline | Class | Architecture (from code) |
|---|---|---|
| PPO (symbolic) | `ActorCritic` | Separate actor and critic MLPs, each 3 x Dense(512) + tanh, then Dense(n_actions) logits or Dense(1); orthogonal init sqrt(2), actor head 0.01, critic head 1.0 |
| PPO (pixels) | `ActorCriticConv` | 3 x [Conv 32 filters 5x5, ReLU, max-pool 3x3 stride 3], flatten; actor Dense(512) ReLU, Dense(n_actions) ReLU, Dense(n_actions); critic Dense(512) ReLU, Dense(1) |
| PPO-RNN | `ActorCriticRNN` | Dense(512) ReLU, GRU(512) with reset on done, then actor 2 x Dense(512) ReLU + logits and critic 2 x Dense(512) ReLU + value |
| ICM | `ICMEncoder`, `ICMForward`, `ICMInverse` | Each 3 x Dense(256) ReLU; encoder to latent 32; forward takes latent + one-hot action; inverse takes two latents, outputs log-softmax over actions |
| E3B | ICM encoder | Uses the ICM encoder output as the episodic embedding |
| RND | `RNDNetwork`, `ActorCriticRND` | Target and predictor 3 x Dense(256) ReLU + Dense(512); actor as PPO; two critics (extrinsic, intrinsic) each 3 x Dense(512) tanh + Dense(1) |

- Sources: [models/actor_critic.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/models/actor_critic.py), [ppo_rnn.py L40-L119](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo_rnn.py#L40-L119), [models/icm.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/models/icm.py), [models/rnd.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/models/rnd.py).
- Paper Section 4.1: "a 4 layer MLP of width 512 for both policy and value networks"; convolutions on the map view performed worse than fully connected networks.
- Paper Appendix E.1: PPO-RNN keeps PPO's hyperparameters with GRU hidden size equal to the MLP layer size.
- PPO-RNN uses ReLU throughout, while PPO's tuned activation is tanh (Table 7).
- `ActorCriticConvSymbolicCraftax` and `ActorCriticWithEmbedding` exist in `models/actor_critic.py` but no script uses them.

## PPO hyperparameters

| Hyperparameter | Code default | Craftax-1B (Table 7) | 1B values tried | Craftax-1M (Table 11) | 1M values tried |
|---|---|---|---|---|---|
| Entropy coefficient | 0.01 | 0.01 | {0, 0.01} | 0.01 | {0.01} |
| GAE lambda | 0.8 | 0.8 | {0.8, 0.9, 0.95} | 0.8 | {0.7, 0.8, 0.9, 0.95} |
| Gamma | 0.99 | 0.99 | {0.98, 0.99, 0.995, 0.999, 0.9995} | 0.99 | {0.99} |
| MLP layer size | 512 | 512 | {256, 512, 1024} | 512 | {128, 256, 512, 1024} |
| Learning rate | 2e-4 | 0.0002 | {0.0002, 0.0003, 0.0004} | 0.0003 | {0.0002, 0.0003, 0.0004} |
| Environment workers | 1024 | 1024 | {256, 512, 1024} | 256 | {16, 64, 256} |
| Minibatches | 8 | 8 | {4, 8, 16} | 8 | {2, 4, 8, 16} |
| Steps per rollout | 64 | 64 | {64} | 16 | {16, 64, 256} |
| Update epochs | 4 | 4 | {4} | 4 | {1, 2, 4, 8, 16} |
| Clip epsilon | 0.2 | 0.2 | {0.2} | 0.2 | {0.2} |
| Value coefficient | 0.5 | 0.5 | {0.5} | 0.5 | {0.5} |
| Activation | tanh (class default) | tanh | {tanh} | tanh | {tanh, relu} |
| Anneal LR | True | True | {True} | True | {True} |
| Max grad norm | 1.0 | not listed | | not listed | |

- Source: paper Appendix E, [Tables 7 and 11](https://arxiv.org/pdf/2402.16801v2) (pages 33-34); code [ppo.py L669-L707](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L669-L707).
- 1B tuning: one seed per run, each hyperparameter tuned individually from informally chosen values (Appendix E.1).
- 1M tuning: random search over PPO, then random search for ICM and E3B (Appendix E.2).
- Derived batch sizes: 1B uses 65,536 steps per update and 8,192 per minibatch; 1M uses 4,096 and 512.

## ICM hyperparameters

| Hyperparameter | Flag | Code default | 1B (Table 8) | 1B values tried | 1M (Table 12) | 1M values tried |
|---|---|---|---|---|---|---|
| Intrinsic reward coefficient beta | `--icm_reward_coeff` | 1.0 | 1.0 | {0.1, 0.5, 1.0} | 10.0 | {0.01, 0.1, 1.0, 10.0, 100.0} |
| World model learning rate | `--icm_lr` | 3e-4 | 0.0003 | {0.0003} | 0.0003 | {0.0003} |
| Forward loss coefficient | `--icm_forward_loss_coef` | 1.0 | 1.0 | {1.0} | 0.1 | {0.1, 0.5, 1.0} |
| Inverse loss coefficient | `--icm_inverse_loss_coef` | 1.0 | 1.0 | {1.0} | 0.5 | {0.1, 0.5, 1.0} |
| World model layer size | `--icm_layer_size` | 256 | 256 | {256} | 256 | {256} |
| Latent size | `--icm_latent_size` | 32 | 32 | {32, 64, 128} | 16 | {16, 32, 64, 128} |
| Exploration update epochs | `--exploration_update_epochs` | 4 | not listed | | not listed | |

- Source: paper [Tables 8 and 12](https://arxiv.org/pdf/2402.16801v2); code [ppo.py L709-L718](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L709-L718).
- Intrinsic reward: mean squared latent forward-prediction error x beta, zeroed on terminal steps, added to the extrinsic reward before GAE ([ppo.py L232-L280](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L232-L280)).
- The encoder is trained only by the inverse loss; the forward loss updates only the forward model.
- ICM nets have their own Adam optimisers and are updated after PPO on the same rollout, in `num_minibatches` minibatches.
- The published ICM numbers came from an older commit with an inverted done mask ([issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4)); see [[Craftax Baselines results]].

## E3B hyperparameters

| Hyperparameter | Flag | Code default | 1B (Table 9) | 1B values tried | 1M (Table 13) | 1M values tried |
|---|---|---|---|---|---|---|
| Intrinsic reward coefficient beta | `--e3b_reward_coeff` | 1.0 | 0.001 | {0.0005, 0.001, 0.005} | 0.01 | {0.0001, 0.001, 0.01, 0.1, 1.0} |
| Ridge regulariser lambda | `--e3b_lambda` | 0.1 | 0.1 | {0.005, 0.1, 0.5} | 0.5 | {0.001, 0.01, 0.1, 0.5} |

- Source: paper [Tables 9 and 13](https://arxiv.org/pdf/2402.16801v2); code [ppo.py L719-L730](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L719-L730).
- E3B requires `--train_icm` and `--icm_reward_coeff 0` (asserted); the ICM forward model still trains but its reward is zeroed.
- Per-env inverse covariance starts at identity / lambda, gets a rank-1 (Sherman-Morrison) update each step and resets at episode end ([ppo.py L250-L278](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo.py#L250-L278)).
- Bonus: `phi^T C phi` of the current observation's embedding x beta, zero on terminal steps.
- The paper does not list the ICM encoder settings used under E3B; the code then uses the ICM defaults (latent 32, layer 256).

## RND hyperparameters

| Hyperparameter | Flag | Code default | 1B and 1M (Table 10) | Values tried |
|---|---|---|---|---|
| Intrinsic reward coefficient beta | `--rnd_reward_coeff` | 1.0 | 1.0 | {0.01, 0.1, 1.0} |
| RND output dimension | `--rnd_output_size` | 512 | 256 | {4, 32, 256, 2048} |
| Distillation network learning rate | `--rnd_lr` | 3e-4 | 0.0003 | {0.00003, 0.0003, 0.003} |
| RND loss coefficient | `--rnd_loss_coeff` | 0.01 | 0.01 | {0.0001, 0.001, 0.01} |
| RND GAE coefficient | `--rnd_gae_coeff` | 0.01 | 0.01 | {0.01, 0.1, 1.0} |
| RND layer size | `--rnd_layer_size` | 256 | not listed | |
| Episodic intrinsic return | `--rnd_is_episodic` | False | not listed | |
| Exploration update epochs | `--exploration_update_epochs` | 1 | not listed | |

- Source: paper [Table 10](https://arxiv.org/pdf/2402.16801v2); code [ppo_rnd.py L659-L672](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo_rnd.py#L659-L672).
- Intrinsic reward: mean squared error between the fixed random target and the predictor on the next observation x beta, zeroed on terminal steps.
- Separate GAE for extrinsic and intrinsic streams; the intrinsic stream ignores episode ends unless `--rnd_is_episodic`.
- Actor advantage: `gae_e + rnd_gae_coeff * gae_i`, then normalised ([ppo_rnd.py L321-L324](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa05b84a2c9e758012f1e6da402e1e3a891/ppo_rnd.py#L321-L324)).
- Value loss: extrinsic + intrinsic clipped value losses.
- Before [PR #5](https://github.com/MichaelTMatthews/Craftax_Baselines/pull/5) (merged 2025-05-12) intrinsic GAE bootstrapped from the extrinsic value head.

## Commands for the paper configurations

Derived from the tables above; flags not shown keep code defaults, which match the tables unless stated.

| Config | Command |
|---|---|
| PPO, 1B | `python ppo.py` |
| PPO, 1M | `python ppo.py --total_timesteps 1e6 --num_envs 256 --num_steps 16 --lr 3e-4` |
| PPO-RNN, 1B | `python ppo_rnn.py` (ReLU network, see above) |
| ICM, 1B | `python ppo.py --train_icm` |
| ICM, 1M | `python ppo.py --train_icm --num_envs 256 --total_timesteps 1e6 --lr 3e-4 --num_steps 16 --icm_reward_coeff 10.0 --icm_forward_loss_coef 0.1 --icm_inverse_loss_coef 0.5 --icm_latent_size 16` |
| E3B, 1B | `python ppo.py --train_icm --use_e3b --icm_reward_coeff 0 --e3b_reward_coeff 0.001` |
| E3B, 1M | PPO 1M flags + `--train_icm --use_e3b --icm_reward_coeff 0 --e3b_reward_coeff 0.01 --e3b_lambda 0.5` |
| RND, 1B | `python ppo_rnd.py --rnd_output_size 256` |

- The ICM 1M command is the one used in [issue #4](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/4), built from Tables 11-12.
- Not stated in the paper: PPO-RNN settings for 1M and the PPO settings under RND at 1M.
