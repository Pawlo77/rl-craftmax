---
title: "Recording training data"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, training, logging, ppo, determinism]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/training/ppo.py]]"
  - "[[src/wut_murl_craftax/training/checkpoints.py]]"
  - "[[src/wut_murl_craftax/training/records.py]]"
  - "[[src/wut_murl_craftax/visualisation/eval_rollouts.py]]"
  - "[[tests/training/test_ppo.py]]"
  - "[[tests/training/test_checkpoints.py]]"
  - "[[tests/training/test_records.py]]"
  - "[[tests/visualisation/test_eval_rollouts.py]]"
---
# Recording training data

What a training run saves so that [[Checkpoint comparison grid]], [[Animated training curves]], [[Achievement timeline]] and [[Agent internals overlay]] render later without retraining. The baseline PPO is adapted from Craftax_Baselines `ppo.py` (same losses and defaults, no distrax, wandb or orbax); both games work (`--env Craftax-Classic-Symbolic-v1` or `Craftax-Symbolic-v1`).

## Steps

1. Train; on Apple silicon start JAX with both platforms so `device_roles()` puts rollouts on the CPU and updates on Metal ([[Craftax on Apple GPU]]):

   ```bash
   JAX_PLATFORMS=metal,cpu uv run python -m wut_murl_craftax.training.ppo --steps 2e6 --out outputs/runs/classic
   # one flag per PPOConfig field: --env --envs --num-steps --width --lr --n-ckpt --seed ...
   ```

2. The run directory holds:

   | File | Grain | Feeds |
   |---|---|---|
   | `config.json` | run: `PPOConfig`, `n_updates`, `batch`, achievement names, `ckpt_updates`, actor and learner devices | all |
   | `metrics.npz` | one row per update: `update, env_steps, wall_s, n_episodes, return, length`, 5 losses, `ach/<name>` (% of episodes finished in the update) | curves, timeline |
   | `episodes.npz` | one row per finished episode: `update, ret, length`, achievements bit-packed (`np.packbits`) | timeline (first unlock) |
   | `ckpt/<update>.msgpack`, `ckpt/index.json` | params only (`flax.serialization`), with `env_steps` and `wall_s` | grid, overlay, eval |

3. Checkpoints are log-spaced (`checkpoint_schedule`), since learning is fastest early: 122 updates and 12 checkpoints give `[0, 1, 2, 4, 6, 9, 14, 21, 33, 51, 79, 122]`.
4. `RunLog.flush` rewrites `metrics.npz` and `episodes.npz` at every checkpoint, so a killed run keeps its curves up to the last checkpoint.
5. Evaluate every checkpoint on fixed seeds; trajectories are stored, frames are not:

   ```bash
   uv run python -m wut_murl_craftax.visualisation.eval_rollouts outputs/runs/classic --seeds 0-15 --steps 1000
   ```

   - `eval/<update>.npz`: `act, probs (float16), value, reward, done, alive, ach, obs (float16)` per seed and step, plus a per-step state `digest`.
   - Key streams per seed are those of `rollout.keys` (reset = `PRNGKey(seed)`, env step `t` = `fold_in(fold_in(base, 1), t)`), plus actions `fold_in(fold_in(base, 2), t)`; every checkpoint starts in the same world.

6. Rebuild frames by replaying the stored actions: `episode_frames(run, update, seed, bps)`; `as_rollout(...)` exports the episode in the gameplay `Rollout` format of [[Rendering Craftax frames]] (highlights, masters, cuts).

## Results

Measured on an M4 Pro while other agents' benchmarks shared the machine (load average 4 to 17).

PPO, Craftax-Classic-Symbolic, 256 envs x 64 steps (16,384 steps per update), MLP 3 x 256 tanh, lr 2e-4 annealed, 4 epochs x 8 minibatches:

| Run | Value |
|---|---|
| 2M steps, CPU, first prototype | 207 s, 9,642 steps/s incl. compile, peak RSS 1.24 GB |
| Training return at 2M steps (last 10 updates) | 5.29 = 24.0% of 22 |
| Eval return at 2M steps, 16 seeds | 27.2% of 22 |
| Training episodes logged in 2M steps | 12,983 |
| 0.5M steps, repo CLI, CPU actor + Metal learner | 52.8 s wall incl. start-up; 9,897 steps/s at the last update |

End-to-end PPO throughput, same config, 1M steps, steps/s between updates 9 and 61 (compile excluded):

| Backend | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---|---|---|
| CPU (`JAX_PLATFORMS=cpu`) | 5,482 | 7,434 | 6,302 |
| CPU actor + Metal learner (`JAX_PLATFORMS=metal,cpu`, metaljax 0.11.9) | 8,810 | 8,200 | |
| All on Metal (`JAX_PLATFORMS=metal`) | 1,496 (updates 9 to 51) | | |

- All-Metal run: update 1 finished 26.3 s after start (includes compilation).
- Craftax-Symbolic, CPU actor + Metal learner, 256 envs, width 256, sharing the machine with a Classic run: 1,124 steps/s to update 13.

Storage:

| Artefact | Size |
|---|---|
| Checkpoint, Classic, width 256 | 3,828,317 B |
| Checkpoint, Craftax-Symbolic, width 256 / 512 | 18,033,349 B / 38,163,145 B |
| `metrics.npz`, 122 updates x 33 columns | 13,614 B |
| `episodes.npz`, 12,983 episodes | 47,151 B |
| `eval/<update>.npz`, 16 seeds x 1000 steps | 1.03 to 1.27 MB |
| Rendered frames of one episode, 144 x 144 RGB | 11.9 to 19.3 MB (replay instead) |

- Eval of 9 checkpoints x 16 seeds x 1000 steps (repo CLI): 13.0 s.
- Determinism, all true: same seed gives the same actions; a seed alone equals the same seed inside a `vmap` of 16; replayed actions give the same state digest every step, also under other params; another seed differs; replayed frames equal the gameplay `replay` of `as_rollout` and the frames stored by the prototype (lossless, 4 of 4 panels).
- Tests (`tests/training`, `tests/visualisation/test_eval_rollouts.py`) train 4 updates of 512 steps; 8 + 5 tests.

## Pitfalls

- `info["Achievements/<name>"]` is `100 x achieved` and non-zero only on the `done` step; per-update rates average episodes that finished in that update, over-weighting short episodes early in training.
- The paper metric is episode return (achievements plus the 0.1 x health term) as % of 22 (Classic) or 226 (Craftax); see [[Craftax scoring and evaluation]].
- Training-episode score and fixed-seed eval score differ (24.0% vs 27.2% at 2M steps); label which one a graphic shows.
- Same seed means the same world and env random stream; trajectories diverge once two checkpoints act differently.
- The frame for step `t` is the state before action `t`.
- A fixed-length eval scan keeps stepping after death, masked by `alive`; `episode_frames` stops one frame after the last step.
- All-Metal PPO is 4 to 6 times slower than CPU because env stepping is slow on metaljax; only the learner belongs on the GPU.
- The renderer textures exist for 7, 16 and 64 px tiles; other sizes go through `Game.renderer`, which builds them.

## Not tested

- Replay determinism across machines, OS or jax versions, and replay of a CPU-recorded trajectory on Metal.
- Greedy (argmax) eval rollouts (`--greedy`).
- Recording RNN hidden states or intrinsic rewards (`reward_i` of Craftax_Baselines `ppo_rnd.py`).
- Checkpoints with optimizer state for resuming.
