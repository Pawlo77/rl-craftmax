---
title: "Recording training data"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, training, logging, ppo, determinism]
created: 2026-10-20
updated: 2026-10-20
---
# Recording training data

What a training run must save so that [[Checkpoint comparison grid]], [[Animated training curves]], [[Achievement timeline]] and [[Agent internals overlay]] can be rendered afterwards without retraining. Tested on a PPO run adapted from Craftax_Baselines `ppo.py` (purejaxrl losses and defaults, no distrax, wandb or orbax) on `Craftax-Classic-Symbolic-v1`.

## Steps

1. Save four artefacts per run directory:

   | File | Grain | Feeds |
   |---|---|---|
   | `config.json` | run: env, hyperparameters, `n_updates`, `batch`, achievement names, `ckpt_updates`, backend and devices | all |
   | `metrics.npz` | one row per PPO update | curves, timeline |
   | `episodes.npz` | one row per finished training episode | timeline (first unlock) |
   | `ckpt/<update>.msgpack` + `ckpt/index.json` | one file per checkpoint, params only | grid, overlay, eval |

2. Checkpoint at update 0 (random policy), the last update and log-spaced updates in between; learning is fastest early, so linear spacing wastes most checkpoints on the plateau:

   ```python
   ckpts = sorted({0, n_updates} | set(np.unique(np.round(np.geomspace(1, n_updates, n_ckpt)).astype(int)).tolist()))
   # n_updates=122, n_ckpt=12 -> [0, 1, 2, 4, 6, 9, 14, 21, 33, 51, 79, 122]
   ```

3. Save params only, with the env-step count and wall clock in the index:

   ```python
   f.write_bytes(flax.serialization.to_bytes(ts.params))
   index.append(dict(update=u, env_steps=u * batch, wall_s=round(time.time() - t0, 1), file=f.name))
   # load: flax.serialization.from_bytes(net.init(key, jnp.zeros((1, *obs_dim))), f.read_bytes())
   ```

4. Log one metrics row per update: `update, env_steps, wall_s, n_episodes, return, length, policy_loss, value_loss, entropy, approx_kl, explained_var` and `ach/<name>` = % of episodes finished in this update that unlocked the achievement.
5. Log every finished episode: `update`, `ret`, `length` and the achievement vector bit-packed (`np.packbits(ach, axis=-1)`, 3 bytes for 22 or 9 bytes for 67 achievements). Inside the jitted rollout keep the achievement flags of every step and mask by `done` on the host:

   ```python
   ach = jnp.stack([info[f"Achievements/{n}"] for n in names], -1) > 0   # info is 100 * achieved, only at done
   ...
   d = np.asarray(tr["d"]).astype(bool); ach_done = np.asarray(tr["ach"])[d]
   ```

6. Flush `metrics.npz` and `episodes.npz` at every checkpoint, not only at the end, so a killed run still has curves up to its last checkpoint.
7. After training, run fixed-seed eval rollouts for every checkpoint and store replayable trajectories, not videos. Key streams per eval seed (identical for every checkpoint, so every checkpoint starts in the same world):

   ```python
   def keys(seed):
       base = jax.random.PRNGKey(seed)
       return base, jax.random.fold_in(base, 1), jax.random.fold_in(base, 2)   # reset, env step t, action t

   def rollout(params, seed, actions=None):            # env made with auto_reset=False
       k_reset, k_env, k_act = keys(seed)
       obs, st = env.reset(k_reset, pe)
       def step(c, t):
           obs, st, alive = c
           logits, v = net.apply(params, obs[None]); logits, v = logits[0], v[0]
           a = actions[t] if actions is not None else jax.random.categorical(jax.random.fold_in(k_act, t), logits)
           obs2, st2, r, d, _ = env.step(jax.random.fold_in(k_env, t), st, a, pe)
           return (obs2, st2, alive & ~d), dict(state=st, obs=obs, act=a, probs=jax.nn.softmax(logits), value=v,
                                                 reward=r * alive, done=d, alive=alive, ach=st2.achievements)
       return jax.lax.scan(step, (obs, st, jnp.bool_(True)), jnp.arange(T))[1]
   ```

8. Per checkpoint save `eval/<update>.npz` with `seeds, act, probs (float16), value, reward, done, alive, ach, obs (float16)` and a per-step state digest; drop `state`.
9. Re-render any episode later by replaying the stored actions with the same seed (`actions=` branch above), then `jax.vmap(make_craftax_pixel_renderer(16))` over the states in chunks of 128; see [[Rendering Craftax frames]].
10. Check determinism once per code or library change (results below).
11. On Apple silicon run the env rollout on CPU and the PPO update on the GPU (`JAX_PLATFORMS=metal,cpu`): `jax.devices()[0]` is the Metal learner, `jax.devices("cpu")[0]` the actor; put env state on the CPU with `jax.default_device`, `jax.device_put(ts.params, cpu)` before each rollout and `jax.device_put(batch, metal)` before each update; see [[Craftax on Apple GPU]].

## Results

PPO, `Craftax-Classic-Symbolic-v1`, 256 envs x 64 steps (16,384 steps per update), MLP 3 x 256 tanh, lr 2e-4 annealed, 4 epochs x 8 minibatches; jax 0.11.2, craftax 1.6.1, M4 Pro.

| Run | Value |
|---|---|
| 2M steps (122 updates), CPU | 207 s wall clock, 9,642 steps/s incl. compile, peak RSS 1.24 GB |
| Training return at 2M steps, last 10 updates | 5.29 = 24.0% of the maximum 22 |
| Eval return at 2M steps, 16 seeds, stochastic policy | 27.2% of 22 |
| Training episodes logged | 12,983 |

End-to-end PPO throughput, same config, 1M steps (61 updates), steps/s between updates 9 and 61 (excludes compilation); other agents' benchmarks ran on the machine at the same time (load average 4 to 17), so repeats differ:

| Backend | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---|---|---|
| CPU (`JAX_PLATFORMS=cpu`) | 5,482 | 7,434 | 6,302 (separate venv, same pins) |
| CPU actor + Metal learner (`JAX_PLATFORMS=metal,cpu`, metaljax 0.11.9) | 8,810 | 8,200 | |
| All on Metal (`JAX_PLATFORMS=metal`) | 1,496 (updates 9 to 51) | | |

- All-Metal run: update 1 finished 26.3 s after start (includes compilation).
- Craftax-Symbolic, CPU actor + Metal learner, 256 envs, width 256, sharing the machine with a Classic run: 1,124 steps/s to update 13.

Storage:

| Artefact | Size |
|---|---|
| Checkpoint, Classic, width 256 (params only) | 3,828,317 B |
| Checkpoint, Craftax-Symbolic, width 256 | 18,033,349 B |
| Checkpoint, Craftax-Symbolic, width 512 | 38,163,145 B |
| `metrics.npz`, 122 updates x 33 columns | 13,614 B |
| `episodes.npz`, 12,983 episodes | 47,151 B |
| `eval/<update>.npz`, 16 seeds x 1000 steps, obs float16 | 1.03 to 1.27 MB per checkpoint |
| Rendered frames, one episode, 144 x 144 RGB `.npy` | 11.9 to 19.3 MB (not needed: replay) |

Eval timing: 16 seeds x 1000 steps in 1.1 to 1.4 s per checkpoint after a 4.1 s first call; rendering one episode 0.5 s.

Determinism checks (`Craftax-Classic-Symbolic-v1`, 300 steps, CPU):

| Check | Result |
|---|---|
| Same params, same seed, two runs: same actions | True |
| Seed alone vs the same seed inside a `vmap` over 16 seeds: same actions | True |
| Replay of stored actions: same state digest every step | True |
| Replay of stored actions under other params: same states | True |
| Different seed: different actions | True |
| Grid panels rendered from stored frames vs from replay (FFV1, lossless) | bit-identical, 4 of 4 |

## Pitfalls

- `info["Achievements/<name>"]` is `100 x achieved` and non-zero only on the `done` step; per-update success rates are averages over episodes that finished in that update, which over-weights short episodes early in training.
- The paper metric is episode return (achievements plus the 0.1 x health term) as % of 22 (Classic) or 226 (Craftax); see [[Craftax scoring and evaluation]].
- Training-episode score and fixed-seed eval score differ (24.0% vs 27.2% at 2M steps); label which one a graphic shows.
- Same seed means the same world and the same env random stream; trajectories still diverge as soon as two checkpoints choose different actions.
- The frame for step `t` is the state before action `t`; store `state` from the scan input, not `st2`, or frames lag the action bars by one step.
- With `auto_reset=True` the observation after `done` already belongs to the next episode; eval uses `auto_reset=False` and an `alive` mask.
- A fixed-length eval scan keeps stepping after death; store only `alive` steps to cut the eval files.
- Saving rendered frames costs 10 to 20 times more than the trajectory; replay instead.
- All-Metal PPO is 4 to 6 times slower than CPU because Craftax env stepping is slow on metaljax; only the learner belongs on the GPU.

## Not tested

- Replay determinism across machines, OS or jax versions, and replay of a CPU-recorded trajectory on Metal.
- Greedy (argmax) eval rollouts (`--greedy` flag exists, not run).
- Recording RNN hidden states or intrinsic rewards (`reward_i` in Craftax_Baselines `ppo_rnd.py`) for the overlay.
- orbax checkpoints with optimizer state for resuming.
- Logging to wandb in parallel with the files above.
