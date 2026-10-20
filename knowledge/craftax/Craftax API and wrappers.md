---
title: "Craftax API and wrappers"
type: handbook
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, handbook, api, jax]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax API and wrappers

## Install

1. `pip install craftax` installs the latest PyPI release with CPU JAX ([README Installation](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md)).
2. `pip install git+https://github.com/MichaelTMatthews/Craftax.git@main` installs the newest commit.
3. For NVIDIA GPUs add the matching JAX wheel, for example `pip install -U "jax[cuda12]"`; on Apple silicon see [[Craftax on Apple GPU]].
4. To edit the game: clone, `pip install -e ".[dev]"` (needs `pip>=23.0`), `pre-commit install`.
5. Pinning and Python support: [[Craftax versions and dependencies]].

## Make, reset, step

```python
import jax
from craftax.craftax_env import make_craftax_env_from_name

env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

rng = jax.random.PRNGKey(0)
rng, k_reset, k_act, k_step = jax.random.split(rng, 4)
obs, state = env.reset(k_reset, env_params)
action = env.action_space(env_params).sample(k_act)
obs, state, reward, done, info = env.step(k_step, state, action, env_params)
```

- Adapted from [README Basic Usage](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md).

| Method | Signature (v1.6.1) | Returns |
|---|---|---|
| `reset` | `reset(key: jax.Array, params=None)` | `(obs, state)` |
| `step` | `step(key: jax.Array, state, action, params=None)` | `(obs, state, reward, done, info)` |
| `reset_env`, `step_env` | same, without auto-reset or the `params=None` default | raw transition |
| `get_obs(state)` | | observation for `state` |
| `is_terminal(state, params)` | | `timestep >= max_timesteps` or dead (or boss beaten / in lava, by variant) |
| `action_space(params)` | | `spaces.Discrete(43)` or `Discrete(17)` |
| `observation_space(params)` | | `spaces.Box(0.0, 1.0, shape, float32)` |

- `params=None` falls back to `env.default_params`.
- `reset` and `step` are wrapped in `jax.jit(static_argnums=(0,))`: only `self` is static, so `params` are traced (in v1.5.0 the vendored `EnvironmentNoAutoReset` marked `params` static too) ([environment_bases.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/environment_base/environment_bases.py); [release v1.6.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.6.0) "made env_params not static").
- `state` is a `flax.struct.dataclass` (`EnvState`); edit it with `state.replace(...)`.
- `info["discount"]` is 0.0 on a terminal step and 1.0 otherwise; `info["Achievements/<name>"]` is 100.0 for unlocked achievements on the terminal step and 0 on other steps ([envs/common.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/envs/common.py)).
- Every action is legal at every step; an action whose prerequisites are missing acts as a no-op ([paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3)).

## Batch with vmap

```python
reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
obs, state = reset_fn(jax.random.split(k_reset, 1024), env_params)
```

- Keys, states and actions are batched, `params` is shared (`None` in `in_axes`), as in `BatchEnvWrapper` ([Craftax_Baselines wrappers.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa/wrappers.py)).
- Checked on 1.6.1: a vmapped reset over 8 keys returns observations of shape (8, 8268).
- To pull one env out of a batched state: `jax.tree_util.tree_map(lambda x: x[i], state)` (maintainer in [issue #13](https://github.com/MichaelTMatthews/Craftax/issues/13)).

## Wrappers

The wrappers are not in the `craftax` package; they live in [Craftax_Baselines wrappers.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa/wrappers.py) (MIT). Copy the file into the project.

| Wrapper | Purpose | Notes |
|---|---|---|
| `GymnaxWrapper` | Base class; forwards unknown attributes to the wrapped env | |
| `LogWrapper` | Tracks episode return and length in `LogEnvState` | Adds `info["returned_episode_returns"]`, `["returned_episode_lengths"]`, `["timestep"]`, `["returned_episode"]` (the `done` flag); the inner state is `state.env_state` |
| `AutoResetEnvWrapper` | Gymnax-style reset on `done` for a no-auto-reset env | Generates a new world every step |
| `BatchEnvWrapper(env, num_envs)` | vmaps `reset` and `step`; splits one key into `num_envs` keys | |
| `OptimisticResetVecEnvWrapper(env, num_envs, reset_ratio)` | Batched step that generates only `num_envs // reset_ratio` new worlds per step and hands them to the envs that finished | `num_envs % reset_ratio == 0` is asserted; if more envs finish than worlds generated, some get duplicate worlds |

- Order used by `ppo.py`: `make_craftax_env_from_name(name, auto_reset=not use_optimistic_resets)`, then `LogWrapper`, then either `OptimisticResetVecEnvWrapper(num_envs, reset_ratio=min(16, num_envs))` or `AutoResetEnvWrapper` followed by `BatchEnvWrapper` ([ppo.py](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa/ppo.py#L61)).
- Defaults in `ppo.py`: `--use_optimistic_resets` on, `--optimistic_reset_ratio 16`, `--num_envs 1024`.
- Optimistic resets give "around 2x" speed-up in Craftax; at 1024 envs and ratio 16, 64 worlds are generated per step ([paper App. C](https://arxiv.org/html/2402.16801v2#A3)).
- A wrapper that changes reward or `done` (for example single-achievement reward) goes inside `LogWrapper` and the reset wrapper (maintainer in [issue #37](https://github.com/MichaelTMatthews/Craftax/issues/37)).
- Never use a no-auto-reset env without one of the two reset wrappers ([README Gotchas](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md)).

## Play and record

| Command | Effect |
|---|---|
| `play_craftax` | pygame GUI for `Craftax-Symbolic-v1` (auto-reset); prints controls and each new achievement |
| `play_craftax_classic` | Same for Craftax-Classic |
| `--god_mode` | Sets `EnvParams.god_mode=True` |
| `--debug` | Runs under `jax.disable_jit()` (slow: a few seconds per step, [issue #17](https://github.com/MichaelTMatthews/Craftax/issues/17)) |
| `--save_trajectories` | Writes `play_data/trajectories_<unix time>[_GM].pkl` (bz2 pickle of states, actions, rewards, dones) |
| `--fps N` | Frame-rate cap, default 60 |

- Entry points from [pyproject.toml](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/pyproject.toml); flags from [play_craftax.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/play_craftax.py).
- First frame takes about 30 s to compile and the first action another 20 s ([README Play](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md)).
- Key bindings: `Action` table in [[Craftax observations and actions]]; strategy guide: [tutorial.md](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/tutorial.md).
- Load saved trajectories with `craftax.environment_base.util.load_compressed_pickle`.

## Render

| Function | Module | Output |
|---|---|---|
| `make_craftax_pixel_renderer(block_pixel_size, do_night_noise=True)` | `craftax.craftax.renderer` | jittable `state -> (H, W, 3)` array in 0..255 |
| `render_craftax_symbolic(state)` | `craftax.craftax.renderer` | the symbolic observation |
| `render_craftax_text(state)` | `craftax.craftax.renderer` | text: map as "creature on item on block" per tile plus inventory and intrinsics ([paper App. D.4](https://arxiv.org/html/2402.16801v2#A4.SS4)) |

- Pixel sizes per tile: `BLOCK_PIXEL_SIZE_AGENT=10` (agent obs), `BLOCK_PIXEL_SIZE_IMG=16`, `BLOCK_PIXEL_SIZE_HUMAN=64` (GUI) ([constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L15)).
- Larger image of the agent's view: `jax.jit(make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_HUMAN))(state)`, as the play script does.
- `render_craftax_text` is not jittable, so it cannot be vmapped; one user measured about 2 minutes for 256 observations ([issue #13](https://github.com/MichaelTMatthews/Craftax/issues/13)).
- Full-map renders: paper figures were made by setting `OBS_DIM` to the map size; `OBS_DIM` must be odd (47x47 was used) and the texture cache must be rebuilt ([issue #33](https://github.com/MichaelTMatthews/Craftax/issues/33)).
- Textures are built once and cached as `assets/texture_cache.pbz2` (Classic: `texture_cache_classic.pbz2`) inside the installed package; `export CRAFTAX_RELOAD_TEXTURES=true` forces a rebuild ([README Texture Caching](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md)). Since v1.6.1 textures load only when a pixel renderer needs them ([release v1.6.1](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.6.1)).
