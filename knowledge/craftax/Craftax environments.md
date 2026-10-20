---
title: "Craftax environments"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, environment]
created: 2026-10-20
---
# Craftax environments

Eight environment classes: {Craftax, Craftax-Classic} x {Symbolic, Pixels} x {auto-reset, no auto-reset}. v1.4.0 split them "to make it very explicit which one is being used" ([release v1.4.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.4.0)).

## Names and classes

`make_craftax_env_from_name(name, auto_reset)` in [craftax_env.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_env.py) maps a name to a class:

| `name` | `auto_reset=True` | `auto_reset=False` |
|---|---|---|
| `Craftax-Symbolic-v1` | `CraftaxSymbolicEnv` | `CraftaxSymbolicEnvNoAutoReset` |
| `Craftax-Pixels-v1` | `CraftaxPixelsEnv` | `CraftaxPixelsEnvNoAutoReset` |
| `Craftax-Classic-Symbolic-v1` | `CraftaxClassicSymbolicEnv` | `CraftaxClassicSymbolicEnvNoAutoReset` |
| `Craftax-Classic-Pixels-v1` | `CraftaxClassicPixelsEnv` | `CraftaxClassicPixelsEnvNoAutoReset` |
| `Craftax-Symbolic-AutoReset-v1` (and the other three `-AutoReset-v1` aliases) | same class as the plain name | `ValueError: Unknown craftax environment` |

- Classes live in `craftax.craftax.envs.craftax_symbolic_env`, `craftax.craftax.envs.craftax_pixels_env`, `craftax.craftax_classic.envs.craftax_symbolic_env`, `craftax.craftax_classic.envs.craftax_pixels_env`.
- `make_craftax_env_from_params(classic: bool, symbolic: bool, auto_reset: bool)` returns the same classes by flags; the README calls it `make_craftax_env_from_args`, which does not exist.
- `env.name` of a no-auto-reset class is `Craftax-Symbolic-NoAutoReset-v1` (etc.), a string the factory does not accept.
- Every class takes an optional `StaticEnvParams` in its constructor; `env.default_params` returns `EnvParams()`.
- Verified on craftax 1.6.1 (CPU, Python 3.12, jax 0.12.0): all eight classes build, reset and step; the `-AutoReset-v1` alias with `auto_reset=False` raises.

## Auto-reset vs no auto-reset

| Base class | `step` behaviour | Use |
|---|---|---|
| `EnvironmentAutoReset` | Runs `step_env` and `reset_env` every call, then `jax.lax.select(done, reset, stepped)`, like gymnax | Drop-in gymnax env |
| `EnvironmentNoAutoReset` | Runs `step_env` only; after `done` the state keeps stepping into "invalid states" | Must be wrapped in `OptimisticResetVecEnvWrapper` or `AutoResetEnvWrapper` |

- Source: [environment_bases.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/environment_base/environment_bases.py), [README Gotchas](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md).
- Wrappers and their composition: [[Craftax API and wrappers]].

## Craftax vs Craftax-Classic

| Property | Craftax | Craftax-Classic |
|---|---|---|
| Package | `craftax.craftax` | `craftax.craftax_classic` |
| View (rows x cols) | 9 x 11 (`OBS_DIM`) | 7 x 9 |
| Pixels per tile (agent obs) | 10 (`BLOCK_PIXEL_SIZE_AGENT`) | 7 |
| Floors x map size | 9 x 48x48 | 1 x 64x64 |
| Block types | 37 | 17 |
| Actions | 43 | 17 |
| Achievements | 67 | 22 |
| `max_timesteps` default | 100000 | 10000 |
| `always_diamond` default | False | True |
| Death by lava | Player cannot enter lava (`COLLISION_LAND_CREATURE` blocks water and lava) | Entering lava ends the episode; since v1.6.0 it sets health to 0, so the health penalty applies |
| `info` keys | 67 `Achievements/<name>` + `discount` | 22 `Achievements/<name>` + `score` + `discount` |

- Sources: [craftax constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py), [classic constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/constants.py), [classic game_logic.py `update_health`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py#L1640), [utils `is_position_in_bounds_not_in_mob_not_colliding`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L294); `info` key counts from an executed reset/step on 1.6.1.

## Craftax params

`EnvParams` (traced since v1.6.0, can differ per call) in [craftax_state.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/craftax_state.py#L112):

| Field | Default | Meaning |
|---|---|---|
| `max_timesteps` | 100000 | Episode ends (`done=True`) at this step |
| `day_length` | 300 | Steps per day/night cycle |
| `always_diamond` | False | Force one diamond into the overworld stone |
| `mob_despawn_distance` | 14 | Mobs farther than this from the player despawn |
| `max_attribute` | 5 | Cap for dexterity, strength, intelligence |
| `god_mode` | False | Full inventory, health floored at 9, free movement and ladders |
| `fractal_noise_angles` | `(None, None, None, None)` | Override Perlin gradient angles (UED level parameters) |

`StaticEnvParams` (shapes; fixed at construction) in [craftax_state.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/craftax_state.py#L127):

| Field | Default |
|---|---|
| `map_size` | (48, 48) |
| `num_levels` | 9 |
| `max_melee_mobs` / `max_passive_mobs` / `max_ranged_mobs` | 3 / 3 / 2 (per floor) |
| `max_mob_projectiles` / `max_player_projectiles` | 3 / 3 |
| `max_growing_plants` | 10 |

## Craftax-Classic params

`EnvParams` in [classic craftax_state.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/envs/craftax_state.py#L78):

| Field | Default |
|---|---|
| `max_timesteps` | 10000 |
| `day_length` | 300 |
| `always_diamond` | True (added v1.4.0; "by default in Craftax-Classic ~35% of worlds have no diamond", [release v1.4.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.4.0)) |
| `zombie_health` / `cow_health` / `skeleton_health` | 5 / 3 / 3 |
| `mob_despawn_distance` | 14 |
| `spawn_cow_chance` | 0.1 |
| `spawn_zombie_base_chance` / `spawn_zombie_night_chance` | 0.02 / 0.1 |
| `spawn_skeleton_chance` | 0.05 |
| `fractal_noise_angles` | `(None, None, None, None)` |

- `StaticEnvParams`: `map_size=(64, 64)`, `max_zombies=3`, `max_cows=3`, `max_skeletons=2`, `max_arrows=3`, `max_growing_plants=10`; the paper gives the same caps ([App. A.1](https://arxiv.org/html/2402.16801v2#A1.SS1)).
