---
title: "Craftax known issues"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, known-issues, gotchas]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax known issues

State as of craftax 1.6.1 and the [issue tracker](https://github.com/MichaelTMatthews/Craftax/issues?q=is%3Aissue) on 2026-10-20 (26 issues, 3 open).

## Open bugs

| Issue | Affects | Problem | Measured impact |
|---|---|---|---|
| [#60](https://github.com/MichaelTMatthews/Craftax/issues/60) (2026-09-24) | Craftax-Classic dynamics | `do_action` rebuilds every zombie, cow and skeleton mask from `health > 0` on each `DO`, reviving despawned mobs and activating unused slots (zombie slots start with health 1); revived mobs are missing from `mob_map` | Random policy, 2,000 episodes: median episode length 136.5 as released vs 155 patched, mean return 1.46 vs 1.56 |
| [#57](https://github.com/MichaelTMatthews/Craftax/issues/57) (2026-09-07) | Symbolic observations of both games, and `render_craftax_text` | `_add_mob_to_map` writes every mob slot without clamping; negative local positions wrap and inactive slots write 0, erasing visible mobs | Craftax-Classic: 87 of 25,600 observations (0.340%) miss a mob; full Craftax not measured; pixel renderers unaffected |
| [#55](https://github.com/MichaelTMatthews/Craftax/issues/55) (2026-09-07) | Install on Python 3.14 | hard dependency on `pygame`, which has no cp314 wheel | install fails |

- Full Craftax uses `jnp.logical_and(mobs.health > 0, mobs.mask)` when attacking ([game_logic_utils.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L32)), so the #60 revival is specific to Classic (code reading, not tested).
- #57 lines: [craftax renderer.py L42-56](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L42), [classic renderer.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/renderer.py); the reporter's fix (`mode="drop"` with off-grid writes) cost about 12% CPU throughput in their test.
- Both fixes change observations or dynamics relative to published results; pin the craftax version and note any local patch.

## Errata (fixed, change results across versions)

| Fixed in | Bug | Source |
|---|---|---|
| 1.5.0 | Any `DO` reset the growth timer of plant slot 0, so the first planted plant (Craftax and Classic) could effectively never ripen; eating an unripe plant also reset its timer | [README Errata](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md), [release v1.5.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.5.0) |
| 1.6.0 | Craftax-Classic: walking into lava gave no health penalty; health could go negative on the last step and add extra penalty | README Errata, [issue #50](https://github.com/MichaelTMatthews/Craftax/issues/50), [release v1.6.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.6.0) |
| 1.4.3 | `observation_space` declared int32 while observations are float32 | [issue #24](https://github.com/MichaelTMatthews/Craftax/issues/24) |
| 1.4.0 | Craftax pixel tiles were 7 px instead of the paper's 10 px (digits unreadable); paper results used symbolic obs and are unaffected | [issue #19](https://github.com/MichaelTMatthews/Craftax/issues/19) |
| 1.4.5 | Texture cache (a pickle tied to the JAX version) was shipped in the wheel | [issue #31](https://github.com/MichaelTMatthews/Craftax/issues/31) |

## API gotchas

- No-auto-reset envs keep stepping after `done` into invalid states; always wrap them ([[Craftax API and wrappers]]).
- The wrappers (`LogWrapper`, `AutoResetEnvWrapper`, `OptimisticResetVecEnvWrapper`, `BatchEnvWrapper`) are not in the `craftax` package; copy them from Craftax_Baselines ([issue #39](https://github.com/MichaelTMatthews/Craftax/issues/39)).
- The README's `make_craftax_env_from_args` does not exist; the function is `make_craftax_env_from_params`.
- `make_craftax_env_from_name("...-AutoReset-v1", auto_reset=False)` raises `ValueError`, and `env.name` of a no-auto-reset env (`...-NoAutoReset-v1`) is not a name the factory accepts (checked on 1.6.1).
- The time limit is not a separate truncation signal: at `max_timesteps` the env returns `done=True` and `info["discount"]=0.0`, the same as death ([environment_bases.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/environment_base/environment_bases.py)).
- With an auto-reset env the returned `state` after `done` is already the new episode; read final achievements from `info["Achievements/*"]`, which are non-zero only on that step.
- `CraftaxSymbolicEnv` and `CraftaxPixelsEnv` generate different worlds from the same key (symbolic splits the key first); Craftax-Classic does not differ ([[Craftax world generation]]).
- Since 1.6.0 `params` are traced, not static, inside `reset`/`step`; custom env code that branches in Python on `params` fields now sees tracers.
- `render_craftax_text` is not jittable or vmappable; index a batched state with `jax.tree_util.tree_map(lambda x: x[i], state)` first ([issue #13](https://github.com/MichaelTMatthews/Craftax/issues/13)).
- Changing reward or `done` in a wrapper: place it inside `LogWrapper` and the reset wrapper ([issue #37](https://github.com/MichaelTMatthews/Craftax/issues/37)).

## Observation and documentation mismatches

- Symbolic values exceed the declared `Box(0, 1)`: enchantment channels are 0/1/2 and intrinsics can reach 2.1 ([[Craftax observations and actions]]).
- [obs_description.md](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/obs_description.md) puts the per-tile light flag at index 83 (code: 82) and lists potions as `n / 10` (code: `sqrt(n) / 10`).
- Paper vs code: 65 achievements in the text vs 67 in code and paper Table 6; creature one-hot of 36 vs 40; light level "a float" vs a binary visibility flag; several creature health and damage values differ ([[Craftax game rules#Source disagreements]]).
- Enum `Achievement` is declared out of value order; `ENTER_DUNGEON = 29` (floor 1) and `ENTER_GNOMISH_MINES = 28` (floor 2) are correct as mapped ([issue #53](https://github.com/MichaelTMatthews/Craftax/issues/53), closed).
- Classic `info["score"]` is a per-episode geometric mean of 0/100 values, not a score over success rates ([[Craftax scoring and evaluation]]).

## Environment and install gotchas

- Textures are cached to `assets/texture_cache.pbz2` inside the installed package on first pixel use, so the package directory must be writable (code reading of [constants.py `load_all_textures`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L1119)).
- `CRAFTAX_RELOAD_TEXTURES` is tested for truthiness of the string, so any non-empty value, including `false`, forces a rebuild.
- A stale or wrong-size texture cache gives shape errors such as `mul got incompatible shapes for broadcasting`; since 1.4.3 the cache is validated and rebuilt, and `CRAFTAX_RELOAD_TEXTURES=true` still forces it ([issue #27](https://github.com/MichaelTMatthews/Craftax/issues/27), [issue #33](https://github.com/MichaelTMatthews/Craftax/issues/33)).
- Importing craftax on a GPU lets JAX preallocate 75% of GPU memory ([issue #28](https://github.com/MichaelTMatthews/Craftax/issues/28)).
- Editable installs need a PEP 660 capable pip/setuptools (setuptools >= 64; README says `pip>=23.0`) ([issue #3](https://github.com/MichaelTMatthews/Craftax/issues/3)).
- The offline human dataset (Google Drive link in the README) loads only with v1.1.0 or earlier; `render_craftax_text` on it failed with `Array(7) is not a valid ItemType` for one user ([issue #16](https://github.com/MichaelTMatthews/Craftax/issues/16)).

## Game-rule surprises

- In Craftax the player cannot step into lava or water (`COLLISION_LAND_CREATURE`); in Craftax-Classic stepping into lava ends the episode.
- Plants grow only on the overworld (`update_plants` touches floor 0 only) and at most 10 grow at once.
- Ladders down stay closed until 8 creatures are killed on the floor.
- Potion colours are re-permuted every episode, so a policy cannot memorise them across episodes.
- More, each checked in code: [[Craftax game rules#Rule surprises]].
