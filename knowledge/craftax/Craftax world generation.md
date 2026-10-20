---
title: "Craftax world generation"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, procedural-generation, seeding]
created: 2026-10-20
---
# Craftax world generation

`reset` builds the whole world, all 9 floors, in one call to `generate_world(rng, params, static_params)` ([world_gen.py L571](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L571)). Nothing is generated later in the episode except creatures, projectiles and plants.

## Pipeline

1. Split the key into 7: one carry key plus 6 keys for `generate_smoothworld`, vmapped over 6 configs (overworld, gnomish mines, troll mines, fire, ice, boss).
2. Split the carry key into 4: one carry key plus 3 keys for `generate_dungeon`, vmapped over 3 configs (dungeon, sewers, vaults).
3. Stack the 9 maps in floor order: overworld, dungeon, gnomish mines, sewers, vaults, troll mines, fire realm, ice realm, graveyard; `state.map` has shape (9, 48, 48).
4. Draw `potion_mapping`, a random permutation of the 6 potion effects.
5. Start the player at the map centre (24, 24) of floor 0, facing up, health/food/drink/energy/mana 9, attributes 1, empty inventory (full inventory in god mode).

- Creature arrays are pre-allocated per floor with fixed capacities (`StaticEnvParams`), because JAX needs static shapes; unspawned creatures are masked ([paper App. A.1](https://arxiv.org/html/2402.16801v2#A1.SS1)).
- World generation is the expensive part of a reset ("9 floors with multiple passes of Perlin Noise and room generation"), which motivates optimistic resets ([paper App. C](https://arxiv.org/html/2402.16801v2#A3)); see [[Craftax API and wrappers]].

## Smooth floors (Perlin noise)

`generate_smoothworld` ([world_gen.py L360](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L360)) thresholds fractal Perlin noise, adapted from [perlin-numpy](https://github.com/pvigier/perlin-numpy) ([util/noise.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/noise.py)).

- Water noise above `water_threshold` becomes the sea block (water, or lava on the fire floor); above `sand_threshold` the coast block.
- Mountain noise above 0.7 becomes the mountain block (stone; walls on the graveyard); tunnels where a second noise exceeds 0.8; caves where mountain > 0.85 and water > 0.4.
- Distance maps around the spawn point push water and mountains away from the player.
- Trees (or stalagmites, fire trees, ice shrubs, graves) where tree noise and a uniform draw pass their thresholds.
- Lava where mountain > 0.85 and tree noise > 0.7.
- Ores replace mountain blocks independently per tile with these probabilities ([world_gen_configs.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen_configs.py)):

| Floor | Coal | Iron | Diamond | Sapphire | Ruby | Default light |
|---|---|---|---|---|---|---|
| Overworld | 0.03 | 0.02 | 0.001 | 0 | 0 | 1.0 (plus day/night) |
| Gnomish mines | 0.04 | 0.02 | 0.005 | 0.0025 | 0.0025 | 0.0 |
| Troll mines | 0.04 | 0.03 | 0.01 | 0.01 | 0.01 | 0.0 |
| Fire realm | 0.05 | 0 | 0 | 0 | 0.025 | 1.0 |
| Ice realm | 0 | 0 | 0.005 | 0.02 | 0 | 0.0 |
| Graveyard | mossy walls and graves instead of ores | | | | | 0.0 |

- One down ladder and one up ladder per floor at random valid tiles; the overworld has no up ladder and the graveyard has neither; dark floors are lit in a 9x9 area around the up ladder.
- `EnvParams.always_diamond=True` forces one diamond into overworld stone; default False in Craftax.

## Dungeon floors

`generate_dungeon` ([world_gen.py L76](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L76)):

- 8 rooms of side 5-10 placed on a grid of 16x16 chunks and joined by paths.
- Every room has a torch in each of its 4 corners and one chest; a fountain with probability 0.5 (sewers use water instead).
- The first room holds the floor's special block: none (dungeon), ice enchantment table (sewers), fire enchantment table (vaults).
- About 10% of walls are mossy; dungeon floors are fully lit.

## Seeding and determinism

- Same key, same world: two resets of `Craftax-Symbolic-v1` with `PRNGKey(0)` gave identical observations (checked on 1.6.1).
- `CraftaxSymbolicEnv*.reset_env` splits the key before `generate_world`; `CraftaxPixelsEnv*.reset_env` does not, so the same key gives a different world in the pixel env (checked: maps differ); in Craftax-Classic symbolic and pixel envs give the same map for the same key.
- In an auto-reset env the next world comes from a key split inside `step`, so every episode gets a new world.
- To train on one fixed world the maintainer suggests setting `rng = jax.random.PRNGKey(0)` at the top of `generate_world` ([issue #35](https://github.com/MichaelTMatthews/Craftax/issues/35)); without editing the package, wrap a no-auto-reset env and reset with a constant key.
- Each step stores a fresh key in `state.state_rng` (used for the night-noise texture).

## Level parameters for UED

- `EnvParams.fractal_noise_angles` (4 entries, default `None`) overrides the Perlin gradient angles of the water, mountain, tunnel and tree noise ([noise.py `override_angles`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/noise.py)).
- The paper's ACCEL "Noise" mutation adds uniform noise to these angle vectors; "Swap" and "Restricted Swap" edit overworld tiles directly ([paper §4.5](https://arxiv.org/html/2402.16801v2#S4.SS5)).
- Because the full state is one pytree, UED methods can edit levels directly ([paper §2.4](https://arxiv.org/html/2402.16801v2#S2.SS4)); hooks for [JaxUED](https://github.com/DramaCow/jaxued) arrived in v1.1.0 ([release v1.1.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.1.0)).

## Craftax-Classic

- One 64x64 map, player at the centre, Perlin noise in place of Crafter's simplex noise, "qualitatively similar results to Crafter" ([paper App. A.2](https://arxiv.org/html/2402.16801v2#A1.SS2); [classic world_gen.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/world_gen.py)).
- `always_diamond` defaults to True; with it off, 63 of 200 generated worlds (31.5%) had no diamond (run on 1.6.1); the v1.4.0 release note says about 35%.
