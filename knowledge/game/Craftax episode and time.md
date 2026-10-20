---
title: "Craftax episode and time"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, game-mechanics, episode, day-night]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax episode and time

Episode start and end, what one step does, the day/night cycle, sleeping and resting, and what the player sees, in full Craftax at commit [`c3c2e0d`](https://github.com/MichaelTMatthews/Craftax/tree/c3c2e0d038c4e641f9481320c158f457f30c28f3) (craftax 1.6.1). Craftax-Classic: [[Craftax-Classic rules]].

## Episode start

| Item | Value at reset | Source |
|---|---|---|
| World | all 9 floors of 48 x 48 generated at once and kept for the whole episode | [`generate_world`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L571) |
| Player position | floor 0, tile (24, 24) (map centre), forced to grass | same, [`generate_smoothworld`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L511) |
| Facing | up (`Action.UP`) | same |
| Health, food, drink, energy, mana | 9 each | same |
| Dexterity, strength, intelligence | 1 each; XP 0 | same |
| Inventory | empty (full in `god_mode`) | [`get_new_empty_inventory`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L14) |
| Kill counters | floor 0 starts at 10 ("First ladder starts open"), floors 1-8 at 0 | `generate_world` |
| Potion effects | random permutation of the 6 effects over the 6 colours | `generate_world` |
| Boss spawn counter | 7 (`BOSS_FIGHT_SPAWN_TURNS`) | `generate_world` |
| Time | `timestep` 0, `light_level` = light(0) = 0.797 | `generate_world` |

- World generation details: [[Craftax world generation]] and [[Craftax floors and progression]].

## Episode end

The episode is done after a step when any of these holds ([`is_game_over`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L4)):

| Condition | Rule |
|---|---|
| Death | `player_health <= 0` (health is clipped to [0, max] at the end of the step) |
| Win | `boss_progress >= 8` (`num_levels - 1`), see [[Craftax boss fight]] |
| Time cap | `timestep >= max_timesteps`, default 100,000 ([`EnvParams`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/craftax_state.py#L112)); paper App. D: "truncated" at 100,000 |

- Health falls only through creature melee hits, projectiles, the health drain when food, drink or energy is 0, and the -3 health potion ([[Craftax survival stats]], [[Craftax combat]]).
- Lava and water cannot kill in full Craftax: the player cannot step onto them ([`move_player`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2021) uses `COLLISION_LAND_CREATURE`); checked on 1.6.1 (moving up into lava left the player in place with 9 health).
- The time cap gives the same `done` and discount 0 as death ([[Craftax known issues]]).

## Step order

One `step` applies one action; while sleeping or resting the action is replaced by `NOOP`. Order inside [`craftax_step`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L3006):

1. Replace the action by `NOOP` if `is_sleeping` or `is_resting`.
2. `change_floor` (`DESCEND` / `ASCEND`).
3. `do_crafting` (all `MAKE_*`).
4. `do_action` (`DO`: attack, mine, drink, eat, open chest, hit the necromancer).
5. `place_block` (`PLACE_*`).
6. `shoot_projectile` (`SHOOT_ARROW`), `cast_spell`, `drink_potion`, `read_book`, `enchant`.
7. `boss_logic` (boss spawn counter -1 on floor 8; `DEFEAT_NECROMANCER`).
8. `level_up_attributes`.
9. `move_player` (`LEFT`/`RIGHT`/`UP`/`DOWN`).
10. `update_mobs`: melee, passive, ranged creatures, then creature projectiles, then player projectiles.
11. `spawn_mobs`.
12. `update_plants`.
13. `update_player_intrinsics` (sleep/rest state, food, drink, energy, health, mana counters).
14. `clip_inventory_and_intrinsics` (inventory <= 99; stats to [0, max]).
15. `calculate_inventory_achievements`.
16. Reward = sum of newly unlocked achievement rewards + 0.1 x (health after - health before) ([[Craftax scoring and evaluation]]).
17. `timestep + 1`; `light_level` recomputed for the new timestep.

- Every action is legal at every step; an action whose conditions fail does nothing and the step still passes ([paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3)).
- Only the current floor is simulated: creatures, projectiles and spawns of other floors are frozen until the player returns ([`update_mobs`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1091) and `spawn_mobs` index `state.player_level` only).
- Plants and all player counters advance every step whatever the floor.

## Day and night

- `light_level = 1 - |cos(pi * ((t / day_length) mod 1 + 0.3))|^3`, `day_length = 300` ([`calculate_light_level`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L349), [`EnvParams`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/craftax_state.py#L112)).
- Values over one 300-step day, computed from the formula:

| t mod 300 | 0 | 30 | 60 | 90 | 120 | 150 | 180 | 210 | 240 | 270 |
|---|---|---|---|---|---|---|---|---|---|---|
| light | 0.797 | 0.970 | 1.000 | 0.970 | 0.797 | 0.470 | 0.140 | 0.000 | 0.140 | 0.470 |

- Brightest at t mod 300 = 60, darkest at 210; light < 0.5 on 42% of steps and < 0.1 on 17%.
- Night raises the floor-0 melee spawn chance to `0.02 + 0.1 (1 - light)^2` (up to 0.12); the night term is 0 on floors 1-8 ([`FLOOR_MOB_SPAWN_CHANCE`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L159)).
- The pixel renderer darkens and adds noise to the view only on floor 0 (`daylight` forced to 1.0 elsewhere); the symbolic map is not darkened, it only carries `light_level` as a scalar ([renderer.py L526-560](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L526)).
- Night has no other effect: creatures, stats and visibility masks ignore it.

## Sleeping

- `SLEEP` starts sleep only when `energy < max_energy` ([`update_player_intrinsics`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1829)).
- While asleep every action is replaced by `NOOP`.
- The player wakes when `energy >= max_energy` (unlocks `WAKE_UP`), when hit by a melee creature (also unlocks `WAKE_UP`), or when hit by a projectile (no achievement) ([`update_mobs`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1203)).
- Daylight does not wake the player.
- Effects while asleep: hunger and thirst counters grow at half rate, energy +1 every 11 steps, health recovery counter +2 per step (or -0.5 if a need is unmet), mana counter +2 per step, melee damage taken x3.5 (projectile damage unchanged); exact counters in [[Craftax survival stats]].
- The pixel view is dimmed to 50% and the player sprite changes; the symbolic observation has an `is_sleeping` flag ([renderer.py L562](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L562)).

## Resting

- `REST` starts rest only when `health < max_health`.
- While resting every action is replaced by `NOOP`; rates are unchanged (resting is not sleeping).
- Rest ends when health reaches max, when food or drink is 0, or when hit by a melee creature or a projectile; energy reaching 0 does not end it.

## What the player sees

- A 9 x 11 tile window (9 rows, 11 columns) centred on the player (`OBS_DIM = (9, 11)`, [constants.py L15](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L15)); tiles outside the map show `OUT_OF_BOUNDS`.
- Each floor has a light map; a tile is visible in the symbolic observation only if its light > 0.05; invisible tiles have every block, item and creature channel zeroed ([renderer.py L116-127](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L116)).
- Pixel observations multiply each tile's colour by its light value.
- Floors 0, 1, 3, 4 and 6 are fully lit (light 1.0); floors 2, 5, 7 and 8 start dark (light 0.0) except a 9 x 9 patch around the arrival ladder and glow around lava on floors 2 and 5 ([[Craftax floors and progression]]).
- A placed torch adds `clip(1 - d / 5, 0, 1)` light (d = Euclidean distance) over a 9 x 9 square, so tiles with d < 4.75 become visible ([`TORCH_LIGHT_MAP`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L592), [`place_block`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L956)).
- The inventory, stats, attributes, direction, enchantments, spells, floor index, ladder-open flag and boss-vulnerable flag are always observed; layout in [[Craftax observations and actions]].

## What persists

- Each floor keeps its map, items (torches, ladders), light map, creatures, projectiles, chest-opened flag and kill counter when the player leaves and returns.
- Creatures on a floor the player has left keep their positions and health; on return the ones at Manhattan distance >= 14 despawn on the first step.
- The player carries inventory, stats, attributes, XP, spells, enchantments and achievements between floors.
- Plants (floor 0) keep growing while the player is on other floors.

## God mode

- `EnvParams.god_mode = True` gives the full inventory, health never below 9, movement through anything, and ladders usable anywhere ([`move_player`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2027), [`change_floor`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2422), [`clip_inventory_and_intrinsics`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L381)); default False.
