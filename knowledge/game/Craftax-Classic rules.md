---
title: "Craftax-Classic rules"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, craftax-classic, crafter]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax-Classic rules

The complete rule set of Craftax-Classic, the JAX rewrite of Crafter, and every difference from full Craftax, from [craftax_classic/game_logic.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py), [constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/constants.py), [craftax_state.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/envs/craftax_state.py) and [world_gen.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/world_gen.py) at commit `c3c2e0d`. Paper: [§3.1 and App. A](https://arxiv.org/html/2402.16801v2#S3.SS1).

## Summary of differences

| Aspect | Craftax-Classic | Full Craftax |
|---|---|---|
| World | 1 floor, 64 x 64 | 9 floors, 48 x 48 |
| View | 7 x 9 tiles | 9 x 11 tiles |
| Actions | 17 (IDs 0-16) | 43 |
| Achievements | 22, +1 each (max 22) | 67, tiered 1/3/5/8 (max 226) |
| Episode cap | 10,000 steps | 100,000 steps |
| Blocks | 17 types (IDs 0-16) | 37 types |
| Creatures | zombie, cow, skeleton | 19 creatures + necromancer |
| Stats | health, food, drink, energy, max 9 each | + mana; maxima depend on attributes |
| Attributes, XP, mana, magic, potions, enchanting, armour, bow, torches, darkness, chests, ladders, rest | none | present |
| Inventory cap | 9 per item | 99 per item |
| Tools | separate wood/stone/iron pickaxe and sword counts | one pickaxe level and one sword level (0-4) |
| Lava | walkable; entering it sets health to 0 and ends the episode | impassable, harmless |
| Damage | integers, no armour | float vectors with defences |

## Episode

- Start: tile (32, 32) forced to grass, facing up, health/food/drink/energy 9, empty inventory ([`generate_world`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/world_gen.py)).
- End: health <= 0, standing on lava, or `timestep >= 10,000` ([`is_game_over`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py#L5)).
- Step order: sleep turns the action into `NOOP`; crafting; `DO`; placing; movement; creatures; spawning; plants; intrinsics; inventory cap at 9; lava check (health 0 on lava, health floored at 0); reward ([`craftax_step`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py#L1655)).
- Reward: number of newly unlocked achievements + 0.1 x health change.
- Day/night: same formula and 300-step day as Craftax; night raises zombie spawning and darkens the pixel view (greyed, noisy below light 0.5).

## World

- One Perlin-noise map ([classic world_gen.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/world_gen.py)); proximity to the centre uses Manhattan distance / 5, capped at 1.
- Water where water noise + proximity - 1 > 0.7; sand where 0.6 < value < 0.75 and not water.
- Stone where mountain noise + 0.05 + proximity - 1 > 0.7; tunnels (path) where a tunnel noise > 0.8 inside mountains; caves (path) where mountain > 0.85 and water > 0.4.
- Ores on stone: coal 0.04, iron 0.03, diamond 0.005 only where mountain > 0.8 (Craftax overworld: 0.03, 0.02, 0.001).
- Trees on grass where tree noise > 0.5 and a uniform draw > 0.8; lava where mountain > 0.85 and tree noise > 0.7.
- `always_diamond` defaults to True: one random stone tile becomes diamond.

## Blocks and movement

- Solid: water, stone, tree, coal, iron, diamond, table, furnace, plant, ripe plant ([constants.py L91](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/constants.py#L91)).
- The player can step on grass, sand, path and lava; creatures cannot enter lava.
- Mining: tree gives wood and becomes grass (checked on 1.6.1); stone and coal need a wood pickaxe; iron needs a stone pickaxe; diamond needs an iron pickaxe; mined blocks become path.
- Tool checks look at the specific pickaxe, not a tier: with only a stone pickaxe, stone cannot be mined (checked on 1.6.1).
- `DO` on grass gives a sapling with probability 0.1; on water drink +1; on a ripe plant food +4.
- `DO` cannot remove tables or furnaces; only skeleton arrows destroy them.

## Crafting and placing

| Action | Inputs | Station |
|---|---|---|
| `PLACE_TABLE` | 2 wood | - |
| `PLACE_FURNACE` | 1 stone | - (no table needed) |
| `PLACE_STONE` | 1 stone | - |
| `PLACE_PLANT` | 1 sapling | - |
| `MAKE_WOOD_PICKAXE`, `MAKE_WOOD_SWORD` | 1 wood | table |
| `MAKE_STONE_PICKAXE`, `MAKE_STONE_SWORD` | 1 wood, 1 stone | table |
| `MAKE_IRON_PICKAXE`, `MAKE_IRON_SWORD` | 1 wood, 1 stone, 1 iron, 1 coal | table and furnace |

- Table and furnace go on any non-solid tile (grass, sand, path, lava); stone on water or any non-solid tile; plants on grass ([`place_block`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py#L586)).
- Stations count within the 8 surrounding tiles, as in Craftax.
- Recipes can be repeated: each craft adds 1 to that tool's count, up to 9.
- Sword damage = `max(1, 2 x wood_swords, 3 x stone_swords, 5 x iron_swords)` using the counts, so extra copies stack (3 wood swords deal 6, checked on 1.6.1) ([`get_player_attack_damage`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py#L43)).

## Stats

- Health, food, drink, energy: start 9, max 9; the same hidden counters and thresholds as Craftax with dexterity 1 (food -1 per 26 steps, drink per 21, energy per 31, health +1 per 26 or -1 per 16 steps, halves and doubles while asleep) ([`update_player_intrinsics`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/game_logic.py#L1237)).
- Sleep starts when energy < 9 and ends at energy 9 (`WAKE_UP`) or when a zombie hits (also `WAKE_UP`) or an arrow hits (no achievement).
- Eating a cow: food + 6 (max 9); ripe plant: +4.
- Plants: 10 growth slots never freed, ripe at age 600, re-grow after eating, as in Craftax.

## Creatures

| Creature | Health | Attack | Spawn chance per step | Spawn tiles (Manhattan distance) | Max alive |
|---|---|---|---|---|---|
| Cow | 3 | none | 0.1 | grass, > 3 and < 14 | 3 |
| Zombie | 5 | 2 melee (7 on a sleeping player), cooldown 5 | 0.02 + 0.1 (1 - light)^2 | grass or path, > 9 and < 14 | 3 |
| Skeleton | 3 | arrow, 2 damage, cooldown 4 | 0.05 | path only, > 9 and < 14 | 2 |

- Values are `EnvParams` and `StaticEnvParams` defaults ([craftax_state.py L78-107](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/envs/craftax_state.py#L78)); at most 3 arrows exist.
- Zombie AI equals the Craftax melee AI (chase with probability 0.75 within Manhattan distance 10, attack when adjacent).
- Skeleton AI equals the Craftax ranged AI except that it approaches only at distance >= 10 (Craftax: 6).
- Cows move like passive creatures (stay half the time).
- All creatures despawn at Manhattan distance >= 14; spawn distances are Manhattan (Craftax: Euclidean).
- An arrow hits the player only when its next tile is the player's; it destroys a table or furnace in its path.
- Killing a zombie or skeleton unlocks its achievement; killing a cow gives food and `EAT_COW`; any weapon kill counts (melee is the only weapon).
- Open bug: every `DO` rebuilds creature masks from `health > 0`, reviving despawned creatures ([issue #60](https://github.com/MichaelTMatthews/Craftax/issues/60); [[Craftax known issues]]).

## Achievements

All 22 pay +1 and unlock at the moment of the event (not from an inventory check):

| ID | Achievement | ID | Achievement |
|---|---|---|---|
| 0 | `COLLECT_WOOD` | 11 | `EAT_PLANT` |
| 1 | `PLACE_TABLE` | 12 | `DEFEAT_SKELETON` |
| 2 | `EAT_COW` | 13 | `MAKE_STONE_PICKAXE` |
| 3 | `COLLECT_SAPLING` | 14 | `MAKE_STONE_SWORD` |
| 4 | `COLLECT_DRINK` | 15 | `WAKE_UP` |
| 5 | `MAKE_WOOD_PICKAXE` | 16 | `PLACE_FURNACE` |
| 6 | `MAKE_WOOD_SWORD` | 17 | `COLLECT_COAL` |
| 7 | `PLACE_PLANT` | 18 | `COLLECT_IRON` |
| 8 | `DEFEAT_ZOMBIE` | 19 | `COLLECT_DIAMOND` |
| 9 | `COLLECT_STONE` | 20 | `MAKE_IRON_PICKAXE` |
| 10 | `PLACE_STONE` | 21 | `MAKE_IRON_SWORD` |

- Same names and IDs as full Craftax 0-21 ([classic constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/constants.py)).
- PPO-RNN learns all but `COLLECT_DIAMOND` and `EAT_PLANT` reliably in 1B steps ([paper App. F.3](https://arxiv.org/html/2402.16801v2#A6.SS3)).
