---
title: "Craftax blocks and resources"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, blocks, resources]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax blocks and resources

Every `BlockType` (37 values) and `ItemType` (5 values) of full Craftax, with movement, mining and spawn rules, from [constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L28), [`do_action`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L167) and [world_gen_configs.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen_configs.py) at commit `c3c2e0d`.

## Movement rules

- Solid blocks (`SOLID_BLOCKS`): stone, tree, coal, iron, diamond, crafting table, furnace, plant, ripe plant, wall, mossy wall, stalagmite, ruby, sapphire, chest, fountain, fire tree, both enchantment tables, the three graves, necromancer ([constants.py L366](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L366)).
- The player and land creatures may enter any non-solid tile except water and lava, and never a tile holding a creature or the player ([`is_position_in_bounds_not_in_mob_not_colliding`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L294)).
- Flying creatures may also enter water and lava; amphibians water but not lava; aquatic creatures only water ([[Craftax creatures]]).
- Projectiles stop at solid blocks and fly over water and lava.
- Ice shrub, darkness, sand, grass, path, fire grass and ice grass are walkable; the ice shrub is not in `SOLID_BLOCKS`, so the player walks through it (checked on 1.6.1).

## Mining with DO

`DO` acts on the facing tile. If a creature stands there it is attacked and the block is untouched. Pickaxe levels: 1 wood, 2 stone, 3 iron, 4 diamond.

| Block | Needs | Gives | Becomes |
|---|---|---|---|
| Tree | nothing | 1 wood | grass |
| Fire tree | nothing | 1 wood | fire grass |
| Ice shrub | nothing | 1 wood | ice grass |
| Stone | pickaxe >= 1 | 1 stone | path |
| Stalagmite | pickaxe >= 1 | 1 stone | path |
| Coal | pickaxe >= 1 | 1 coal | path |
| Iron | pickaxe >= 2 | 1 iron | path |
| Diamond | pickaxe >= 3 | 1 diamond | path |
| Sapphire | pickaxe >= 4 | 1 sapphire | path |
| Ruby | pickaxe >= 4 | 1 ruby | path |
| Crafting table | nothing | nothing (no refund) | path |
| Furnace | nothing | nothing (no refund) | path |
| Chest | nothing | loot ([[Craftax items and inventory#Chest loot]]) | path |
| Grass | nothing | sapling with probability 0.1 | grass (unchanged) |
| Water, fountain | nothing | drink +1 | unchanged |
| Ripe plant | nothing | food +4 | plant |
| Necromancer | boss vulnerable | boss progress +1 ([[Craftax boss fight]]) | unchanged |

- Trees are consumed: one `DO` gives one wood and turns the tree into grass (checked on 1.6.1); wood is finite on each floor.
- Wall, mossy wall, graves, enchantment tables, plants, sand, lava and path cannot be mined; `DO` on them does nothing.
- Each successful `DO` yields one unit; there is no mining time or tool durability.

## Block reference

| ID | Block | Solid | Where it appears | Notes |
|---|---|---|---|---|
| 0 | `INVALID` | no | never | placeholder |
| 1 | `OUT_OF_BOUNDS` | n/a | outside the map in observations | also the "no ore" filler in the overworld ore list |
| 2 | `GRASS` | no | floor 0 ground | sapling source; spawn tile; only tile for saplings |
| 3 | `WATER` | no (impassable) | floor 0 lakes, floors 2 and 5 pools, floor 3 water tiles, floor 7 sea and pockets | drink; stone can be placed on it; deep things spawn here |
| 4 | `STONE` | yes | mountains of floors 0, 2, 5, 6, 7; player-placed | ore host |
| 5 | `TREE` | yes | floor 0, on grass | wood |
| 6 | `WOOD` | n/a | never on the map | used only as the inventory icon ([renderer.py L723](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L723)) |
| 7 | `PATH` | no | mined tiles, floor 0 tunnels and caves, dungeon floors, mine floors, graveyard floor | spawn tile; ladders on floors 0-5 and 8 |
| 8 | `COAL` | yes | stone on floors 0, 2, 5, 6 | |
| 9 | `IRON` | yes | stone on floors 0, 2, 5 | |
| 10 | `DIAMOND` | yes | stone on floors 0, 2, 5, 7 | |
| 11 | `CRAFTING_TABLE` | yes | player-placed | destroyed by `DO` or by a creature projectile |
| 12 | `FURNACE` | yes | player-placed | destroyed by `DO` or by a creature projectile |
| 13 | `SAND` | no | floor 0 coasts | not a spawn tile; torches allowed |
| 14 | `LAVA` | no (impassable) | floors 0, 2, 5, 6 in mountains; floor 6 sea | glows on floors 2 and 5; stone can be placed on it |
| 15 | `PLANT` | yes | player-placed sapling | grows to ripe ([[Craftax farming and food]]) |
| 16 | `RIPE_PLANT` | yes | grown sapling | food +4 |
| 17 | `WALL` | yes | dungeon walls (floors 1, 3, 4), graveyard walls | unbreakable |
| 18 | `DARKNESS` | no | dungeon tiles with no non-wall 4-neighbour | unreachable filler outside walls |
| 19 | `WALL_MOSS` | yes | 10% of dungeon walls, 10% of graveyard walls | unbreakable |
| 20 | `STALAGMITE` | yes | floors 2 and 5, on path | stone |
| 21 | `SAPPHIRE` | yes | stone on floors 2, 5, 7 | needs diamond pickaxe |
| 22 | `RUBY` | yes | stone on floors 2, 5, 6 | needs diamond pickaxe |
| 23 | `CHEST` | yes | one per room on floors 1, 3, 4 | opened once |
| 24 | `FOUNTAIN` | yes | 50% of rooms on floors 1 and 4 | drink |
| 25 | `FIRE_GRASS` | no | floor 6 ground | spawn tile |
| 26 | `ICE_GRASS` | no | floor 7 ground | spawn tile |
| 27 | `GRAVEL` | no | never generated | |
| 28 | `FIRE_TREE` | yes | floor 6, on fire grass | wood |
| 29 | `ICE_SHRUB` | no | floor 7, on ice grass | wood; walkable |
| 30 | `ENCHANTMENT_TABLE_FIRE` | yes | floor 4 (vaults), first room | ruby enchanting |
| 31 | `ENCHANTMENT_TABLE_ICE` | yes | floor 3 (sewers), first room | sapphire enchanting |
| 32 | `NECROMANCER` | yes | floor 8, map centre (24, 24) | the boss |
| 33-35 | `GRAVE`, `GRAVE2`, `GRAVE3` | yes | floor 8, about 5% of path tiles | boss-wave spawn points |
| 36 | `NECROMANCER_VULNERABLE` | n/a | rendering only | drawn instead of the necromancer while it can be hit |

## Items layer

A second map per floor holds items on walkable tiles ([`ItemType`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L68)).

| ID | Item | Meaning |
|---|---|---|
| 0 | `NONE` | empty |
| 1 | `TORCH` | placed torch or pre-placed dungeon corner torch; permanent, cannot be picked up |
| 2 | `LADDER_DOWN` | `DESCEND` here once the floor is cleared |
| 3 | `LADDER_UP` | `ASCEND` here |
| 4 | `LADDER_DOWN_BLOCKED` | pixel rendering of a down ladder while the floor has < 8 kills ([renderer.py L267](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L267)) |

- A tile with an item cannot receive a table, furnace, torch or plant.

## Resource spawn rules per floor

Ores replace the floor's ore-host block independently per tile, in this order (coal, iron, diamond, sapphire, ruby), each with the listed probability ([world_gen_configs.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen_configs.py)).

| Floor | Coal | Iron | Diamond | Sapphire | Ruby | Wood source | Water source |
|---|---|---|---|---|---|---|---|
| 0 Overworld | 0.03 | 0.02 | 0.001 | 0 | 0 | trees | lakes |
| 1 Dungeon | 0 | 0 | 0 | 0 | 0 | none | fountains |
| 2 Gnomish mines | 0.04 | 0.02 | 0.005 | 0.0025 | 0.0025 | none (stalagmites give stone) | pools |
| 3 Sewers | 0 | 0 | 0 | 0 | 0 | none | water tiles |
| 4 Vaults | 0 | 0 | 0 | 0 | 0 | none | fountains |
| 5 Troll mines | 0.04 | 0.03 | 0.01 | 0.01 | 0.01 | none | pools |
| 6 Fire realm | 0.05 | 0 | 0 | 0 | 0.025 | fire trees | none |
| 7 Ice realm | 0 | 0 | 0.005 | 0.02 | 0 | ice shrubs | sea and pockets |
| 8 Graveyard | 0 | 0 | 0 | 0 | 0 | none | none (not needed) |

- Dungeon floors (1, 3, 4) have no ores; their chests are the only source of ores there ([[Craftax items and inventory#Chest loot]]).
- With `EnvParams.always_diamond = True` one random overworld stone becomes diamond; default False in Craftax ([`generate_smoothworld`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L483)).
- Saplings come only from grass (floor 0).
- Terrain thresholds and dungeon layout: [[Craftax floors and progression]] and [[Craftax world generation]].
