---
title: "Craftax floors and progression"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, floors, ladders]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax floors and progression

The 9 floors of full Craftax, how they are built, and how ladders, kills and XP move the player down, from [world_gen.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py), [world_gen_configs.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen_configs.py) and [`change_floor`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2422) at commit `c3c2e0d`. Generation pipeline and seeding: [[Craftax world generation]].

## Floors at a glance

| Floor | Name | Generator | Lit | Passive / melee / ranged | Food | Water | Special |
|---|---|---|---|---|---|---|---|
| 0 | Overworld | smooth (Perlin) | yes, plus day/night | cow / zombie / skeleton | cows, plants | lakes | trees; down ladder open from the start |
| 1 | Dungeon | rooms | yes | snail / orc soldier / orc mage | snails | fountains | first chest gives the bow |
| 2 | Gnomish Mines | smooth | no | bat / gnome warrior / gnome archer | bats | pools | all five ores; lava glow |
| 3 | Sewers | rooms | yes | snail / lizard / kobold | snails | water tiles | ice enchantment table; first chest gives a book |
| 4 | Vaults | rooms | yes | snail / knight / knight archer | snails | fountains | fire enchantment table; first chest gives a book |
| 5 | Troll Mines | smooth | no | bat / troll / deep thing | bats | pools | richest ores; deep things in water |
| 6 | Fire Realm | smooth | yes | bat / pigman / fire elemental | bats | none | lava sea; coal and ruby |
| 7 | Ice Realm | smooth | no | none / frost troll / ice elemental | none | sea | sapphire and diamond |
| 8 | Graveyard | smooth | no | none / boss waves / boss waves | none needed | none needed | necromancer; no ladders |

- Floor order is the stacking order in [`generate_world`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L591): the six smooth configs (overworld, gnomish mines, troll mines, fire, ice, graveyard) interleaved with the three dungeon configs (dungeon, sewers, vaults).
- Creature mapping: [`FLOOR_MOB_MAPPING`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L142); details in [[Craftax creatures]].
- Every floor is 48 x 48 tiles; tile (24, 24) is the "centre" used by every smooth floor's generator.

## Smooth floors

`generate_smoothworld` ([L360](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L360)) thresholds four Perlin noise maps; the distance to the map centre is added so that water and mountains thin out near (24, 24).

| Step | Rule |
|---|---|
| Sea | water noise + `min(d / 5, 1)` - 1 > water threshold (0.7; 0.5 on floors 6 and 7) |
| Coast | water value > 0.6 and not sea (no effect when the sea threshold is 0.5) |
| Mountain | mountain noise + 0.05 + proximity - 1 > 0.7 (proximity `min(d / 5, 1)` on 0 and 6, `min(d / 17, 1.5)` on 2, 5, 7, `min(d / 10, 10)` on 8) |
| Tunnels | mountain and a tunnel noise (or its transpose) > 0.8 |
| Caves | mountain value > 0.85 and water value > 0.4 |
| Trees | tree noise > 0.5 and a uniform draw > 0.8 (0.4 on floor 7; on floor 8 any tree noise and draw > 0.95), only on the floor's tree host |
| Ores | per-tile draws on the ore host ([[Craftax blocks and resources#Resource spawn rules per floor]]) |
| Lava | mountain value > 0.85 and tree noise > 0.7 |

| Config | Ground | Sea | Coast | Mountain | Tunnels / caves | "Trees" | Lava cells | Ladder tile | Light |
|---|---|---|---|---|---|---|---|---|---|
| Overworld (0) | grass | water | sand | stone | path / path | tree on grass | lava | path | 1.0 |
| Gnomish mines (2) | path | water | path | stone | stone / stone | stalagmite on path | lava | path | 0.0 |
| Troll mines (5) | path | water | path | stone | stone / stone | stalagmite on path | lava | path | 0.0 |
| Fire realm (6) | fire grass | lava | sand | stone | stone / stone | fire tree on fire grass | lava | fire grass | 1.0 |
| Ice realm (7) | ice grass | water | ice grass | stone | stone / stone | ice shrub on ice grass | water | ice grass | 0.0 |
| Graveyard (8) | path | path | path | wall | wall / wall | grave on path | wall | path | 0.0 |

- Configs: [world_gen_configs.py L37-259](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen_configs.py#L37).
- On floor 0 tunnels and caves are the only path tiles, so the down ladder is always inside mountains.
- On floors 2 and 5 the open cave floor is path, with water pools and stalagmites; ores lie anywhere inside the stone and are reached by mining from the cave edges.
- On floor 6 lava replaces the sea, giving islands of fire grass in lava; no water exists.
- On floor 7 the "lava" cells are water, and ice shrubs are dense (uniform threshold 0.4).
- On floor 8 the necromancer replaces the centre tile; mossy walls (10% of walls) and three grave textures (graves split evenly into `GRAVE`, `GRAVE2`, `GRAVE3`) come from the ore step.

## Dungeon floors

`generate_dungeon` ([L76](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L76)):

- The map starts as wall and is split into 3 x 3 chunks of 16 x 16; 8 rooms go into 8 distinct chunks at a random offset of 0-10 tiles.
- Room sides are 5-9 tiles each (`randint(5, 10)`, upper bound exclusive); rooms may overlap or be cut by the map edge.
- Each room gets a torch in each of its 4 corners, one chest on a random interior tile, and with probability 0.5 a fountain on a random interior tile (sewers: a water tile instead).
- Rooms are joined by L-shaped corridors: room i connects to a random room already in the network (room 7 starts it), horizontal leg first.
- The floor's special block sits at the first room's corner + (2, 2): nothing on floor 1, the ice enchantment table on floor 3, the fire enchantment table on floor 4.
- 10% of path tiles without items are replaced by the rare block: path on floors 1 and 4 (no change), water on floor 3.
- 10% of walls become mossy walls; walls with no non-wall 4-neighbour become `DARKNESS`.
- Light is 1.0 everywhere; the down and up ladders sit on random path tiles.

## Light on dark floors

- Floors 2, 5, 7 and 8 start with light 0.0, so nothing is visible without light ([[Craftax episode and time#What the player sees]]).
- A 9 x 9 torch-shaped light patch surrounds the arrival tile (the up-ladder position) on every smooth floor ([L549](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L549)).
- Lava cells glow on floors 2 and 5: +1.0 on the cell, +0.7 on its 4 neighbours, +0.2 on diagonals, capped at 1.
- Everything else must be lit by placing torches (1 wood + 1 coal makes 4).

## Ladders

| Rule | Detail |
|---|---|
| Placement | one down ladder and one up ladder per floor on a random valid tile; floor 0 has no up ladder; floor 8 has neither |
| `DESCEND` | player stands on the down ladder, the floor's kill counter is >= 8 (`MONSTERS_KILLED_TO_CLEAR_LEVEL`), and floor < 8 |
| `ASCEND` | player stands on the up ladder and floor > 0; no kill requirement |
| Arrival | descending lands on the next floor's up-ladder tile; ascending lands on the previous floor's down-ladder tile |
| Floor 0 | the kill counter starts at 10, so its ladder is open from the start |
| Kills | melee and ranged creatures killed on that floor by any means; passive kills do not count |
| Display | the pixel view draws a blocked ladder until 8 kills; the symbolic observation has a ladder-open flag |

- Floor 8 stores an up-ladder position (the arrival tile) but no ladder item, so there is no way back up ([[Craftax boss fight]]).
- While a floor has fewer than 8 kills, its melee and ranged spawn chances are tripled ([[Craftax creatures#Spawning]]).

## Progress rewards

- Entering floor k for the first time unlocks its `ENTER_*` achievement and +1 XP ([`LEVEL_ACHIEVEMENT_MAP`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L539)).

| Floor | Achievement | Reward |
|---|---|---|
| 1 | `ENTER_DUNGEON` | 3 |
| 2 | `ENTER_GNOMISH_MINES` | 3 |
| 3 | `ENTER_SEWERS` | 5 |
| 4 | `ENTER_VAULT` | 5 |
| 5 | `ENTER_TROLL_MINES` | 5 |
| 6 | `ENTER_FIRE_REALM` | 8 |
| 7 | `ENTER_ICE_REALM` | 8 |
| 8 | `ENTER_GRAVEYARD` | 8 |

- Spending XP: [[Craftax survival stats#XP and attributes]].
