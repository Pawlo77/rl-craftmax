---
title: "Craftax crafting and placing"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, crafting]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax crafting and placing

Every recipe and placement rule of full Craftax from [`do_crafting`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L520) and [`place_block`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L828) at commit `c3c2e0d`. Craftax-Classic differences: [[Craftax-Classic rules]].

## Station distance

- "Near a table" and "near a furnace" mean the station occupies one of the 8 tiles around the player, diagonals included (`CLOSE_BLOCKS`, [`is_near_block`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L337)).
- Facing does not matter for crafting; one table and one furnace can serve any number of crafts.
- Crafting is instant (one step, one item per action).

## Recipes

| Action | Inputs | Station | Output | Extra condition |
|---|---|---|---|---|
| `MAKE_WOOD_PICKAXE` | 1 wood | table | pickaxe level 1 | pickaxe < 1 |
| `MAKE_STONE_PICKAXE` | 1 wood, 1 stone | table | pickaxe level 2 | pickaxe < 2 |
| `MAKE_IRON_PICKAXE` | 1 wood, 1 stone, 1 iron, 1 coal | table and furnace | pickaxe level 3 | pickaxe < 3 |
| `MAKE_DIAMOND_PICKAXE` | 1 wood, 3 diamond | table | pickaxe level 4 | pickaxe < 4 |
| `MAKE_WOOD_SWORD` | 1 wood | table | sword level 1 | sword < 1 |
| `MAKE_STONE_SWORD` | 1 wood, 1 stone | table | sword level 2 | sword < 2 |
| `MAKE_IRON_SWORD` | 1 wood, 1 stone, 1 iron, 1 coal | table and furnace | sword level 3 | sword < 3 |
| `MAKE_DIAMOND_SWORD` | 1 wood, 2 diamond | table | sword level 4 | sword < 4 |
| `MAKE_IRON_ARMOUR` | 3 iron, 3 coal | table and furnace | first armour slot at 0 becomes 1 (iron) | a slot with level 0 exists |
| `MAKE_DIAMOND_ARMOUR` | 3 diamond | table | first armour slot below 2 becomes 2 (diamond) | a slot below 2 exists |
| `MAKE_ARROW` | 1 wood, 1 stone | table | +2 arrows | arrows < 99 |
| `MAKE_TORCH` | 1 wood, 1 coal | table | +4 torches | torches < 99 |

- Tiers can be skipped: a diamond pickaxe can be crafted with no lower pickaxe, and lower tiers can never be crafted once a higher one is held.
- Diamond tools and diamond armour do not need a furnace.
- Diamond armour goes into the lowest-index slot below level 2, whether that slot is empty or holds iron.
- Four armour crafts of one kind fill all four slots; there are no named pieces (helmet etc.), only slots.
- Crafting `MAKE_IRON_ARMOUR` and `MAKE_DIAMOND_ARMOUR` sets their achievements directly; every other crafted item unlocks its achievement through the inventory check at the end of the step ([[Craftax achievements]]).

## Placement target

All `PLACE_*` actions act on the facing tile and fail if that tile is out of the map or holds a creature or the player.

| Action | Cost | Allowed target | Result |
|---|---|---|---|
| `PLACE_TABLE` | 2 wood (needs >= 2) | any non-solid tile without an item | crafting table |
| `PLACE_FURNACE` | 1 stone | any non-solid tile without an item | furnace |
| `PLACE_STONE` | 1 stone | water, or any non-solid tile without an item | stone |
| `PLACE_TORCH` | 1 torch | grass, sand, path, fire grass or ice grass, without an item | torch item; lights the area |
| `PLACE_PLANT` | 1 sapling | grass without an item | plant (sapling) |

- Non-solid tiles include water and lava, so tables and furnaces can be placed on water and lava, replacing them (checked on 1.6.1: a table placed on water); also on ice shrubs and darkness.
- The furnace does not need a nearby table.
- Placing stone on lava or water and mining it again turns the tile into path: the way to bridge lava (floor 6) and fill water (floor 3) ([tutorial.md](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/tutorial.md)).
- Placed stone is a wall against creatures; blocking oneself in before sleeping is the tutorial's advice.
- A torch adds the light pattern `clip(1 - d / 5, 0, 1)` (d = Euclidean distance, 9 x 9 square) to the floor's light map, capped at 1; torches are permanent ([[Craftax episode and time#What the player sees]]).
- A sapling placed while 10 plants are already tracked still becomes a plant block but never ripens ([[Craftax farming and food]]).

## Removing placed blocks

- `DO` facing a crafting table or furnace removes it (becomes path) with no refund ([`do_action`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L236)).
- A creature projectile that reaches a crafting table or furnace destroys it (becomes path) ([`_move_mob_projectile`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1662)).
- Placed stone is mined like any stone (pickaxe >= 1, returns the stone).
- Plants and torches cannot be removed.
