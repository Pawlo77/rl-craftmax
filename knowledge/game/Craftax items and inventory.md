---
title: "Craftax items and inventory"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, inventory, items]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax items and inventory

The full Craftax `Inventory` ([craftax_state.py L10](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/craftax_state.py#L10)) and the equipment state stored beside it, with every source and use, at commit `c3c2e0d`.

## Inventory fields

Every field is capped at 99 at the end of each step ([`clip_inventory_and_intrinsics`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L381)); crafting arrows and torches is also refused once the count is 99.

| Field | Range | Obtained from | Used for |
|---|---|---|---|
| `wood` | 0-99 | `DO` on tree, fire tree, ice shrub | table (2), every tool (1), arrows (1), torches (1) |
| `stone` | 0-99 | `DO` on stone or stalagmite | furnace (1), `PLACE_STONE` (1), stone and iron tools (1), arrows (1) |
| `coal` | 0-99 | `DO` on coal; chests | iron tools (1), iron armour (3), torches (1) |
| `iron` | 0-99 | `DO` on iron; chests | iron tools (1), iron armour (3) |
| `diamond` | 0-99 | `DO` on diamond; chests | diamond pickaxe (3), sword (2), armour (3) |
| `sapphire` | 0-99 | `DO` on sapphire; chests | ice enchantments (1) |
| `ruby` | 0-99 | `DO` on ruby; chests | fire enchantments (1) |
| `sapling` | 0-99 | `DO` on grass, probability 0.1 | `PLACE_PLANT` (1) |
| `pickaxe` | level 0-4 | crafting; chests | mining tier |
| `sword` | level 0-4 | crafting; chests | melee damage |
| `bow` | 0 or 1 | first chest opened on floor 1 | `SHOOT_ARROW`, `ENCHANT_BOW` |
| `arrows` | 0-99 | crafting (2 per craft); chests | `SHOOT_ARROW` (1) |
| `armour` | 4 slots, each 0 none, 1 iron, 2 diamond | crafting | damage reduction |
| `torches` | 0-99 | crafting (4 per craft); chests | `PLACE_TORCH` (1) |
| `potions` | 6 counts (red, green, blue, pink, cyan, yellow) | chests | `DRINK_POTION_*` (1) |
| `books` | 0-99 (2 per episode at most) | first chest opened on floor 3 and on floor 4 | `READ_BOOK` (1) |

- Tools, swords, the bow and armour never wear out; there is no durability.
- There is no way to drop or trade items; the only losses are crafting, placing, shooting, drinking, reading and enchanting costs.

## Equipment state outside the inventory

| Field | Values | Set by |
|---|---|---|
| `sword_enchantment` | 0 none, 1 fire, 2 ice | `ENCHANT_SWORD` |
| `bow_enchantment` | 0 none, 1 fire, 2 ice | `ENCHANT_BOW` |
| `armour_enchantments` | 4 slots, each 0, 1 fire, 2 ice | `ENCHANT_ARMOUR` |
| `learned_spells` | [fireball, iceball] booleans | `READ_BOOK` |
| `potion_mapping` | permutation of the 6 effects over the 6 colours, fixed per episode | reset |

- Details: [[Craftax tools weapons and armour]], [[Craftax magic potions and enchanting]].

## Chest loot

Opening a chest (`DO` facing it) turns it into path and rolls these independent draws ([`add_items_from_chest`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L25)); `randint` upper bounds are exclusive, so ranges below are inclusive.

| Loot | Chance | Amount |
|---|---|---|
| Torches | 0.6 | 4-7 |
| One ore type | 0.6 | coal 1-3 (p 0.3), iron 1-2 (p 0.3), diamond 1 (p 0.15), sapphire 1 (p 0.125), ruby 1 (p 0.125) |
| Potion | 0.5 | 1-2 of one uniformly random colour |
| Arrows | 0.25 | 1-4 |
| Tool | 0.2 | pickaxe or sword (p 0.5 each) of level 1-4 with p 0.4, 0.3, 0.2, 0.1; kept only if better than the current one |
| Bow | first chest opened on floor 1 | 1 |
| Book | first chest opened on floor 3; first chest opened on floor 4 | 1 |

- Chests exist only on floors 1, 3 and 4, one per room, 8 rooms per floor, so at most 24 per episode; a fountain, an overlapping later room or the special block can overwrite one ([`generate_dungeon`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L156)).
- A chest is opened once; `chests_opened[floor]` records the first opening per floor and gates the bow and books.
- The bow and both books are guaranteed (first chest of their floor), not random.
- Tool-tier achievements (`MAKE_*_PICKAXE`, `MAKE_*_SWORD`) and `COLLECT_*` achievements unlock from chest loot too, because they are read from the inventory ([[Craftax achievements]]).

## Other sources

- Cows, bats and snails give food, not items ([[Craftax farming and food]]).
- Creatures drop nothing.
- Pre-placed dungeon torches and ladders stay on the map and are never collected.
