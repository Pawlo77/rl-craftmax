---
title: "Craftax action effects"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, game-mechanics, action-space]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax action effects

What each of the 43 full-Craftax actions does, when it succeeds and what it costs, from [`craftax_step`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L3006) and the functions it calls at commit `c3c2e0d`. Action IDs and keys: [[Craftax observations and actions#Actions]].

## General rules

- Every action can be chosen at every step; if its conditions fail it does nothing and time still advances one step.
- While sleeping or resting, the chosen action is replaced by `NOOP`.
- "Facing tile" = player position + the last movement direction; the player starts facing up.
- Facing-tile actions (`DO`, `PLACE_*`, `ENCHANT_*`) fail if the tile is outside the map; `PLACE_*` also fails if a creature stands there.
- Crafting needs a table (and for iron items a furnace) among the 8 surrounding tiles ([[Craftax crafting and placing]]).

## Movement and interaction

| Action | Effect | Conditions |
|---|---|---|
| `NOOP` | nothing | - |
| `LEFT`, `RIGHT`, `UP`, `DOWN` | turn to that direction, then step one tile | the step needs a tile in the map, not solid, not water or lava, no creature; turning happens even when blocked |
| `DO` | on the facing tile: attack a creature, else mine, collect a sapling, drink, eat a ripe plant, open a chest, remove a table or furnace, or hit the vulnerable necromancer | per target ([[Craftax blocks and resources#Mining with DO]]) |
| `SLEEP` | fall asleep until energy is full or a hit wakes the player | energy < max |
| `REST` | auto-`NOOP` until health is full, food or drink is 0, or a hit | health < max |
| `DESCEND` | go to the next floor's up-ladder tile | standing on a down ladder, floor kills >= 8, floor < 8 |
| `ASCEND` | go to the previous floor's down-ladder tile | standing on an up ladder, floor > 0 |

## Placing

| Action | Effect | Cost and target |
|---|---|---|
| `PLACE_STONE` | stone block | 1 stone; water or any non-solid tile without an item |
| `PLACE_TABLE` | crafting table | 2 wood; any non-solid tile without an item |
| `PLACE_FURNACE` | furnace | 1 stone; any non-solid tile without an item |
| `PLACE_PLANT` | sapling (plant block) | 1 sapling; grass without an item |
| `PLACE_TORCH` | torch item, lights a radius of about 4.75 tiles | 1 torch; grass, sand, path, fire grass, ice grass without an item |

## Crafting

| Action | Result | Inputs (station) |
|---|---|---|
| `MAKE_WOOD_PICKAXE` | pickaxe 1 | 1 wood (table) |
| `MAKE_STONE_PICKAXE` | pickaxe 2 | 1 wood, 1 stone (table) |
| `MAKE_IRON_PICKAXE` | pickaxe 3 | 1 wood, 1 stone, 1 iron, 1 coal (table, furnace) |
| `MAKE_DIAMOND_PICKAXE` | pickaxe 4 | 1 wood, 3 diamond (table) |
| `MAKE_WOOD_SWORD` | sword 1 | 1 wood (table) |
| `MAKE_STONE_SWORD` | sword 2 | 1 wood, 1 stone (table) |
| `MAKE_IRON_SWORD` | sword 3 | 1 wood, 1 stone, 1 iron, 1 coal (table, furnace) |
| `MAKE_DIAMOND_SWORD` | sword 4 | 1 wood, 2 diamond (table) |
| `MAKE_IRON_ARMOUR` | one iron armour slot | 3 iron, 3 coal (table, furnace) |
| `MAKE_DIAMOND_ARMOUR` | one diamond armour slot | 3 diamond (table) |
| `MAKE_ARROW` | +2 arrows | 1 wood, 1 stone (table) |
| `MAKE_TORCH` | +4 torches | 1 wood, 1 coal (table) |

- A pickaxe or sword recipe fails if the held tier is already equal or higher.

## Combat and magic

| Action | Effect | Conditions and cost |
|---|---|---|
| `SHOOT_ARROW` | arrow (5 physical, x dexterity bonus, + bow enchantment) flies in the facing direction | bow, 1 arrow, < 3 player projectiles |
| `CAST_FIREBALL` | fireball (3 fire x intelligence bonus) | spell learned, 2 mana, < 3 player projectiles |
| `CAST_ICEBALL` | iceball (3 ice x intelligence bonus) | spell learned, 2 mana, < 3 player projectiles |
| `READ_BOOK` | learn a random unknown spell | 1 book |
| `ENCHANT_SWORD` | sword gets the table's element | facing an enchantment table, sword > 0, 9 mana, 1 ruby (fire) or sapphire (ice) |
| `ENCHANT_ARMOUR` | one armour slot gets the table's element | same, at least one armour piece |
| `ENCHANT_BOW` | bow gets the table's element | same, bow |

## Potions and attributes

| Action | Effect | Conditions |
|---|---|---|
| `DRINK_POTION_RED` ... `DRINK_POTION_YELLOW` (6 actions) | the episode's effect for that colour: +8 or -3 health, mana or energy | at least one potion of that colour |
| `LEVEL_UP_DEXTERITY` | dexterity +1 | 1 XP, dexterity < 5 |
| `LEVEL_UP_STRENGTH` | strength +1 | 1 XP, strength < 5 |
| `LEVEL_UP_INTELLIGENCE` | intelligence +1 | 1 XP, intelligence < 5 |

- Formulas and edge cases: [[Craftax combat]], [[Craftax magic potions and enchanting]], [[Craftax survival stats]].
- Craftax-Classic has only actions 0-16 ([[Craftax-Classic rules]]).
