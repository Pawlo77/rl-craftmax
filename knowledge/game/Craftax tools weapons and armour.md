---
title: "Craftax tools weapons and armour"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, equipment, combat]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax tools weapons and armour

Pickaxes, swords, the bow, arrows and armour of full Craftax at commit `c3c2e0d`. Recipes: [[Craftax crafting and placing]]; damage formulas: [[Craftax combat]].

## Pickaxe

| Level | Name | Mines |
|---|---|---|
| 0 | none | trees, fire trees, ice shrubs |
| 1 | wood | + stone, stalagmite, coal |
| 2 | stone | + iron |
| 3 | iron | + diamond |
| 4 | diamond | + sapphire, ruby |

- Levels are cumulative: each level mines everything below it ([`do_action`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L219) compares `pickaxe >= n`).
- The pickaxe does not affect combat.

## Sword

| Level | Name | Physical damage (strength 1) | Strength 5 |
|---|---|---|---|
| 0 | bare hands | 1 | 2 |
| 1 | wood | 2 | 4 |
| 2 | stone | 3 | 6 |
| 3 | iron | 5 | 10 |
| 4 | diamond | 8 | 16 |

- Physical damage = `[1, 2, 3, 5, 8][sword] x (1 + 0.25 (strength - 1))` ([`get_player_damage_vector`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L252)).
- A fire or ice enchantment adds `0.5 x base` (before strength) of that element, times `1 + 0.05 (intelligence - 1)`; a diamond sword adds 4 at intelligence 1, 4.8 at 5.
- Melee is `DO` on the facing creature; there is no attack cooldown for the player.

## Bow and arrows

- The bow comes only from the first chest opened on floor 1 ([[Craftax items and inventory#Chest loot]]).
- `SHOOT_ARROW` needs the bow, at least 1 arrow, and fewer than 3 player projectiles on the floor; it spends 1 arrow and fires from the player's tile in the facing direction ([`shoot_projectile`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2498)).
- The player's arrow is projectile type `ARROW2`: 5 physical damage, times `1 + 0.2 (dexterity - 1)` (9 at dexterity 5) ([`_move_player_projectile`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1713)).
- A bow enchantment adds half the physical damage (2.5) as fire or ice, also scaled by dexterity; intelligence does not scale it.
- Arrows fly one tile per step until they hit a creature, a solid block or the map edge; they cross water and lava.
- Arrows: 2 per `MAKE_ARROW` (1 wood, 1 stone), or 1-4 from 25% of chests.

## Armour

- 4 armour slots, each 0 (none), 1 (iron) or 2 (diamond) ([`Inventory.armour`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L41)).
- Physical defence = `0.1 x sum of slot levels`: 0.1 per iron piece, 0.2 per diamond piece, 0.4 for full iron, 0.8 for full diamond ([`get_damage_done_to_player`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L233)).
- Fire defence = `0.2 x number of fire-enchanted slots`, ice defence = `0.2 x number of ice-enchanted slots` (up to 0.8 each).
- Enchantments sit on slots, not on pieces: an empty slot can be enchanted and still counts ([[Craftax magic potions and enchanting#Enchanting]]).
- Armour does not slow the player or cost anything to wear.

## Upgrade path

| Stage | Tool reached | Needs |
|---|---|---|
| Overworld start | wood pickaxe and sword | 2 wood table + 1 wood each |
| Stone | stone pickaxe and sword | wood pickaxe for stone |
| Iron | iron pickaxe and sword, iron armour | stone pickaxe for iron, coal, furnace |
| Diamond | diamond sword (2), pickaxe (3), armour (3 per piece) | iron pickaxe for diamond; floors 2 and 5 are the richest ([[Craftax blocks and resources#Resource spawn rules per floor]]) |
| Gems | sapphire, ruby | diamond pickaxe; used only for enchanting |
