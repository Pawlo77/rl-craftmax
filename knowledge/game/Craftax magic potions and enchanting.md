---
title: "Craftax magic potions and enchanting"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, game-mechanics, magic, potions, enchanting]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax magic potions and enchanting

Mana, books, spells, potions and enchantment tables of full Craftax from [game_logic.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2539) at commit `c3c2e0d`. Mana regeneration: [[Craftax survival stats]].

## Mana

- Start 9, maximum `6 + 3 x intelligence` (9 to 21).
- Regenerates +1 every 31 awake steps at intelligence 1 (16 asleep), faster with intelligence; never decays.
- Spent by casting (2) and enchanting (9).

## Books and spells

- Books come only from the first chest opened on floor 3 (sewers) and the first chest opened on floor 4 (vaults): 2 per episode ([`add_items_from_chest`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L142)).
- `READ_BOOK` consumes 1 book and teaches one spell chosen uniformly among the spells not yet learned (fireball or iceball), unlocking `LEARN_FIREBALL` or `LEARN_ICEBALL` ([`read_book`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2693)).
- The two books therefore teach both spells in a random order.
- `CAST_FIREBALL` / `CAST_ICEBALL` need the spell learned, mana >= 2 and fewer than 3 player projectiles on the floor; they cost 2 mana and fire from the player's tile in the facing direction ([`cast_spell`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2539)).
- Fireball: 3 fire damage; iceball: 3 ice damage; both x`(1 + 0.5 (intelligence - 1))`, 9 at intelligence 5 ([[Craftax combat#Player projectiles]]).
- Each first successful cast unlocks `CAST_FIREBALL` or `CAST_ICEBALL`.

## Potions

- Six colours: red, green, blue, pink, cyan, yellow (inventory indices 0-5, actions `DRINK_POTION_RED` ... `DRINK_POTION_YELLOW`).
- Six effects ([`drink_potion`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2594)):

| Effect index | Effect |
|---|---|
| 0 | health +8 |
| 1 | health -3 |
| 2 | mana +8 |
| 3 | mana -3 |
| 4 | energy +8 |
| 5 | energy -3 |

- At reset `potion_mapping` is a random permutation of the six effects over the six colours; it stays fixed for the episode and changes every episode ([`generate_world`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen.py#L643)).
- Drinking needs at least one potion of that colour, consumes one, unlocks `DRINK_POTION`; results are clipped to [0, max] at the end of the step.
- The -3 health potion can kill (health <= 0 ends the episode).
- Potions come only from chests: 50% of chests hold 1-2 potions of one uniformly random colour ([[Craftax items and inventory#Chest loot]]).
- The colour-to-effect mapping is not observed; the agent must learn it by drinking and watching its stats ([paper §3.2](https://arxiv.org/html/2402.16801v2#S3.SS2)).

## Enchanting

- Enchantment tables: ice table on floor 3 (sewers), fire table on floor 4 (vaults), each at the first room's corner + (2, 2) ([[Craftax floors and progression]]).
- Conditions ([`enchant`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2726)): the player faces the table (the tile in front, not merely adjacent), mana >= 9, and holds 1 gem of the table's element: ruby for fire, sapphire for ice.
- Cost: 9 mana and 1 gem.

| Action | Requires | Effect |
|---|---|---|
| `ENCHANT_SWORD` | sword level > 0 | `sword_enchantment` = table element; achievement `ENCHANT_SWORD` |
| `ENCHANT_BOW` | bow | `bow_enchantment` = table element; no achievement |
| `ENCHANT_ARMOUR` | at least one armour piece | one armour slot gets the table element; achievement `ENCHANT_ARMOUR` |

- A new sword or bow enchantment replaces the old one (fire overwrites ice and vice versa).
- Armour slot choice: a uniformly random slot with no enchantment; if none is left, a random slot with the other element; if all 4 slots already have this element, nothing changes but the mana and gem are still spent.
- The slot choice ignores whether the slot holds armour, so with one armour piece the enchantment can land on an empty slot; enchanted empty slots still add 0.2 elemental defence each ([[Craftax combat#Damage to the player]]).
- Sword enchantment: +50% of the sword's base physical damage as that element (x`1 + 0.05 (intelligence - 1)`); bow enchantment: +50% of arrow physical damage as that element; armour: +0.2 defence per slot against that element.
- 4 rubies (or sapphires) and 36 mana fully enchant the armour slots against one element.
