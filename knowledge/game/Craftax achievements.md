---
title: "Craftax achievements"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, game-mechanics, achievements, reward]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax achievements

All 67 achievements of full Craftax (IDs 0-66) with their reward and the exact code trigger, from [`Achievement`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L410), [`achievement_mapping`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L523) and [`calculate_inventory_achievements`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2845) at commit `c3c2e0d`; tiers match [paper Table 6](https://arxiv.org/html/2402.16801v2#A4.T6). Craftax-Classic's 22: [[Craftax-Classic rules#Achievements]].

## Reward rule

- Each achievement pays once per episode, in the step it first becomes true; achievements are never lost.
- Reward by tier: Basic 1, Intermediate 3, Advanced 5, Very advanced 8 (`achievement_mapping`: ID <= 24 gives 1, `INTERMEDIATE_ACHIEVEMENTS` 3, `VERY_ADVANCED_ACHIEVEMENTS` 8, all others 5).
- Totals: 25 x 1 + 18 x 3 + 15 x 5 + 9 x 8 = 226, counted by running craftax 1.6.1.
- The step reward also adds 0.1 x the health change ([[Craftax scoring and evaluation]]).
- "Inventory" triggers are read from the inventory at the end of each step, so chest loot unlocks them as well as crafting or mining.
- Enum members are declared out of value order (`COLLECT_SAPPHIRE = 54` appears before `MAKE_DIAMOND_SWORD = 25`); index arrays by `.value`.

## Basic (reward 1)

| ID | Achievement | Trigger |
|---|---|---|
| 0 | `COLLECT_WOOD` | inventory wood > 0 |
| 1 | `PLACE_TABLE` | a crafting table is placed |
| 2 | `EAT_COW` | a cow is killed with `DO` |
| 3 | `COLLECT_SAPLING` | inventory sapling > 0 |
| 4 | `COLLECT_DRINK` | `DO` facing water or a fountain |
| 5 | `MAKE_WOOD_PICKAXE` | pickaxe level >= 1 |
| 6 | `MAKE_WOOD_SWORD` | sword level >= 1 |
| 7 | `PLACE_PLANT` | a sapling is planted |
| 8 | `DEFEAT_ZOMBIE` | a zombie is killed (any weapon) |
| 9 | `COLLECT_STONE` | inventory stone > 0 |
| 10 | `PLACE_STONE` | stone is placed |
| 11 | `EAT_PLANT` | `DO` on a ripe plant |
| 12 | `DEFEAT_SKELETON` | a skeleton is killed |
| 13 | `MAKE_STONE_PICKAXE` | pickaxe level >= 2 |
| 14 | `MAKE_STONE_SWORD` | sword level >= 2 |
| 15 | `WAKE_UP` | sleep ends with energy at max, or a melee hit wakes the player |
| 16 | `PLACE_FURNACE` | a furnace is placed |
| 17 | `COLLECT_COAL` | inventory coal > 0 |
| 18 | `COLLECT_IRON` | inventory iron > 0 |
| 19 | `COLLECT_DIAMOND` | inventory diamond > 0 |
| 20 | `MAKE_IRON_PICKAXE` | pickaxe level >= 3 |
| 21 | `MAKE_IRON_SWORD` | sword level >= 3 |
| 22 | `MAKE_ARROW` | inventory arrows > 0 |
| 23 | `MAKE_TORCH` | inventory torches > 0 |
| 24 | `PLACE_TORCH` | a torch is placed |

## Intermediate (reward 3)

| ID | Achievement | Trigger |
|---|---|---|
| 25 | `MAKE_DIAMOND_SWORD` | sword level >= 4 |
| 26 | `MAKE_IRON_ARMOUR` | an iron armour piece is crafted |
| 27 | `MAKE_DIAMOND_ARMOUR` | a diamond armour piece is crafted |
| 28 | `ENTER_GNOMISH_MINES` | first arrival on floor 2 |
| 29 | `ENTER_DUNGEON` | first arrival on floor 1 |
| 36 | `DEFEAT_GNOME_WARRIOR` | a gnome warrior is killed |
| 37 | `DEFEAT_GNOME_ARCHER` | a gnome archer is killed |
| 38 | `DEFEAT_ORC_SOLIDER` | an orc soldier is killed (enum spelling "SOLIDER") |
| 39 | `DEFEAT_ORC_MAGE` | an orc mage is killed |
| 50 | `EAT_BAT` | a bat is killed with `DO` |
| 51 | `EAT_SNAIL` | a snail is killed with `DO` |
| 52 | `FIND_BOW` | inventory bow > 0 |
| 53 | `FIRE_BOW` | an arrow is shot |
| 54 | `COLLECT_SAPPHIRE` | inventory sapphire > 0 |
| 59 | `COLLECT_RUBY` | inventory ruby > 0 |
| 60 | `MAKE_DIAMOND_PICKAXE` | pickaxe level >= 4 |
| 61 | `OPEN_CHEST` | a chest is opened |
| 62 | `DRINK_POTION` | any potion is drunk |

## Advanced (reward 5)

| ID | Achievement | Trigger |
|---|---|---|
| 30 | `ENTER_SEWERS` | first arrival on floor 3 |
| 31 | `ENTER_VAULT` | first arrival on floor 4 |
| 32 | `ENTER_TROLL_MINES` | first arrival on floor 5 |
| 40 | `DEFEAT_LIZARD` | a lizard is killed |
| 41 | `DEFEAT_KOBOLD` | a kobold is killed |
| 42 | `DEFEAT_TROLL` | a troll is killed |
| 43 | `DEFEAT_DEEP_THING` | a deep thing is killed |
| 55 | `LEARN_FIREBALL` | a book teaches fireball |
| 56 | `CAST_FIREBALL` | a fireball is cast |
| 57 | `LEARN_ICEBALL` | a book teaches iceball |
| 58 | `CAST_ICEBALL` | an iceball is cast |
| 63 | `ENCHANT_SWORD` | the sword is enchanted |
| 64 | `ENCHANT_ARMOUR` | an armour slot is enchanted |
| 65 | `DEFEAT_KNIGHT` | a knight is killed |
| 66 | `DEFEAT_ARCHER` | a knight archer is killed |

## Very advanced (reward 8)

| ID | Achievement | Trigger |
|---|---|---|
| 33 | `ENTER_FIRE_REALM` | first arrival on floor 6 |
| 34 | `ENTER_ICE_REALM` | first arrival on floor 7 |
| 35 | `ENTER_GRAVEYARD` | first arrival on floor 8 |
| 44 | `DEFEAT_PIGMAN` | a pigman is killed |
| 45 | `DEFEAT_FIRE_ELEMENTAL` | a fire elemental is killed |
| 46 | `DEFEAT_FROST_TROLL` | a frost troll is killed |
| 47 | `DEFEAT_ICE_ELEMENTAL` | an ice elemental is killed |
| 48 | `DAMAGE_NECROMANCER` | first hit on the vulnerable necromancer |
| 49 | `DEFEAT_NECROMANCER` | boss progress reaches 8 (ends the episode) |

## Trigger notes

- `DEFEAT_*` unlock on a kill by melee, arrow or spell, on the creature's home floor or in a boss wave ([`attack_mob_class`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L32)).
- `EAT_*` unlock only on a `DO` kill; a projectile kill of a passive creature gives nothing.
- `ENTER_DUNGEON` (29) is floor 1 and `ENTER_GNOMISH_MINES` (28) is floor 2; the IDs are out of floor order but the mapping is correct ([issue #53](https://github.com/MichaelTMatthews/Craftax/issues/53), closed).
- `WAKE_UP` is not unlocked when a projectile wakes the player.
- `ENCHANT_BOW` has no achievement.
- IDs 0-21 are the 22 Crafter / Craftax-Classic achievements; Craftax-Classic pays 1 for each.
- The paper's text says 65 achievements; code and paper Table 6 have 67 ([paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3)).

## Achievements by floor

| Floor | Achievements available there |
|---|---|
| 0 | all 25 basic achievements (IDs 0-24); `MAKE_IRON_ARMOUR` and `MAKE_DIAMOND_*` if enough ore is found |
| 1 | `ENTER_DUNGEON`, orc kills, `EAT_SNAIL`, `OPEN_CHEST`, `FIND_BOW`, `FIRE_BOW`, `DRINK_POTION` |
| 2 | `ENTER_GNOMISH_MINES`, gnome kills, `EAT_BAT`, `COLLECT_SAPPHIRE`, `COLLECT_RUBY` (diamond pickaxe) |
| 3 | `ENTER_SEWERS`, lizard and kobold kills, spells from the book, ice enchanting |
| 4 | `ENTER_VAULT`, knight and archer kills, second book, fire enchanting |
| 5 | `ENTER_TROLL_MINES`, troll and deep thing kills |
| 6 | `ENTER_FIRE_REALM`, pigman and fire elemental kills |
| 7 | `ENTER_ICE_REALM`, frost troll and ice elemental kills |
| 8 | `ENTER_GRAVEYARD`, `DAMAGE_NECROMANCER`, `DEFEAT_NECROMANCER` |
