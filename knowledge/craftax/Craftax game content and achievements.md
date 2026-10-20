---
title: "Craftax game content and achievements"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, game-mechanics, achievements]
created: 2026-10-20
---
# Craftax game content and achievements

Game rules of the full Craftax environment as implemented in [game_logic.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py), [game_logic_utils.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py) and [constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py) at 1.6.1, with the player guide [tutorial.md](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/tutorial.md). Craftax-Classic keeps only the overworld, Crafter's 3 creatures, the wood/stone/iron tools and achievements 0-21.

## Floors

| Floor | Name | Layout | Dark | Melee / ranged / passive creatures | Specials |
|---|---|---|---|---|---|
| 0 | Overworld | Perlin terrain (grass, water, sand, mountains, caves, lava) | Day/night cycle | zombie / skeleton / cow | Ladder down is open from the start |
| 1 | Dungeon | 8 rooms joined by paths | No | orc soldier / orc mage / snail | First chest opened gives a bow; fountains |
| 2 | Gnomish Mines | Caverns | Yes | gnome warrior / gnome archer / bat | Water pools; coal, iron, diamond, sapphire, ruby |
| 3 | Sewers | Rooms with water | No | lizard / kobold / snail | First chest gives a book; ice enchantment table |
| 4 | Vaults | Rooms | No | knight / knight archer / snail | First chest gives a book; fire enchantment table |
| 5 | Troll Mines | Caverns, richest ores | Yes | troll / deep thing (spawns in water) / bat | |
| 6 | Fire Realm | Islands in lava | No | pigman / fire elemental / bat | No water; coal and ruby |
| 7 | Ice Realm | Ice terrain | Yes | frost troll / ice elemental / none | No food; sapphire, diamond |
| 8 | Graveyard | Walls and graves | Yes | waves of every floor's creatures | Necromancer boss; no ladder out |

- Floor order, creatures and specials: `FLOOR_MOB_MAPPING`, `LEVEL_ACHIEVEMENT_MAP`, `generate_world` stacking order, `add_items_from_chest`, [world_gen_configs.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/world_gen/world_gen_configs.py); dark floors 2, 5, 7, 8 per [paper App. D.3](https://arxiv.org/html/2402.16801v2#A4.SS3). Generation details: [[Craftax world generation]].
- `DESCEND` works on a down ladder only after 8 creature kills on the current floor (`MONSTERS_KILLED_TO_CLEAR_LEVEL = 8`); the overworld counter starts at 10.
- `ASCEND` works on an up ladder; the player arrives on the matching ladder of the next floor.
- Monster spawn rate is tripled on a floor until it is cleared (8 kills).
- On dark floors the player places torches (`PLACE_TORCH`, lights a 9x9 area) to see; tiles with light <= 0.05 are blanked in the symbolic observation.

## Creatures

Stats from code (`MOB_TYPE_HEALTH_MAPPING`, `MOB_TYPE_DAMAGE_MAPPING`, `MOB_TYPE_DEFENSE_MAPPING`, indexed by creature type); damage as physical + fire + ice; defence as the fraction of physical / fire / ice damage blocked.

| Creature | Class | Health | Damage | Defence | Paper Table 3 differs |
|---|---|---|---|---|---|
| Zombie | melee | 5 | 2 | 0 | |
| Skeleton | ranged (arrow) | 3 | 2 | 0 | |
| Cow | passive | 3 | - | 0 | |
| Orc soldier | melee | 9 | 3 | 0 | health 7 |
| Orc mage | ranged (fireball) | 6 | 0 + 3 + 0 | 0 | health 5 |
| Snail | passive | 6 | - | 0 | |
| Gnome warrior | melee | 7 | 4 | 0 | health 9 |
| Gnome archer | ranged (arrow) | 5 | 2 | 0 | health 6 |
| Bat | passive | 4 | - | 0 | |
| Lizard | melee, swims | 11 | 5 | 0 | |
| Kobold | ranged (dagger) | 8 | 4 | 0 | |
| Knight | melee | 12 | 6 | 0.5 physical | |
| Knight archer | ranged (arrow2) | 12 | 5 | 0.5 physical | damage 4 |
| Troll | melee | 20 | 6 + 1 + 1 | 0.2 physical | |
| Deep thing | ranged (slimeball), aquatic | 4 | 4 + 3 + 3 | 0 | health 6 |
| Pigman | melee | 20 | 3 + 5 + 0 | 0.9 physical, 1.0 fire | |
| Fire elemental | ranged, flying | 14 | 3 + 5 + 0 | 0.9 physical, 1.0 fire | |
| Frost troll | melee | 24 | 4 + 0 + 5 | 0.9 physical, 1.0 ice | |
| Ice elemental | ranged, flying | 16 | 4 + 0 + 5 | 0.9 physical, 1.0 ice | damage 4 + 4 |

- 19 creatures in total ([paper §3.2](https://arxiv.org/html/2402.16801v2#S3.SS2), [Table 3](https://arxiv.org/html/2402.16801v2#A4.T3)).
- Melee damage to a sleeping player is multiplied by 3.5; all damage on the boss floor by 1.5 (`BOSS_FIGHT_EXTRA_DAMAGE = 0.5`).
- Player defence: each armour piece blocks 0.1 per level of physical damage (iron 1, diamond 2) and each fire/ice-enchanted piece 0.2 of that element ([`get_damage_done_to_player`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L233)).
- Player melee damage by sword level 0-4: 1, 2, 3, 5, 8, times `1 + 0.25 (strength - 1)`; an enchanted sword adds 50% as fire or ice damage ([`get_player_damage_vector`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L252)).

## Crafting and placing

| Action | Cost | Needs nearby |
|---|---|---|
| `PLACE_TABLE` | 2 wood | |
| `PLACE_FURNACE` | 1 stone | |
| `PLACE_STONE` | 1 stone (onto water, lava or free floor) | |
| `PLACE_PLANT` | 1 sapling (onto grass) | |
| `PLACE_TORCH` | 1 torch | |
| `MAKE_WOOD_PICKAXE` / `MAKE_WOOD_SWORD` | 1 wood | table |
| `MAKE_STONE_PICKAXE` / `MAKE_STONE_SWORD` | 1 wood + 1 stone | table |
| `MAKE_IRON_PICKAXE` / `MAKE_IRON_SWORD` | 1 wood + 1 stone + 1 iron + 1 coal | table + furnace |
| `MAKE_DIAMOND_PICKAXE` | 1 wood + 3 diamond | table |
| `MAKE_DIAMOND_SWORD` | 1 wood + 2 diamond | table |
| `MAKE_IRON_ARMOUR` (one piece) | 3 iron + 3 coal | table + furnace |
| `MAKE_DIAMOND_ARMOUR` (one piece) | 3 diamond | table |
| `MAKE_ARROW` | 1 wood + 1 stone, gives 2 arrows | table |
| `MAKE_TORCH` | 1 wood + 1 coal, gives 4 torches | table |

- Source: [`do_crafting`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L520), `place_block`; "nearby" means one of the 8 surrounding tiles.
- Mining with `DO`: tree needs nothing; stone, coal and stalagmite need pickaxe level 1; iron level 2; diamond level 3; sapphire and ruby level 4 (diamond pickaxe).
- `DO` on grass gives a sapling with probability 0.1; on water or a fountain +1 drink; on a ripe plant +4 food; killing a cow, bat or snail +6 food.
- Inventory counts are capped at 99.
- Chests: torches 60%, one ore type 60%, a potion 50%, arrows 25%, a pickaxe or sword 20% ([`add_items_from_chest`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L25)).

## Player stats

| Stat | Start | Max | Notes |
|---|---|---|---|
| Health | 9 | 8 + strength | Regenerates while food, drink and energy (or sleep) are non-zero; drains otherwise |
| Food, drink, energy | 9 | 7 + 2 x dexterity | Decay slower with dexterity; no loss on the boss floor |
| Mana | 9 | 6 + 3 x intelligence | Recovers over time, faster with intelligence and while sleeping |
| Dexterity, strength, intelligence | 1 | 5 (`max_attribute`) | +1 per XP spent with `LEVEL_UP_*` |
| XP | 0 | | +1 the first time each floor is entered |

- Source: [`get_max_*`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L361), [`update_player_intrinsics`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1829), `change_floor`, `level_up_attributes`, `generate_world`.
- `SLEEP` (when energy is below max) and `REST` (when health is below max) lock the player into `NOOP` until energy, respectively health, is full; rest also stops when food or drink hits 0.
- Dexterity raises arrow damage by 20% per level above 1; intelligence raises spell damage by 50% per level above 1.
- Day length is 300 steps; at night zombies spawn more often in the overworld.

## Magic, potions, enchanting, boss

- `READ_BOOK` consumes a book and teaches a random not-yet-learned spell (fireball or iceball).
- `CAST_FIREBALL` / `CAST_ICEBALL` cost 2 mana and fire a projectile; at most 3 player projectiles exist at once.
- `SHOOT_ARROW` needs a bow and an arrow.
- Six potion colours; each episode `potion_mapping` is a random permutation of six effects: +8 or -3 health, mana or energy ([`drink_potion`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2594)).
- `ENCHANT_SWORD` / `ENCHANT_ARMOUR` / `ENCHANT_BOW` while facing an enchantment table: 9 mana plus 1 ruby (fire table) or 1 sapphire (ice table); armour enchants one random piece ([`enchant`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2726)).
- Boss: the necromancer is vulnerable when no melee or ranged creatures are alive and the 7-step spawn timer has run out; each hit adds 1 to `boss_progress` and starts a new wave; the game is won at `boss_progress >= 8`, which also ends the episode ([`has_beaten_boss`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L28)).
- The episode ends when health <= 0, the boss is beaten, or `timestep >= max_timesteps` ([`is_game_over`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L4)).

## Achievements

67 achievements (`len(Achievement)`, IDs 0-66), each rewarded once per episode by tier ([`achievement_mapping`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L523); [paper Table 6](https://arxiv.org/html/2402.16801v2#A4.T6)). Counts and the 226 total were computed by running craftax 1.6.1.

| Tier (reward) | Count | Achievements (ID) |
|---|---|---|
| Basic (1) | 25 | `COLLECT_WOOD` 0, `PLACE_TABLE` 1, `EAT_COW` 2, `COLLECT_SAPLING` 3, `COLLECT_DRINK` 4, `MAKE_WOOD_PICKAXE` 5, `MAKE_WOOD_SWORD` 6, `PLACE_PLANT` 7, `DEFEAT_ZOMBIE` 8, `COLLECT_STONE` 9, `PLACE_STONE` 10, `EAT_PLANT` 11, `DEFEAT_SKELETON` 12, `MAKE_STONE_PICKAXE` 13, `MAKE_STONE_SWORD` 14, `WAKE_UP` 15, `PLACE_FURNACE` 16, `COLLECT_COAL` 17, `COLLECT_IRON` 18, `COLLECT_DIAMOND` 19, `MAKE_IRON_PICKAXE` 20, `MAKE_IRON_SWORD` 21, `MAKE_ARROW` 22, `MAKE_TORCH` 23, `PLACE_TORCH` 24 |
| Intermediate (3) | 18 | `MAKE_DIAMOND_SWORD` 25, `MAKE_IRON_ARMOUR` 26, `MAKE_DIAMOND_ARMOUR` 27, `ENTER_GNOMISH_MINES` 28, `ENTER_DUNGEON` 29, `DEFEAT_GNOME_WARRIOR` 36, `DEFEAT_GNOME_ARCHER` 37, `DEFEAT_ORC_SOLIDER` 38, `DEFEAT_ORC_MAGE` 39, `EAT_BAT` 50, `EAT_SNAIL` 51, `FIND_BOW` 52, `FIRE_BOW` 53, `COLLECT_SAPPHIRE` 54, `COLLECT_RUBY` 59, `MAKE_DIAMOND_PICKAXE` 60, `OPEN_CHEST` 61, `DRINK_POTION` 62 |
| Advanced (5) | 15 | `ENTER_SEWERS` 30, `ENTER_VAULT` 31, `ENTER_TROLL_MINES` 32, `DEFEAT_LIZARD` 40, `DEFEAT_KOBOLD` 41, `DEFEAT_TROLL` 42, `DEFEAT_DEEP_THING` 43, `LEARN_FIREBALL` 55, `CAST_FIREBALL` 56, `LEARN_ICEBALL` 57, `CAST_ICEBALL` 58, `ENCHANT_SWORD` 63, `ENCHANT_ARMOUR` 64, `DEFEAT_KNIGHT` 65, `DEFEAT_ARCHER` 66 |
| Very advanced (8) | 9 | `ENTER_FIRE_REALM` 33, `ENTER_ICE_REALM` 34, `ENTER_GRAVEYARD` 35, `DEFEAT_PIGMAN` 44, `DEFEAT_FIRE_ELEMENTAL` 45, `DEFEAT_FROST_TROLL` 46, `DEFEAT_ICE_ELEMENTAL` 47, `DAMAGE_NECROMANCER` 48, `DEFEAT_NECROMANCER` 49 |

- Total: 25 x 1 + 18 x 3 + 15 x 5 + 9 x 8 = 226.
- IDs 0-21 are the 22 Crafter / Craftax-Classic achievements; Craftax-Classic gives +1 for each.
- `ENTER_DUNGEON` (29) is floor 1 and `ENTER_GNOMISH_MINES` (28) is floor 2; the IDs are out of floor order on purpose ([issue #53](https://github.com/MichaelTMatthews/Craftax/issues/53), closed by its author as a misread).
- `DEFEAT_ORC_SOLIDER` is spelled that way in the enum.
- Item achievements (`COLLECT_*`, `MAKE_*` tools, `FIND_BOW`, `MAKE_ARROW`, `MAKE_TORCH`) are set from the inventory at the end of each step, so loot from chests also unlocks them.
- `ENCHANT_BOW` has no achievement.
- Enum members are not declared in value order (for example `COLLECT_SAPPHIRE = 54` comes before `MAKE_DIAMOND_SWORD = 25`); index arrays by `.value`, not by declaration position.
