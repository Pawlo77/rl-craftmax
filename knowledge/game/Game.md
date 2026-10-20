---
title: "Game"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, game-mechanics, index]
created: 2026-10-20
updated: 2026-10-20
---
# Game

The complete rules of Craftax and Craftax-Classic, one note per topic, read from the game code at commit [`c3c2e0d`](https://github.com/MichaelTMatthews/Craftax/tree/c3c2e0d038c4e641f9481320c158f457f30c28f3) (same code as craftax 1.6.1). The paper ([arXiv 2402.16801v2](https://arxiv.org/html/2402.16801v2)) and [tutorial.md](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/tutorial.md) are secondary; where they disagree with the code the code wins and the difference is listed below.

## Core loop

- The player starts on the overworld (floor 0) with an empty inventory and must keep health, food, drink and energy above zero.
- Chop trees for wood, place a crafting table (2 wood), craft a wood pickaxe and sword, then mine stone, coal, iron and diamond for better tools and armour.
- Find the down ladder; on every floor below the overworld, kill 8 melee or ranged creatures to open it, then `DESCEND`.
- Each new floor brings new creatures, resources and darkness; the first visit gives 1 XP to raise dexterity, strength or intelligence.
- Chests on floors 1, 3 and 4 give the bow, two spell books, potions and ores; enchantment tables on floors 3 and 4 turn rubies and sapphires into fire and ice damage or defence.
- Late creatures resist physical damage, so fire and ice (spells, enchantments) become necessary.
- Floor 8 holds the necromancer: clear 8 waves of creatures and hit it after each to win.
- Every first-time achievement pays 1, 3, 5 or 8 reward (226 in total); health changes add +-0.1 per point.

## Notes

| Note | Contents |
|---|---|
| [[Craftax episode and time]] | start state, end conditions, step order, day/night, sleep, rest, view and light, persistence |
| [[Craftax survival stats]] | health, food, drink, energy, mana; hidden counters; XP and attributes |
| [[Craftax blocks and resources]] | all 37 block types and 5 item types, mining table, resources per floor |
| [[Craftax items and inventory]] | every inventory field, caps, sources, chest loot |
| [[Craftax crafting and placing]] | recipes, station distance, placement rules |
| [[Craftax tools weapons and armour]] | pickaxe tiers, sword damage, bow and arrows, armour |
| [[Craftax farming and food]] | saplings, plant growth, eating, drinking |
| [[Craftax creatures]] | 19 creatures, stats, AI, projectiles, spawning, despawning |
| [[Craftax combat]] | damage vectors, defence, elements, worked examples |
| [[Craftax magic potions and enchanting]] | mana, books, spells, potions, enchanting |
| [[Craftax floors and progression]] | floors 0-8, terrain and dungeon generation, light, ladders |
| [[Craftax boss fight]] | the necromancer, waves, winning |
| [[Craftax achievements]] | all 67 achievements with reward and trigger |
| [[Craftax action effects]] | what each of the 43 actions does |
| [[Craftax-Classic rules]] | the full Classic rule set and every difference |
| [[Craftax game complexity]] | state-space size compared with tic-tac-toe, chess and Go |

- Observation and action encodings: [[Craftax observations and actions]]; generation pipeline and seeding: [[Craftax world generation]]; reward and benchmarks: [[Craftax scoring and evaluation]].

## Rule surprises

Each checked in the code; those marked "run" were also checked by stepping craftax 1.6.1.

- Trees, fire trees and ice shrubs disappear after one `DO` (one wood each), in Craftax and Craftax-Classic (run).
- In full Craftax the player cannot walk onto water or lava, so lava is harmless; in Craftax-Classic walking into lava ends the episode (run).
- Ice shrubs are not solid: the player walks through them (run).
- Tables and furnaces can be placed on water and lava (run), can be removed with `DO` without refund, and are destroyed by creature projectiles.
- Skeletons (and every overworld monster) spawn on grass as well as path in full Craftax; in Craftax-Classic skeletons need path.
- Only melee and ranged kills open ladders; passive kills never count.
- Projectile kills of cows, bats and snails give no food and no `EAT_*` achievement.
- Only the first 10 saplings planted in an episode can ever ripen; growth slots are never freed.
- Creatures on other floors are frozen while the player is away.
- Enchanting needs the player to face the enchantment table, and armour enchantments can land on empty armour slots and still protect.
- The bow and both books are guaranteed in the first chest opened on their floor, not random.
- Food, drink and energy do not fall on floor 8, and all damage taken there is x1.5.
- Each necromancer hit counts as one of 8 progress points regardless of weapon; arrows and spells cannot hit it.
- In Craftax-Classic extra copies of a sword stack damage (3 wood swords deal 6) (run), and only the wood pickaxe can mine stone (run).

## Source disagreements

| Topic | Code (c3c2e0d) | Other source |
|---|---|---|
| Number of achievements | 67 | paper §3.3 text: 65 ([paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3)); paper Table 6 lists 67 |
| Orc soldier health | 9 | paper Table 3: 7 |
| Orc mage health | 6 | Table 3: 5 |
| Gnome warrior health | 7 | Table 3: 9 |
| Gnome archer health | 5 | Table 3: 6 |
| Knight archer damage | 5 | Table 3: 4 ([Table 3](https://arxiv.org/html/2402.16801v2#A4.T3)) |
| Deep thing health | 4 | Table 3: 6 |
| Ice elemental damage | 4 physical + 5 ice | Table 3: 4 + 4 |
| Floor numbers in prose | fire elementals floor 6, deep things floor 5 | paper §3.2: fire elementals "on floor 7", aquatic enemies "on floor 6" (one-based); Table 3 matches the code |
| Bow source | always the first chest opened on floor 1 | paper §3.2: "found in a random chest on the first dungeon floor" |
| Enchanting position | must face the table | tutorial: "standing next to the enchantment table" |
| Ice realm gems | sapphire 0.02, diamond 0.005, ruby 0 | tutorial: "lots of sapphires and rubies" |
| End of rest | health full, food or drink 0, or a hit | tutorial: "until an intrinsic decays to 0" (energy 0 does not end it) |
| Intelligence and mana | speeds mana regeneration; mana never decays | tutorial: "reduces mana decay" |
| Sword enchantment scaling | intelligence 5 gives x1.2 | code comment: "Int=5 does 25% more enchant damage" |
| Death | health <= 0 | tutorial: "falls beneath 0" |
| Light channel | binary visibility flag | paper App. D.4: a float ([[Craftax observations and actions]]) |

- Dungeon rooms have sides 5-9 (`randint(5, 10)` excludes 10).

## Sources

- Code: [craftax/craftax/](https://github.com/MichaelTMatthews/Craftax/tree/c3c2e0d/craftax/craftax) (`constants.py`, `game_logic.py`, `util/game_logic_utils.py`, `world_gen/`, `craftax_state.py`, `envs/`, `renderer.py`) and [craftax/craftax_classic/](https://github.com/MichaelTMatthews/Craftax/tree/c3c2e0d/craftax/craftax_classic).
- Paper: Matthews et al., "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning", ICML 2024, [arXiv HTML v2](https://arxiv.org/html/2402.16801v2) (§3, App. A, App. D, Tables 2, 3, 5, 6).
- Runtime checks: craftax 1.6.1 on CPU, stepping hand-edited states (tree, lava, ice shrub, table on water; Classic tree, sword stacking, pickaxe check, lava).
