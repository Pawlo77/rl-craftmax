---
title: "Craftax creatures"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, game-mechanics, creatures, ai]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax creatures

All 19 creatures of full Craftax plus the necromancer: stats, floors, movement, attacks, spawning and despawning, from [constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L142), [`update_mobs`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1091) and [`spawn_mobs`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2044) at commit `c3c2e0d`.

## Classes and type IDs

- Creatures come in three classes (passive, melee, ranged), each with type IDs 0-7; every stat table is indexed by type ID, not by floor (the "Floor" comments in `MOB_TYPE_HEALTH_MAPPING` are misleading).
- Each floor picks one type per class from `FLOOR_MOB_MAPPING` ([constants.py L142](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L142)).
- Capacity per floor (`StaticEnvParams`): 3 melee, 3 passive, 2 ranged, 3 creature projectiles, 3 player projectiles ([craftax_state.py L127](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/craftax_state.py#L127)).

## Stat table

Damage is physical + fire + ice; defence is the fraction of each damage type removed ([`MOB_TYPE_DAMAGE_MAPPING`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L262), [`MOB_TYPE_HEALTH_MAPPING`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L277), [`MOB_TYPE_DEFENSE_MAPPING`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L294), [`MOB_TYPE_COLLISION_MAPPING`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L182)).

| Creature | Class, type | Floor | Health | Attack | Damage | Defence (phys, fire, ice) | Movement |
|---|---|---|---|---|---|---|---|
| Cow | passive 0 | 0 | 3 | none | - | 0, 0, 0 | land |
| Bat | passive 1 | 2, 5, 6 | 4 | none | - | 0, 0, 0 | flying |
| Snail | passive 2 | 1, 3, 4 | 6 | none | - | 0, 0, 0 | land |
| Zombie | melee 0 | 0 | 5 | melee | 2 | 0, 0, 0 | land |
| Gnome warrior | melee 1 | 2 | 7 | melee | 4 | 0, 0, 0 | land |
| Orc soldier | melee 2 | 1 | 9 | melee | 3 | 0, 0, 0 | land |
| Lizard | melee 3 | 3 | 11 | melee | 5 | 0, 0, 0 | amphibian |
| Knight | melee 4 | 4 | 12 | melee | 6 | 0.5, 0, 0 | land |
| Troll | melee 5 | 5 | 20 | melee | 6 + 1 + 1 | 0.2, 0, 0 | land |
| Pigman | melee 6 | 6 | 20 | melee | 3 + 5 + 0 | 0.9, 1.0, 0 | land |
| Frost troll | melee 7 | 7 | 24 | melee | 4 + 0 + 5 | 0.9, 0, 1.0 | land |
| Skeleton | ranged 0 | 0 | 3 | arrow | 2 | 0, 0, 0 | land |
| Gnome archer | ranged 1 | 2 | 5 | arrow | 2 | 0, 0, 0 | land |
| Orc mage | ranged 2 | 1 | 6 | fireball | 0 + 3 + 0 | 0, 0, 0 | land |
| Kobold | ranged 3 | 3 | 8 | dagger | 4 | 0, 0, 0 | land |
| Knight archer | ranged 4 | 4 | 12 | arrow2 | 5 | 0.5, 0, 0 | land |
| Deep thing | ranged 5 | 5 | 4 | slimeball | 4 + 3 + 3 | 0, 0, 0 | aquatic |
| Fire elemental | ranged 6 | 6 | 14 | fireball2 | 3 + 5 + 0 | 0.9, 1.0, 0 | flying |
| Ice elemental | ranged 7 | 7 | 16 | iceball2 | 4 + 0 + 5 | 0.9, 0, 1.0 | flying |

- Movement types: land = ground tiles only; amphibian = ground and water; aquatic = water only; flying = ground, water and lava. Solid blocks, creatures and the player block everyone.
- Floor 7 maps its passive type to the bat, but its passive spawn chance is 0.0, so no bats appear there.
- The paper's Table 3 differs for 7 values ([[Game#Source disagreements]]).
- The necromancer is a block, not a creature ([[Craftax boss fight]]).

## Melee AI

Per step, for each live melee creature ([`_move_melee_mob`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1094)):

1. If its Manhattan distance to the player is exactly 1 and its cooldown is <= 0, it attacks: damage to the player, cooldown set to 5, it does not move.
2. Otherwise its cooldown drops by 1 and it proposes a move: towards the player along the axis with the larger gap (random on ties) with probability 0.75 if the Manhattan distance is < 10 (always on floor 8); else one random step among 4 directions.
3. The move happens only if the target tile is in the map, free of creatures and the player, and allowed by its movement type.

- A melee hit on a sleeping player deals 3.5x damage (`1 + 2.5 x is_sleeping`) and wakes the player, unlocking `WAKE_UP`.
- A melee hit also ends resting.
- Melee creatures attack only orthogonally adjacent targets; a diagonal neighbour cannot hit.

## Ranged AI

Per step, for each live ranged creature, with d = Manhattan distance ([`_move_ranged_mob`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1385)):

1. Intended move: towards the player if d >= 6, away if d <= 3, random otherwise; the intended move is kept with probability 0.15, else replaced by a random step.
2. It attacks if cooldown <= 0 and either 4 <= d <= 5, or d <= 3 and its proposed move is blocked.
3. An attack sets cooldown 4, keeps the creature in place, and spawns its projectile on its own tile flying along the larger-gap axis towards the player, if fewer than 3 creature projectiles exist on the floor.
4. Otherwise cooldown -1 and it moves if the tile is valid.

- Because the projectile flies along one axis, it hits only if the player is on that line when it arrives.

## Passive AI

- Each step a passive creature picks one of 8 options: 4 directions or 4 "stay" entries, so it stays put half the time ([`_move_passive_mob`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1293)).
- Passive creatures never attack.

## Projectiles

| Projectile | Fired by | Damage |
|---|---|---|
| Arrow (0) | skeleton, gnome archer | 2 |
| Dagger (1) | kobold | 4 |
| Fireball (2) | orc mage; player spell | 0 + 3 + 0 |
| Iceball (3) | player spell | 0 + 0 + 3 |
| Arrow2 (4) | knight archer; player bow | 5 |
| Slimeball (5) | deep thing | 4 + 3 + 3 |
| Fireball2 (6) | fire elemental | 3 + 5 + 0 |
| Iceball2 (7) | ice elemental | 4 + 0 + 5 |

- Projectiles move 1 tile per step after creatures move ([`_move_mob_projectile`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1610)).
- A creature projectile hits the player if it is on the player's tile or its next tile is the player's; it then disappears, ends sleep and rest (no `WAKE_UP`), and deals damage reduced by armour; sleeping does not multiply it.
- It disappears without effect at the map edge, a solid block or a creature; it crosses water and lava.
- If its next tile is a crafting table or furnace, that block becomes path.
- Player projectiles: [[Craftax combat#Player projectiles]].

## Spawning

At most one spawn attempt per class per step on the current floor ([`spawn_mobs`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2044), [`FLOOR_MOB_SPAWN_CHANCE`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L159)).

| Floor | Passive | Melee | Ranged |
|---|---|---|---|
| 0 | 0.1 | 0.02 + 0.1 (1 - light)^2 | 0.05 |
| 1-6 | 0.1 | 0.06 | 0.05 |
| 7 | 0.0 | 0.06 | 0.05 |
| 8 | none during the fight | waves only | waves only |

- Melee and ranged chances are tripled while the floor has fewer than 8 kills (`monsters_killed < 8`); floor 0 starts at 10, so it is never tripled; passive chances are never tripled.
- A spawn needs a free slot (fewer than the class capacity alive) and a valid tile.
- Valid tiles: grass, path, fire grass or ice grass with no creature; Euclidean distance to the player > 3 and < 14 for passive, > 9 and < 14 for melee and ranged.
- Deep things spawn only on water tiles (same distances).
- Skeletons and zombies in full Craftax spawn on overworld grass as well as path; in Craftax-Classic skeletons need path ([[Craftax-Classic rules]]).
- Spawned creatures start with full health; light does not matter, so creatures spawn in the dark.
- Boss-floor waves: [[Craftax boss fight]].

## Despawning and kills

- A creature despawns when its Manhattan distance to the player is >= 14 at the start of its update (`mob_despawn_distance`); melee and ranged creatures never despawn on floor 8.
- A creature dies when its health reaches <= 0; it drops nothing.
- Killing a melee or ranged creature by any means adds 1 to the floor's kill counter and unlocks its `DEFEAT_*` achievement ([`attack_mob`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L83), [`MOB_ACHIEVEMENT_MAP`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L553)).
- Killing a passive creature with `DO` gives +6 food and its `EAT_*` achievement; it never counts as a kill.
- Creatures do not fight each other, do not heal, and only take damage from the player.
- Creatures on floors the player is not on are frozen ([[Craftax episode and time#What persists]]).
