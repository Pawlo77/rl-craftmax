---
title: "Craftax boss fight"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, game-mechanics, boss]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax boss fight

The necromancer fight on floor 8 (the Graveyard), the only way to win full Craftax, from [game_logic_utils.py L7-30](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L7), [`do_action`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L441), [`boss_logic`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2830) and [`spawn_mobs`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2044) at commit `c3c2e0d`.

## Arena

- Floor 8 is a dark path arena near the map centre, ringed by walls, with graves on about 5% of path tiles ([[Craftax floors and progression#Smooth floors]]).
- The necromancer is a solid block at the map centre (24, 24); it never moves or attacks itself.
- The player arrives at the floor's stored up-ladder tile, in a 9 x 9 lit patch; there is no ladder item, so the player cannot leave.
- "Fighting the boss" (`is_fighting_boss`) simply means being on floor 8.

## Floor 8 rules

- Food, drink and energy never fall, and unmet needs never drain health ([[Craftax survival stats]]).
- All damage to the player is x1.5 (`BOSS_FIGHT_EXTRA_DAMAGE = 0.5`).
- No passive creatures spawn; melee creatures always chase (the 75% chase roll still applies); melee and ranged creatures never despawn.

## Waves

- A counter `boss_timesteps_to_spawn_this_round` starts at 7 (`BOSS_FIGHT_SPAWN_TURNS`) at reset and drops by 1 every step spent on floor 8.
- While the counter is >= 1 after that drop, the melee and ranged spawn chances are multiplied by 1000 (and by 3 while floor 8 has < 8 kills), so a spawn is certain whenever a slot and a tile are free: 6 spawn steps per wave.
- Wave creatures spawn on grave tiles within Euclidean distance 6 of the player (and < 14), not on occupied tiles; graves are solid, so creatures appear inside them and step out.
- At most 3 melee and 2 ranged creatures are alive at once; killed ones are refilled while the counter is >= 1.
- Wave k (k = `boss_progress`, 0-7) uses the melee and ranged types of floor k:

| Wave | Melee | Ranged |
|---|---|---|
| 0 | zombie | skeleton |
| 1 | orc soldier | orc mage |
| 2 | gnome warrior | gnome archer |
| 3 | lizard | kobold |
| 4 | knight | knight archer |
| 5 | troll | deep thing |
| 6 | pigman | fire elemental |
| 7 | frost troll | ice elemental |

- Deep things spawned in graves cannot leave them (they move only in water) and shoot from there.
- If no grave lies within 6 tiles of the player during the spawn steps, the wave is empty.

## Hitting the necromancer

- The necromancer is vulnerable when no melee and no ranged creature is alive on floor 8 and the counter is <= 0 (`is_boss_vulnerable`); it is then drawn as `NECROMANCER_VULNERABLE`, and the symbolic observation has a boss-vulnerable flag.
- `DO` facing it while vulnerable adds 1 to `boss_progress`, unlocks `DAMAGE_NECROMANCER` (8 reward, first time), and resets the counter to 7, starting the next wave.
- Weapon, strength and enchantments do not matter: each hit is one progress point; arrows and spells cannot hit it.
- `DO` while it is not vulnerable does nothing.

## Winning

- The boss is beaten when `boss_progress >= 8` (`num_levels - 1`), that is after 8 hits and 8 cleared waves (waves 0-7).
- That step unlocks `DEFEAT_NECROMANCER` (8 reward) and ends the episode (`is_game_over`).
- Floor 8 achievements: `ENTER_GRAVEYARD` (8), `DAMAGE_NECROMANCER` (8), `DEFEAT_NECROMANCER` (8); wave kills also unlock any missing `DEFEAT_*` achievements of earlier floors.
