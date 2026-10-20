---
title: "Craftax combat"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, combat, damage]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax combat

Damage formulas of full Craftax from [game_logic_utils.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L233) and [`update_mobs`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1091) at commit `c3c2e0d`. Creature stats: [[Craftax creatures]]; equipment: [[Craftax tools weapons and armour]].

## Damage vectors

- Every attack is a vector (physical, fire, ice); every target has a defence vector of fractions in [0, 1].
- Damage dealt = `sum over the three types of (1 - defence) x damage` ([`get_damage`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L274)).
- Damage is a float; health is a float; there is no randomness and no critical hit in damage.

## Player melee

`DO` facing a creature deals ([`get_player_damage_vector`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L252)):

- physical = `[1, 2, 3, 5, 8][sword] x (1 + 0.25 (strength - 1))`;
- fire = `0.5 x [1, 2, 3, 5, 8][sword] x (1 + 0.05 (intelligence - 1))` if the sword has the fire enchantment, else 0;
- ice = same with the ice enchantment.

- Only the first creature found on the facing tile is hit; melee, passive and ranged classes are checked in that order ([`attack_mob`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L83)).
- A `DO` that hits a creature does not also mine the block behind it.
- The player has no attack cooldown: one hit per step.

## Player projectiles

| Projectile | Base damage | Multiplier | Max at attribute 5 |
|---|---|---|---|
| Arrow (bow, type `ARROW2`) | 5 physical, + 2.5 fire or ice with a bow enchantment | `1 + 0.2 (dexterity - 1)` on all parts | 9 physical + 4.5 elemental |
| Fireball (spell) | 3 fire | `1 + 0.5 (intelligence - 1)` | 9 fire |
| Iceball (spell) | 3 ice | `1 + 0.5 (intelligence - 1)` | 9 ice |

- Fired from the player's tile in the facing direction; at most 3 player projectiles per floor ([`_move_player_projectile`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1713)).
- Each step the projectile damages a creature on its current tile, or else on its next tile, then disappears; otherwise it moves 1 tile and disappears at the map edge or a solid block.
- Projectiles cross water and lava; they cannot damage the necromancer.
- Projectile kills count for the floor's kill counter and `DEFEAT_*` achievements; passive creatures killed by projectiles give no food.

## Damage to the player

`get_damage_done_to_player` ([L233](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L233)):

- defence physical = `0.1 x (sum of armour slot levels)` (0 to 0.8);
- defence fire = `0.2 x (number of fire-enchanted slots)` (0 to 0.8);
- defence ice = `0.2 x (number of ice-enchanted slots)` (0 to 0.8);
- incoming damage x1.5 on floor 8 (`BOSS_FIGHT_EXTRA_DAMAGE = 0.5`);
- melee damage x3.5 while asleep (applied before armour).

| Example | Raw | No armour | Full diamond (0.8 phys) | Full diamond + 4 fire slots |
|---|---|---|---|---|
| Zombie hit | 2 | 2 | 0.4 | 0.4 |
| Zombie hit while asleep | 7 | 7 | 1.4 | 1.4 |
| Pigman hit | 3 + 5 | 8 | 0.6 + 5 = 5.6 | 0.6 + 1 = 1.6 |
| Frost troll hit on floor 8 | (4 + 5) x 1.5 | 13.5 | 1.2 + 7.5 = 8.7 | 8.7 |

- Values computed from the formulas above.
- Fire and ice defences are independent, so 4 slots split as 2 fire + 2 ice give 0.4 against each.

## Elements

- Knights and knight archers block 50% of physical damage; trolls 20%.
- Pigmen and fire elementals block 90% of physical and 100% of fire, so ice damage is the effective tool against them.
- Frost trolls and ice elementals block 90% of physical and 100% of ice, so fire damage is the effective tool.
- Other creatures have no defence.
- Fire and ice reach the player from orc mages, trolls, deep things, pigmen, fire elementals, frost trolls and ice elementals ([[Craftax creatures#Stat table]]).

## Hits needed (worked examples)

Computed from the formulas and creature table.

| Attacker | Target | Damage per hit | Hits |
|---|---|---|---|
| Diamond sword, strength 1, no enchantment | Knight (12 HP, 0.5 phys) | 8 x 0.5 = 4 | 3 |
| Diamond sword, strength 5, ice enchant, int 1 | Pigman (20 HP) | 16 x 0.1 + 4 = 5.6 | 4 |
| Diamond sword, strength 5, fire enchant, int 5 | Frost troll (24 HP) | 1.6 + 4.8 = 6.4 | 4 |
| Iceball, intelligence 1 | Fire elemental (14 HP) | 3 | 5 |
| Iceball, intelligence 5 | Fire elemental (14 HP) | 9 | 2 |
| Bare hands, strength 1 | Pigman (20 HP) | 0.1 | about 200 |

## Other damage sources

- Food, drink or energy at 0: -1 health per 16 awake steps ([[Craftax survival stats]]).
- The -3 health potion ([[Craftax magic potions and enchanting]]).
- No fall, lava, drowning or darkness damage exists in full Craftax.
