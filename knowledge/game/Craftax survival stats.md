---
title: "Craftax survival stats"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, game-mechanics, intrinsics, attributes]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax survival stats

Health, food, drink, energy and mana (the "intrinsics"), their hidden counters, XP and the three attributes, as computed in [`update_player_intrinsics`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1829), [`get_max_*`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L361) and [`clip_inventory_and_intrinsics`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L381) at commit `c3c2e0d`.

## Maxima and start values

| Stat | Type | Start | Maximum | At attribute 1 / 5 |
|---|---|---|---|---|
| Health | float | 9 | `8 + strength` | 9 / 13 |
| Food | int | 9 | `7 + 2 x dexterity` | 9 / 17 |
| Drink | int | 9 | `7 + 2 x dexterity` | 9 / 17 |
| Energy | int | 9 | `7 + 2 x dexterity` | 9 / 17 |
| Mana | int | 9 | `6 + 3 x intelligence` | 9 / 21 |

- All five are clipped to [0, max] at the end of every step, so potions and food never overshoot the maximum.
- Health is a float because creature damage is scaled by armour and multipliers ([[Craftax combat]]).

## Hidden counters

Each stat changes by one unit when a float counter crosses a threshold; the counter then resets to 0. `c_dex = 1 - 0.125 (dexterity - 1)` (1.0 at dexterity 1, 0.5 at 5). `not_boss` is 0 on floor 8, else 1.

| Counter | Added per step | Threshold | Effect at threshold |
|---|---|---|---|
| `player_hunger` | `c_dex` awake, `0.5 c_dex` asleep | > 25 | food -1 x `not_boss` (min 0) |
| `player_thirst` | `c_dex` awake, `0.5 c_dex` asleep | > 20 | drink -1 x `not_boss` (min 0) |
| `player_fatigue` | `+c_dex` awake; asleep: set to `min(fatigue - 1, 0)` | > 30 / < -10 | energy -1 x `not_boss` / energy +1 (max) |
| `player_recover` | +1 awake, +2 asleep if food > 0, drink > 0 and (energy > 0 or asleep); else -1 awake, -0.5 asleep, times `not_boss` | > 25 / < -15 | health +1 (max) / health -1 |
| `player_recover_mana` | `(counter + 1) x c_int` awake, `(counter + 2) x c_int` asleep, `c_int = 1 + 0.25 (intelligence - 1)` | > 30 | mana +1 |

- Eating resets `player_hunger` to 0; drinking resets `player_thirst` to 0 ([`do_action`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L362), [`attack_mob`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L119)).
- Falling asleep sets a positive fatigue counter to 0 at once (the `min(..., 0)`), then it drops by 1 per step.
- `player_recover` is one shared counter: steps with unmet needs pull it down from wherever regeneration left it.
- The mana multiplier applies to the whole accumulated counter, so with intelligence > 1 it compounds.
- On floor 8 food, drink and energy never fall and unmet needs never drain health; health still regenerates when all needs are met.

## Steps per unit

Computed by iterating the counters from 0 (float rounding may shift a value by one step).

| Dexterity | Food -1 awake / asleep | Drink -1 awake / asleep | Energy -1 awake |
|---|---|---|---|
| 1 | 26 / 51 | 21 / 41 | 31 |
| 2 | 29 / 58 | 23 / 46 | 35 |
| 3 | 34 / 67 | 27 / 54 | 41 |
| 4 | 41 / 81 | 33 / 65 | 49 |
| 5 | 51 / 101 | 41 / 81 | 61 |

| Intelligence | Mana +1 awake | Mana +1 asleep |
|---|---|---|
| 1 | 31 | 16 |
| 2 | 9 | 7 |
| 3 | 6 | 5 |
| 4 | 5 | 4 |
| 5 | 5 | 4 |

- Energy while asleep: +1 every 11 steps (counter 0, -1, ..., -11 < -10), independent of dexterity.
- Health: +1 every 26 steps awake or 13 asleep while all needs are met; -1 every 16 steps awake or 31 asleep while any need is unmet (counter starting at 0).
- Mana has no decay; it only regenerates and is spent on spells (2) and enchanting (9) ([[Craftax magic potions and enchanting]]).

## Damage when a stat is zero

- Food 0, drink 0, or energy 0 while awake makes `player_recover` fall, costing 1 health every 16 awake steps (31 asleep).
- Energy 0 does not hurt while asleep (`energy > 0 or is_sleeping`).
- There is no other penalty for a zero stat; mana 0 only blocks spells and enchanting.

## Restoring stats

| Stat | Source | Amount |
|---|---|---|
| Food | kill a cow, bat or snail with `DO` | +6, hunger counter reset ([[Craftax farming and food]]) |
| Food | `DO` on a ripe plant | +4, hunger counter reset |
| Drink | `DO` facing water or a fountain | +1 per action, thirst counter reset |
| Energy | sleep | +1 per 11 steps |
| Health | needs met | +1 per 26 steps (13 asleep) |
| Health, mana, energy | potions | +8 or -3 ([[Craftax magic potions and enchanting]]) |
| Mana | regeneration | table above |

## XP and attributes

- XP +1 the first time the player enters each of floors 1-8 (8 XP per episode at most); re-entering a floor gives nothing ([`change_floor`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2483)).
- `LEVEL_UP_DEXTERITY`, `LEVEL_UP_STRENGTH`, `LEVEL_UP_INTELLIGENCE` spend 1 XP to raise that attribute by 1, up to `max_attribute = 5` ([`level_up_attributes`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L2970)).
- Attributes start at 1 and never decrease; 12 upgrades exist but only 8 XP can be earned.

| Attribute | Effect per level above 1 | At 5 | Source |
|---|---|---|---|
| Dexterity | max food, drink, energy +2 | 17 | `get_max_food/drink/energy` |
| Dexterity | hunger, thirst, fatigue growth x(1 - 0.125) per level | x0.5 | `update_player_intrinsics` |
| Dexterity | arrow damage +20% | x1.8 | [`_move_player_projectile`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1745) |
| Strength | max health +1 | 13 | `get_max_health` |
| Strength | melee physical damage +25% | x2.0 | [`get_player_damage_vector`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L252) |
| Intelligence | max mana +3 | 21 | `get_max_mana` |
| Intelligence | mana counter multiplier +0.25 | x2.0 | `update_player_intrinsics` |
| Intelligence | fireball and iceball damage +50% | x3.0 | `_move_player_projectile` |
| Intelligence | sword enchantment damage +5% | x1.2 | `get_player_damage_vector` |

- The code comment says intelligence 5 gives "25% more enchant damage"; the formula `1 + 0.05 (int - 1)` gives 20%.
- Intelligence does not affect bow or armour enchantments.
