---
title: "Craftax farming and food"
type: concept
domain: craftax
parent: "[[Craftax game rules]]"
tags: [craftax, concept, game-mechanics, food, farming]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax farming and food

Saplings, plants, eating and drinking in full Craftax at commit `c3c2e0d`. Counter arithmetic: [[Craftax survival stats]].

## Saplings and plants

- `DO` facing grass gives 1 sapling with probability 0.1; the grass stays ([`do_action`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L343)).
- Grass exists only on floor 0, so saplings are collected and planted only there.
- `PLACE_PLANT` turns the facing grass tile (no item, no creature) into a `PLANT` block and registers it in a free growth slot ([`add_new_growing_plant`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L796)).
- There are 10 growth slots (`max_growing_plants`) and they are never freed, so at most the first 10 saplings planted in an episode can ever ripen; later plants stay unripe forever.
- Each tracked plant's age grows by 1 every step on any floor; at age >= 600 the block becomes `RIPE_PLANT` ([`update_plants`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py#L1978)).
- `DO` on a ripe plant: food +4 (capped), hunger counter reset, `EAT_PLANT`, the block returns to `PLANT` and its age to 0, so it ripens again 600 steps later.
- `DO` on an unripe plant does nothing; plants are solid and cannot be destroyed by the player or creatures.
- Before 1.5.0 any `DO` reset slot 0's age ([[Craftax known issues]]).

## Eating

| Food | How | Food gained | Achievement |
|---|---|---|---|
| Cow (floor 0) | kill with `DO` (melee) | +6 | `EAT_COW` |
| Bat (floors 2, 5, 6) | kill with `DO` | +6 | `EAT_BAT` |
| Snail (floors 1, 3, 4) | kill with `DO` | +6 | `EAT_SNAIL` |
| Ripe plant (floor 0) | `DO` | +4 | `EAT_PLANT` |

- Food is capped at `7 + 2 x dexterity`; eating at full food wastes the surplus but still resets the hunger counter.
- A passive creature killed by an arrow or spell gives no food and no achievement ([`attack_mob`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L83), `can_eat` is False for projectiles).
- Passive kills do not count toward the 8 kills that open a ladder.
- Floor 7 spawns no passive creatures (spawn chance 0.0) and floor 8 spawns none during the boss fight, so neither has food ([`FLOOR_MOB_SPAWN_CHANCE`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L159)).

## Drinking

- `DO` facing water or a fountain: drink +1 (capped), thirst counter reset, `COLLECT_DRINK`; the source is never used up.
- Water by floor: lakes (0), fountains in about half the rooms (1, 4), water tiles (3), pools (2, 5), sea and pockets (7); none on floors 6 and 8 ([[Craftax blocks and resources#Resource spawn rules per floor]]).
- Floor 8 needs no food or drink: food, drink and energy do not fall there.
- Potions do not restore food or drink.
