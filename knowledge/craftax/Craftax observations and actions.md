---
title: "Craftax observations and actions"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, observation-space, action-space]
created: 2026-10-20
---
# Craftax observations and actions

All observations are `float32` and centred on the player; shapes below were checked by running craftax 1.6.1. Code wins over [obs_description.md](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/obs_description.md) and the paper where they differ (see the last section).

## Shapes

| Env | Observation | Shape | Source |
|---|---|---|---|
| `Craftax-Symbolic-v1` | flat vector | (8268,) | [craftax_symbolic_env.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/envs/craftax_symbolic_env.py) |
| `Craftax-Pixels-v1` | RGB image in [0, 1] | (130, 110, 3) = ((9 + 4) x 10, 11 x 10, 3) | [craftax_pixels_env.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/envs/craftax_pixels_env.py) |
| `Craftax-Classic-Symbolic-v1` | flat vector | (1345,) | [classic craftax_symbolic_env.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/envs/craftax_symbolic_env.py) |
| `Craftax-Classic-Pixels-v1` | RGB image in [0, 1] | (63, 63, 3) = ((7 + 2) x 7, 9 x 7, 3) | [classic craftax_pixels_env.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/envs/craftax_pixels_env.py) |

- Pixel image = map view on top plus an inventory band of `INVENTORY_OBS_HEIGHT` tile rows (4 in Craftax, 2 in Classic), divided by 255.
- Craftax uses 10-pixel tiles instead of 7 because two-digit counts were unreadable at 7 ([paper App. D.4](https://arxiv.org/html/2402.16801v2#A4.SS4), fixed in code after [issue #19](https://github.com/MichaelTMatthews/Craftax/issues/19)).
- At night on the overworld the pixel renderer darkens the view and adds noise drawn from `state.state_rng`; dark floors show only lit tiles ([renderer.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L201)).
- The paper experiments use symbolic observations only; symbolic runs about 10x faster than pixels ([paper §3.1, §3.4](https://arxiv.org/html/2402.16801v2#S3.SS4)).

## Craftax symbolic layout

From `render_craftax_symbolic` in [renderer.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/renderer.py#L9).

| Slice | Size | Content | Encoding |
|---|---|---|---|
| `[0:8217]` | 9 x 11 x 83 | Map view, reshape to `(9, 11, 83)` (rows, cols, channels) | per tile, below |
| `[8217:8233]` | 16 | wood, stone, coal, iron, diamond, sapphire, ruby, sapling, torches, arrows | `sqrt(n) / 10` |
| | | books | `n / 2` |
| | | pickaxe level, sword level | `level / 4` |
| | | sword enchantment, bow enchantment | 0 none, 1 fire, 2 ice |
| | | bow | 0 or 1 |
| `[8233:8239]` | 6 | potions red, green, blue, pink, cyan, yellow | `sqrt(n) / 10` |
| `[8239:8248]` | 9 | health, food, drink, energy, mana, xp, dexterity, strength, intelligence | `x / 10` |
| `[8248:8252]` | 4 | facing direction left, right, up, down | one-hot |
| `[8252:8256]` | 4 | armour helmet, chestplate, leggings, boots | `level / 2` (0.5 iron, 1 diamond) |
| `[8256:8260]` | 4 | armour enchantments (same order) | 0 none, 1 fire, 2 ice |
| `[8260:8268]` | 8 | day light level, sleeping, resting, learned fireball, learned iceball, floor / 10, floor ladder open (8 kills), boss vulnerable | float / flags |

Per-tile channels of the map view:

| Channels | Size | Content |
|---|---|---|
| `[0:37]` | 37 | `BlockType` one-hot (0 `INVALID` ... 36 `NECROMANCER_VULNERABLE`) |
| `[37:42]` | 5 | `ItemType` one-hot: `NONE`, `TORCH`, `LADDER_DOWN`, `LADDER_UP`, `LADDER_DOWN_BLOCKED` |
| `[42:82]` | 40 | creature one-hot, index `class * 8 + type_id`; classes 0 melee, 1 passive, 2 ranged, 3 enemy projectile, 4 player projectile |
| `[82]` | 1 | tile visible: 1 if tile light > 0.05, else 0 and the 82 channels above are zeroed |

Creature `type_id` per class ([constants.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py), texture lists):

| type_id | Melee | Passive | Ranged | Projectile |
|---|---|---|---|---|
| 0 | zombie | cow | skeleton | arrow |
| 1 | gnome warrior | bat | gnome archer | dagger |
| 2 | orc soldier | snail | orc mage | fireball |
| 3 | lizard | | kobold | iceball |
| 4 | knight | | knight archer | arrow2 |
| 5 | troll | | deep thing | slimeball |
| 6 | pigman | | fire elemental | fireball2 |
| 7 | frost troll | | ice elemental | iceball2 |

- Out-of-map tiles are `OUT_OF_BOUNDS` (block 1).
- Values above 1 occur although `observation_space` is `Box(0, 1)`: enchantment channels reach 2 and intrinsics reach `max / 10` (mana up to 6 + 3 x 5 = 21, so 2.1) ([game_logic_utils.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/util/game_logic_utils.py#L361); paper [Table 4](https://arxiv.org/html/2402.16801v2#A4.T4) lists the same maxima).

## Craftax-Classic symbolic layout

From [classic renderer.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/renderer.py); the same split was posted in [issue #38](https://github.com/MichaelTMatthews/Craftax/issues/38).

| Slice | Size | Content | Encoding |
|---|---|---|---|
| `[0:1323]` | 7 x 9 x 21 | Map view, per tile 17 `BlockType` one-hot then 4 creature flags (zombie, cow, skeleton, arrow) | one-hot |
| `[1323:1335]` | 12 | wood, stone, coal, iron, diamond, sapling, wood/stone/iron pickaxe, wood/stone/iron sword | `n / 10` (inventory capped at 9) |
| `[1335:1339]` | 4 | health, food, drink, energy | `x / 10` |
| `[1339:1343]` | 4 | direction left, right, up, down | one-hot |
| `[1343]` | 1 | light level | float |
| `[1344]` | 1 | sleeping | flag |

## Actions

`Discrete(43)` for Craftax; Craftax-Classic uses IDs 0-16 with the same meaning ([constants.py `Action`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/constants.py#L76); [paper Table 5](https://arxiv.org/html/2402.16801v2#A4.T5)). Keys are those of `play_craftax`.

| ID | Action | Key | ID | Action | Key |
|---|---|---|---|---|---|
| 0 | `NOOP` | q | 22 | `MAKE_IRON_ARMOUR` | y |
| 1 | `LEFT` | a | 23 | `MAKE_DIAMOND_ARMOUR` | u |
| 2 | `RIGHT` | d | 24 | `SHOOT_ARROW` | i |
| 3 | `UP` | w | 25 | `MAKE_ARROW` | o |
| 4 | `DOWN` | s | 26 | `CAST_FIREBALL` | g |
| 5 | `DO` | space | 27 | `CAST_ICEBALL` | h |
| 6 | `SLEEP` | tab | 28 | `PLACE_TORCH` | j |
| 7 | `PLACE_STONE` | r | 29 | `DRINK_POTION_RED` | z |
| 8 | `PLACE_TABLE` | t | 30 | `DRINK_POTION_GREEN` | x |
| 9 | `PLACE_FURNACE` | f | 31 | `DRINK_POTION_BLUE` | c |
| 10 | `PLACE_PLANT` | p | 32 | `DRINK_POTION_PINK` | v |
| 11 | `MAKE_WOOD_PICKAXE` | 1 | 33 | `DRINK_POTION_CYAN` | b |
| 12 | `MAKE_STONE_PICKAXE` | 2 | 34 | `DRINK_POTION_YELLOW` | n |
| 13 | `MAKE_IRON_PICKAXE` | 3 | 35 | `READ_BOOK` | m |
| 14 | `MAKE_WOOD_SWORD` | 5 (Classic: 4) | 36 | `ENCHANT_SWORD` | k |
| 15 | `MAKE_STONE_SWORD` | 6 (Classic: 5) | 37 | `ENCHANT_ARMOUR` | l |
| 16 | `MAKE_IRON_SWORD` | 7 (Classic: 6) | 38 | `MAKE_TORCH` | [ |
| 17 | `REST` | e | 39 | `LEVEL_UP_DEXTERITY` | ] |
| 18 | `DESCEND` | . | 40 | `LEVEL_UP_STRENGTH` | - |
| 19 | `ASCEND` | , | 41 | `LEVEL_UP_INTELLIGENCE` | = |
| 20 | `MAKE_DIAMOND_PICKAXE` | 4 | 42 | `ENCHANT_BOW` | ; |
| 21 | `MAKE_DIAMOND_SWORD` | 8 | | | |

- Movement actions 1-4 also set the facing direction; `DO` acts on the faced tile (mine, attack, eat, drink, open chest, hit the necromancer) ([game_logic.py](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/game_logic.py)).
- While sleeping or resting every action is replaced by `NOOP`.
- Crafting needs a crafting table (and for iron items a furnace) in one of the 8 surrounding tiles, diagonals included (`is_near_block`).
- Recipes and effects: [[Craftax game content and achievements]].

## Source discrepancies

| Item | Code (1.6.1) | Other source |
|---|---|---|
| Light channel index | 82 | obs_description.md says 83 |
| Light channel value | binary (light > 0.05) | paper App. D.4: "a float between 0 and 1" |
| Creature one-hot size | 40 (5 classes x 8) | paper App. D.4 says 36; total 8268 matches code |
| Potion encoding | `sqrt(n) / 10` | obs_description.md says `n / 10` |
