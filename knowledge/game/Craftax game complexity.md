---
title: "Craftax game complexity"
type: concept
domain: craftax
parent: "[[Game]]"
tags: [craftax, concept, state-space, complexity, reinforcement-learning]
created: 2026-10-20
updated: 2026-10-20
---
# Craftax game complexity

How big a problem Craftax is for reinforcement learning (RL), measured the way game sizes are usually measured, and compared with tic-tac-toe, chess, Go and Craftax-Classic. Every Craftax number comes from the game code at commit `c3c2e0d` (craftax 1.6.1) or from a short Python script run for this note.

## Terms

- **State**: everything the game needs to continue from this moment: the map, the player, the inventory, the creatures, the timers. Two moments with the same state behave the same from then on.
- **State space**: the set of all possible states. Its size is the "number of states".
- **Action space**: the set of moves the player can choose from at each step. Craftax has 43 actions.
- **Branching factor**: how many different moves are available at a typical step.
- **Horizon** (episode length): how many steps one game lasts.
- **Partial observability**: the player sees only part of the state, so two different states can look the same.
- **Upper bound**: a number that is surely at least as large as the true count. If we say "at most 10^100", the truth may be smaller.
- **Power of ten / order of magnitude**: 10^3 = 1,000, 10^6 = 1,000,000. The exponent counts the digits: 10^44 has 45 digits. "One order of magnitude" means a factor of 10.
- **Multiplying powers of ten**: 10^a x 10^b = 10^(a + b), so we can multiply huge numbers by adding exponents. Any number n is 10^(log10 n); for example 37 = 10^1.568.

## Method, step by step

1. **List what a state contains.** Take the state class from the code: [`EnvState`](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax/craftax_state.py#L39) for Craftax and its [Classic counterpart](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/craftax/craftax_classic/envs/craftax_state.py#L34). Each field is one "part".
2. **Drop parts that are computed from other parts.** The creature occupancy map follows from creature positions, ladder positions from the item map, the light map from the world and the placed torches, and `light_level` from the time step. The random-number key is skipped because it only drives rendering noise.
3. **Count the values each part can take**, using the game's constants: 9 floors of 48 x 48 tiles, 37 block types, 5 item types, inventory cap 99, stat maxima, 3 melee + 3 passive + 2 ranged creature slots per floor, and so on ([[Game]] lists them).
4. **Make continuous values countable.** Health and the hidden counters are decimals; we count them in fixed small steps (health in steps of 0.1, food counters in steps of 0.0625). This is an assumption; it changes the total by only a few orders of magnitude.
5. **Multiply all parts.** Independent parts multiply: 2 choices x 3 choices = 6 combinations. In powers of ten we add the exponents.
6. **Read the result as an upper bound.** The product includes impossible combinations: a map with no trees but 0 wood, 99 diamonds on step 0, a creature inside a wall. The true number of reachable states is smaller, but nobody knows by how much.

- The script that does steps 3-5 is about 100 lines of Python using only `math.log10`; the tables below are its output.

## Craftax state count

| Part | Values | Power of ten |
|---|---|---|
| Blocks: 9 floors x 48 x 48 = 20,736 tiles, 37 types each | 37^20736 | 10^32,518 |
| Items layer (torches, ladders): 20,736 tiles, 5 types | 5^20736 | 10^14,494 |
| Player tile, floor, facing | 2304 x 9 x 4 | 10^4.9 |
| Health, food, drink, energy, mana | 131 x 18^3 x 22 | 10^7.2 |
| Sleeping, resting | 2 x 2 | 10^0.6 |
| Hidden counters (hunger, thirst, fatigue, recovery, mana) | 401 x 321 x 641 x 81 x 31 | 10^11.3 |
| XP and three attributes | 9 x 5^3 | 10^3.1 |
| 17 inventory counts from 0 to 99 | 100^17 | 10^34.0 |
| Pickaxe, sword, bow, 4 armour slots | 5 x 5 x 2 x 3^4 | 10^3.6 |
| Enchantments and learned spells | 3 x 3 x 3^4 x 2^2 | 10^3.5 |
| Creatures: 8 slots per floor, each empty or (tile, health, cooldown, type) | product over slots | 10^400.9 |
| Projectiles: 6 slots per floor, each empty or (tile, direction, type) | product over slots | 10^251.4 |
| Growing plants: 10 slots, each empty or (tile, age 0-600) | (1 + 2304 x 601)^10 | 10^61.4 |
| Chests opened, kill counters | 2^9 x 9^9 | 10^11.3 |
| Potion colour mapping | 6! = 720 | 10^2.9 |
| Boss progress and spawn counter | 9 x 8 | 10^1.9 |
| Achievements unlocked | 2^67 | 10^20.2 |
| Time step 0-99,999 | 100,000 | 10^5.0 |
| **Total** | | **10^47,835** |

- The map alone (blocks) gives 68% of the exponent; the player's own situation (stats, inventory, attributes) adds fewer than 100 orders of magnitude.
- Same method for Craftax-Classic (one 64 x 64 map, 17 block types, 3 creature kinds, inventory cap 9): about **10^5,192**, of which the map is 10^5,040.

## What the agent sees

- The agent sees a window of 9 x 11 = 99 tiles around itself ([[Craftax observations and actions]]).
- That is 4.3% of the current floor and 0.48% of all 20,736 tiles of the world (Craftax-Classic: 63 of 4,096 tiles, 1.5%).
- On dark floors it sees even less: unlit tiles are blanked ([[Craftax episode and time#What the player sees]]).
- The symbolic observation has 8,268 numbers: 8,217 describe the 99 visible tiles and 51 describe the player.
- Hidden from the agent: the rest of the map, other floors, creatures out of view, the potion colour mapping, the hidden counters, the growth age of plants.
- So Craftax is **partially observable**: the agent must remember what it saw and infer what it cannot see. Chess, Go and tic-tac-toe are fully observable: the board is the whole state.

## Other size measures

| Measure | Craftax | Source |
|---|---|---|
| Actions per step | 43, all allowed at every step (failed ones do nothing) | [paper §3.3](https://arxiv.org/html/2402.16801v2#S3.SS3) |
| Effective branching | fewer than 43 distinct results in most states (many actions fail), but random events (creature moves, spawns, loot) multiply the possible next states | [[Craftax action effects]] |
| Episode cap | 100,000 steps (Classic 10,000) | [[Craftax episode and time]] |
| Length of a full playthrough | not measured exactly; the paper says successful runs "typically take tens of thousands of frames" and an expert needed about 5 hours of play for a first perfect run | [paper §3.2](https://arxiv.org/html/2402.16801v2#S3.SS2) |
| Typical training episode | "in the hundreds of timesteps" for learning agents | [paper App. C](https://arxiv.org/html/2402.16801v2#A3) |
| Action sequences | 43^1000 = 10^1,633 sequences of 1,000 steps; 43^100000 = 10^163,347 for the full cap | script |
| Achievements | 67, each rewarded once, 226 reward in total | [[Craftax achievements]] |
| Reward sparsity | at most 67 achievement rewards in up to 100,000 steps (0.067% of steps); the only other signal is +-0.1 per health point | [[Craftax scoring and evaluation]] |

## Comparison

| Game | Number of states | Exact or estimate | Actions per turn | Typical game length | Observed | Chance |
|---|---|---|---|---|---|---|
| Tic-tac-toe | 5,478 legal positions (10^3.7); simple upper bound 3^9 = 19,683 (10^4.3) | exact count | at most 9 | at most 9 moves | fully | none |
| Chess | about 4.8 x 10^44 legal positions | estimate (Tromp) | about 35 | about 70 half-moves | fully | none |
| Go (19 x 19) | 2.08 x 10^170 legal positions | exact count (Tromp, 2016) | about 250 | about 211 moves | fully | none |
| Craftax-Classic | at most 10^5,192 | upper bound (this note) | 17 | up to 10,000 steps | partially (7 x 9 view) | yes |
| Craftax | at most 10^47,835 | upper bound (this note) | 43 | up to 100,000 steps; a full win takes tens of thousands | partially (9 x 11 view) | yes |

- Tic-tac-toe: 5,478 legal positions and 255,168 possible games, from exact enumeration ([Wikipedia, Tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe)).
- Chess positions: "approximately 4.8 x 10^44" ([Tromp, Chess Position Ranking](https://tromp.github.io/chess/chess.html)).
- Go positions: exact 171-digit count, about 2.08 x 10^170, determined on 2016-01-20 ([Tromp, Number of legal Go positions](https://tromp.github.io/go/legal.html)).
- Chess game tree: Shannon (1950) estimated about 10^3 choices per pair of moves over about 40 pairs, so about 10^120 possible games, the "Shannon number" ([Wikipedia, Shannon number](https://en.wikipedia.org/wiki/Shannon_number)).
- Branching factors (35 chess, 250 Go) and average lengths (70 and 211 half-moves) from the table in [Wikipedia, Game complexity](https://en.wikipedia.org/wiki/Game_complexity).
- The board-game counts are counts of legal positions (tighter); the Craftax counts are loose upper bounds, so the comparison exaggerates the gap, but even a bound loose by thousands of orders of magnitude leaves Craftax far beyond Go.

## What the numbers mean for RL

- **Tabular methods cannot work.** A table with one value per state (as in tabular Q-learning or the dynamic-programming methods of Sutton and Barto) would need 10^47,835 entries; the commonly quoted estimate for the number of atoms in the observable universe is only about 10^80.
- **Most states are never seen twice.** Every episode builds a new random world, so an agent almost never meets a state it has met before.
- **Generalisation is required.** The agent must use function approximation, usually a neural network, that maps similar observations to similar values and actions ("a zombie next to me is dangerous" on any map).
- **Memory is required.** Because the view is partial, a policy that looks only at the current frame cannot know where the ladder was or which potion colour hurt; recurrent policies (PPO-RNN, PQN-RNN) carry such memory and are among the top scoreboard entries ([[Craftax scoring and evaluation]]).
- **Exploration is hard.** Rewards are rare and late: the deep achievements need hundreds or thousands of correct steps first (tools, ladders, kills), so random actions almost never find them.
- **Long horizons make credit assignment hard.** An action (keeping a sapphire) may pay off thousands of steps later; the agent must learn which earlier choices caused a later reward.

## Limits of this estimate

- It is an upper bound: impossible and unreachable states are counted; the true reachable count is unknown.
- It depends on choices: counting health in 0.1 steps, or treating each counter value as different, changes the total by a few orders of magnitude, not its scale.
- The map dominates the count, but most map details (which grass tile is where) barely matter to a good policy; a "number of strategically different situations" would be far smaller.
- Size is not difficulty: Go has fewer states than Craftax-Classic yet needed decades of research, while Craftax-Classic is mostly solved by PPO-RNN in 1B steps ([paper App. F.3](https://arxiv.org/html/2402.16801v2#A6.SS3)).
- Other things make a game hard for RL: randomness, partial observability, sparse rewards, long horizons, and the need to learn rules (like the potion mapping) inside each episode.
