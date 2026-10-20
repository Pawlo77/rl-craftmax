---
title: "Tech-Tree Potential-Based Reward Shaping"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - reward-shaping
  - tech-tree
  - sparse-reward
origin: "Ng, Harada, Russell. Policy invariance under reward transformations: Theory and application to reward shaping. ICML 1999. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf"
created: 2026-10-20
---
# Tech-Tree Potential-Based Reward Shaping

## Idea in plain words

- Give the agent a score $\Phi(s)$ for "how far along the tech tree am I": tools, materials needed for the next recipe, floor reached.
- Pay the change of that score as an extra reward: $F = \gamma\,\Phi(s') - \Phi(s)$.
- Getting closer to the next achievement pays a little; moving away costs a little.
- With this exact form, the best policy of the shaped game is the best policy of the original game (policy invariance). Shaping only speeds up learning.
- $\Phi$ is computed automatically from the recipe table, not tuned by hand per item.

## Why it fits Craftax

- Craftax has an explicit dependency graph: wood, table, wood pickaxe, stone, stone pickaxe, furnace, coal and iron, iron pickaxe, diamonds ([[Craftax crafting and placing]], [[Craftax tools weapons and armour]]). Each step costs many actions with no reward until the achievement fires.
- The deep floors add gates: 8 kills to open a ladder, then `DESCEND` ([[Craftax floors and progression]]).
- Every recipe input and tool level is in the symbolic observation, so $\Phi$ is a cheap function of the observation ([[Craftax observations and actions]]).
- Hand-made dense bonuses on Craftax were exploited by the agent ("Agent exploits easy bonuses"; [auto-craftax report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md) §5). A pure potential cannot be farmed in a loop: going up and down returns the same total.

## How it works

Builds on reward design and returns ([[03a MDP setting and returns]]) and value functions ([[03b MDPs, value functions, optimality]]).

1. Write the tech tree as a list of targets $j$ (iron pickaxe, diamond sword, open ladder, next floor, ...) with required inputs and stations.
2. Progress toward target $j$: $p_j(s) = \frac{1}{|I_j|} \sum_{i \in I_j} \min(1, \text{have}_i / \text{need}_i)$, where $I_j$ are its inputs.
3. Potential: $\Phi(s) = \sum_j w_j\, p_j(s) + w_{\text{floor}} \cdot \text{floor}(s) + w_{\text{tool}} \cdot \text{tool levels}(s)$, with $w_j$ equal to the achievement reward of target $j$ (1, 3, 5 or 8) times a small scale.
4. Shaped reward: $r'_t = r_t + \gamma\,\Phi(s_{t+1}) - \Phi(s_t)$, with $\Phi(\text{terminal}) = 0$.
5. Use the same $\gamma$ as PPO (0.99 in the baselines).

- Tools weigh more than their ingredients, so crafting (which consumes ingredients) still raises $\Phi$.
- Report the unshaped return only.

## How to implement here

- A reward wrapper around the environment, about 80-120 lines: read inventory and tool levels from `env_state.inventory` and floor from `env_state.player_level`, compute $\Phi$ before and after `step`, add the difference.
- PPO code unchanged; log the shaped and the native return separately.
- Extra hyperparameters: scale of $\Phi$ (try 0.1, 0.3, 1.0), $w_{\text{floor}}$, which targets to include.

## Cost and risk

- Compute: negligible. Every scale fits Craftax-1M on a Mac; Craftax-1B needs a GPU.
- Risk: mismatched $\gamma$ or a non-zero terminal potential breaks invariance and can reward suicide or stalling.
- Risk: with $\gamma < 1$ a constant high $\Phi$ slowly "costs" reward each step ($\gamma\Phi - \Phi < 0$); keep the scale modest.
- Risk: invariance holds for the optimum, not for what PPO learns in 1B steps; a bad $\Phi$ can still mislead a finite learner.

## Novelty check

- Searched: web "Craftax potential-based reward shaping", GitHub repositories named "craftax", Craftax pull requests.
- Partly done:
  - maskedPPO-GTrXL with reward shaping uses a depth-only potential ($\Phi = 2 \cdot$ floor) plus many non-potential item bonuses, and action masking; 33.8% at 1B in its best run ([repository](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping), scoreboard request [Craftax PR #52](https://github.com/MichaelTMatthews/Craftax/pull/52), closed). The effect of the potential alone is not isolated.
  - auto-craftax implemented PBRS, but it read achievements from `info`, so the shaping was a no-op.
  - SCALAR uses LLM-written dense rewards per skill ([arXiv 2603.09036](https://arxiv.org/abs/2603.09036)).
- New here: a potential built automatically from the full recipe graph, potential-only (policy invariant), no masking, so its effect can be measured cleanly.

## How to evaluate

- Native (unshaped) reward % of max on Craftax-1M and Craftax-1B against blocks D and A of the [[Existing Implementations]] table.
- Compare with the maskedPPO-RS 33.8% claim only as context: it uses masking, extra non-potential rewards and a transformer.
- Ablation: depth-only potential versus full tech-tree potential versus none.
- Achievement success rates along the tool chain (`MAKE_STONE_PICKAXE`, `MAKE_IRON_PICKAXE`, `MAKE_DIAMOND_PICKAXE`) and floor entries.

## References

- Ng, A., Harada, D., Russell, S. "Policy invariance under reward transformations." ICML 1999.
- Doguhan58. craftax-maskedPPO-reward-shaping. https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping
- auto-craftax research report. https://github.com/roger-creus/auto-craftax
