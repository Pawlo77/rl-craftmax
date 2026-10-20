---
title: "SMDP Q-Learning over Hand-Defined Skills"
type: candidate
status: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - hierarchical-rl
  - options
  - q-learning
  - tabular
origin: "Sutton, Precup, Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in RL. Artificial Intelligence 112, 1999. https://doi.org/10.1016/S0004-3702(99)00052-1; Crites, Barto. Improving Elevator Performance Using Reinforcement Learning. NeurIPS 1995 (SMDP Q-learning)"
created: 2026-10-20
updated: 2026-10-20
---
# SMDP Q-Learning over Hand-Defined Skills

## Idea in plain words

- A skill (an "option" in RL terms) is a small policy for one job with a clear end test, for example "get 1 wood", "place a table", "drink until full", "go down the ladder".
- We write the skill list and the end tests by hand from the game rules: about 12 skills for Craftax-Classic, about 25 for Craftax.
- The top level picks the next skill. It is a plain table $Q[k, o]$: one row per progress key $k$, one column per skill $o$.
- Progress key: a short summary of the state (floor, tool levels, rounded inventory counts, "food/drink/energy low" flags), the same key as in [[Count-Based Progress Bonus]].
- The top level learns with SMDP Q-learning: the ordinary Q-learning update, but one "step" is a whole skill run, and the discount is $\gamma^\tau$ for a skill that took $\tau$ game steps.
- SMDP (semi-Markov decision process): a decision problem where each decision lasts a variable number of time steps.
- The skills are one shared Q-network trained with Q($\lambda$), as in [[PQN-RNN (Gallici 2025)]]; it gets the skill id as input and a reward of 1 when the skill's end test succeeds.
- No PPO anywhere: both levels learn action values with Q-learning.

## Why it fits Craftax

- Tech tree: deep achievements need long chains of jobs ([[Craftax crafting and placing]], [[Craftax achievements]]). At the top level, wood → table → wood pickaxe → stone → stone pickaxe → furnace → iron → iron pickaxe → diamond is about 10 decisions instead of thousands of key presses.
- Small top-level state: only floor, tools and coarse inventory enter the key, so a few thousand keys are visited; a table is enough, and tabular Q-learning needs no network and no tuning of features.
- Native reward: the top level gets the sum of the real rewards earned while the skill ran (achievements plus 0.1 x health change, [[Craftax scoring and evaluation]]), so it optimises the benchmark objective.
- Survival stats: "drink", "eat" and "sleep" skills plus low-stat flags in the key let the top level learn when to look after food, water and energy ([[Craftax survival stats]], [[Craftax farming and food]]).
- Floors: a "descend" skill and the floor in the key; the down ladder on floors 1+ opens only after enough kills, so a "fight nearby creature" skill is needed ([[Craftax floors and progression]], [[Craftax combat]]).
- Skill end tests are cheap to compute from the JAX state: inventory counts, `achievements` flags, floor, intrinsics ([[Craftax items and inventory]], [[Craftax API and wrappers]]).
- Massively parallel envs: every environment runs its own skill; top-level decisions happen at different times in different environments, and the table update is one scatter-add.
- Partial observability: only the low-level skill network sees the 9 x 11 view; it is recurrent, as in PQN-RNN ([[Craftax observations and actions]]).

## How it works

Builds on the semi-Markov Q-learning backup of the elevator case study ([[14.4 Elevator dispatching]]), Q-learning ([[06 Temporal-difference learning]]), Watkins's Q($\lambda$) for the skills ([[07b Eligibility-trace control and off-policy]]), Q-learning with function approximation ([[09b Control with approximation, bootstrapping]]) and optimistic initial values ([[02 Multi-armed bandits]]).

1. Skill list: $K$ skills; each has an end test on the state (success), a step limit $T_{\max}$, and fails on death or episode end.
2. Progress key: $k = \text{hash}(\phi(s)) \in [0, N)$.
3. When an environment has no active skill, the top level picks $o$ with ε-greedy over $Q[k, \cdot]$ and stores $k_{\text{start}} = k$, $R = 0$, $\tau = 0$.
4. Every game step, the skill network picks the action: ε-greedy over $Q_{\text{low}}(\text{obs}, h, o, \cdot)$. Accumulate $R \mathrel{+}= \gamma^\tau r_t$ (native reward) and $\tau \mathrel{+}= 1$.
5. When the skill ends (success, step limit, death), apply the SMDP Q-learning update:

$$Q[k, o] \leftarrow Q[k, o] + \alpha\Big[R + \gamma^{\tau}\,\max_{o'} Q[k', o'] - Q[k, o]\Big], \qquad R = \sum_{i=0}^{\tau-1} \gamma^{i} r_{t+i}$$

   with the bootstrap term set to 0 if the episode ended.
6. Train the skill network like PQN: λ-returns of the skill reward $r^o_t$ (1 on success, 0 otherwise; the skill's run ends on success or failure), minibatches and epochs over the rollout.

```
if skill_done[i]:                                  # per environment i
    target = R[i] + gamma**tau[i] * (0 if dead[i] else Q_top[k_new[i]].max())
    Q_top[k_start[i], skill[i]] += alpha * (target - Q_top[k_start[i], skill[i]])
    skill[i] = eps_greedy(Q_top[k_new[i]])         # next skill
    k_start[i], R[i], tau[i] = k_new[i], 0, 0
a = eps_greedy(Q_low(obs, h, one_hot(skill)))      # every step
R += gamma**tau * r_native;  tau += 1
r_skill = success(skill, state_before, state_after)   # 0 or 1, trains Q_low
```

- Optimistic start: initialise $Q[k, o]$ above any real value (for example 10), so the top level tries every skill at least once at each new key.
- Ablation: replace step 5 with a Monte Carlo update of the top level ([[05a Monte Carlo prediction and control]]) to measure what bootstrapping adds.

## How to implement here

- Skill network: purejaxql `pqn_rnn_craftax.py` ([[PQN-RNN (Gallici 2025)]]) with the skill one-hot added to the input next to the last action.
- Scaffolding: environment creation, recording and evaluation from `src/wut_murl_craftax/training/ppo.py` and `training/records.py`; the progress-key function shared with [[Count-Based Progress Bonus]].
- New code, about 300-450 lines:
  - skill table: one JAX function per skill returning `(success, failed)` from `(state_before, state_after)`, selected with `jax.lax.switch` on the skill id;
  - skill wrapper state per environment: `skill`, `k_start`, `R`, `tau`, `steps_in_skill`;
  - top-level table `Q_top = jnp.full((N, K), q_init)` updated with `Q_top.at[k_start, skill].add(alpha * td * skill_done)`;
  - logging of per-skill success rate, mean skill length and the greedy skill sequence from the start key.
- Read unlocks from `env_state.achievements`, not from `info` ([[Craftax Baselines benchmark protocol]]).
- Extra hyperparameters: skill list, $T_{\max}$ per skill (300-1000 steps), table size $N$ ($2^{16}$-$2^{20}$), $\alpha$ (0.05-0.2), top-level ε schedule, optimistic $q_{\text{init}}$, skill reward for death (0 or -1).
- Warm-up option: a first phase with a uniform random top level, so every skill gets practice before the top level becomes greedy.

## Risks

- Prior knowledge: the skill list and end tests are hand-written from the rules, like the masks and shaping of [[maskedPPO-GTrXL (Doguhan58 2026)]]; results must be reported apart from block A of [[Existing Implementations]].
- Deep skills ("get diamond") succeed rarely early on, so the top level sees few samples for them; optimistic initial values and the warm-up phase are the planned fixes.
- The key ignores the map (is water nearby?), so the top level is not exactly Markov; its values are averages over the situations behind one key.
- One shared skill network: skills can interfere with each other; the skill one-hot may need its own small head per skill.
- Step limits: too short and hard skills always fail; too long and a stuck skill wastes the episode.
- Hash collisions with a small table; log the fill rate.

## Novelty check

- Searched (2026-10-20): Semantic Scholar citations of arXiv 2402.16801 (106 papers, titles and abstracts filtered for option, hierarchical, SMDP, skill, Q-learning terms); arXiv API `all:craftax` (30 papers) and `crafter AND (hierarchical OR options OR planning)`; web "Craftax hierarchical DQN options SMDP Q-learning", "Crafter hierarchical options meta-controller Q-learning"; GitHub repository search "craftax options", "craftax hierarchical".
- Closest work:
  - SCALAR on Craftax: LLM-written skills, a STRIPS planner as the top level, PPO options ([[SCALAR (Zabounidis 2026)]]).
  - SEA on Crafter (Zhou, Garg, ICLR 2023, [arXiv 2305.00508](https://arxiv.org/abs/2305.00508)): learned achievement graph, a rule-based meta-controller that picks a random unfinished achievement whose prerequisites are done, achievement-conditioned sub-policies.
  - h-DQN (Kulkarni et al. 2016, [arXiv 1604.06057](https://arxiv.org/abs/1604.06057)) and H²RL (2026, [arXiv 2603.06565](https://arxiv.org/abs/2603.06565)): DQN meta-controller over hand-defined goals, on Atari; neither mentions Craftax.
  - Scalable Option Learning (2025, [arXiv 2509.00338](https://arxiv.org/abs/2509.00338)): options at high throughput, NetHack and MiniHack; cites Craftax without experiments.
  - [[Achievement Skill Options]]: the PPO version of this idea (PPO top level, PPO options).
  - [[Multitask Preplay (Carvalho 2025)]]: goal-conditioned Q($\lambda$) on Craftax, but with the true simulator as model and no top-level SMDP learner.
- Not found: a learned SMDP Q-learning top level over a progress key, with value-based skills, on Craftax or Craftax-Classic.
- New here: the top level is a learned table (not rule-based as in SEA, not an LLM plan as in SCALAR), both levels are Q-learning, and the setting is full Craftax with floors and survival.

## How to evaluate

- Craftax-Classic first, to debug the skill list (22 achievements, about 12 skills).
- Craftax-1M against block D of [[Existing Implementations]] (PPO-RNN 2.3%, RND 2.2%) and Craftax-1B against block A (PPO-RNN 15.3%, PQN-RNN 16.0%); every primitive game step counts toward the budget; flag the prior knowledge.
- Score: native reward as % of the maximum (226), our own PPO-RNN and PQN-RNN reruns as controls, 5+ seeds.
- Achievement success along the chain: `MAKE_STONE_PICKAXE`, `MAKE_IRON_PICKAXE`, `COLLECT_DIAMOND`, `ENTER_DUNGEON`, `ENTER_GNOMISH_MINES` (SCALAR reaches 9.1% for the last, under a different protocol).
- Ablations: random top level against learned top level; Monte Carlo top level against SMDP Q-learning; with and without low-stat flags in the key.
- Partner run: [[Dyna-Q over a Skill-Level Progress Model]] on the same skills isolates the effect of planning.

## References

- Sutton, R. S., Precup, D., Singh, S. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial Intelligence 112, 1999.
- Crites, R. H., Barto, A. G. "Improving Elevator Performance Using Reinforcement Learning." NeurIPS 1995.
- Kulkarni, T. D. et al. "Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation." NeurIPS 2016. https://arxiv.org/abs/1604.06057
- Zhou, Z., Garg, A. "Learning Achievement Structure for Structured Exploration in Domains with Sparse Reward." ICLR 2023. https://arxiv.org/abs/2305.00508
- Gallici, M. et al. "Simplifying Deep Temporal Difference Learning." ICLR 2025. https://arxiv.org/abs/2407.04811
- "SCALAR." 2026. https://arxiv.org/abs/2603.09036
