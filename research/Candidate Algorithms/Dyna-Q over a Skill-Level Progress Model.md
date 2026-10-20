---
title: "Dyna-Q over a Skill-Level Progress Model"
type: candidate
status: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - model-based
  - dyna
  - planning
  - tabular
origin: "Sutton. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming (Dyna). ICML 1990; Sutton & Barto, Reinforcement Learning: An Introduction, Ch. 8 (Dyna-Q, Dyna-Q+, prioritized sweeping)"
created: 2026-10-20
updated: 2026-10-20
---
# Dyna-Q over a Skill-Level Progress Model

## Idea in plain words

- Same skills, progress keys and skill network as [[SMDP Q-Learning over Hand-Defined Skills]].
- New part: remember what happened every time a skill ran from a key: the reward, how long it took, and the key it ended in. This memory is the model.
- Between real updates, replay imagined skill runs drawn from the model and apply the same SMDP Q-learning update. This is called planning.
- This is Dyna-Q from the book, used one level up: on skills and progress keys instead of single moves and screens.
- A rare success (first diamond) is copied back through the whole chain of keys by planning, instead of waiting for many real repeats.
- Skills not tried for a long time get a small bonus (Dyna-Q+), so the agent re-tests old ideas when the world or its own skills change.

## Why it fits Craftax

- The skill-level game is small and close to deterministic: "make a wood pickaxe" from a key with wood and a nearby table ends in the key "has wood pickaxe" almost every time. A lookup-table model learns this exactly.
- Learned world models of the screen are the hard part of the model-based agents on Craftax ([[Simulus (Cohen 2025)]], [[Efficient MBRL (Dedieu 2025)]], [[DreamerV3 (Hafner 2023)]]); a model over progress keys avoids predicting terrain, mobs and light.
- The tech tree is a graph of recipes ([[Craftax crafting and placing]], [[Craftax tools weapons and armour]]); the model learns that graph from experience, with no recipe table given.
- Long chains: the value of a deep key reaches the start key after a few planning sweeps, which helps most at the small Craftax-1M budget.
- Changing situations: new floors, new creatures and improving skills make old outcomes stale; Dyna-Q+ is the book's answer to a model that becomes wrong ([[Craftax floors and progression]]).
- Thousands of parallel envs produce thousands of finished skills per update, so the model fills quickly.

## How it works

Builds on Dyna-Q, Dyna-Q+ and prioritized sweeping ([[08a Models, planning, Dyna]]), value iteration as the full-backup version ([[04 Dynamic programming]]), the SMDP backup ([[14.4 Elevator dispatching]]) and planning dimensions ([[08b Trajectory sampling, search, MCTS]]).

1. Act in real envs exactly as in [[SMDP Q-Learning over Hand-Defined Skills]]; each finished skill gives a record $(k, o, R, \tau, k', \text{done})$.
2. Direct learning: the SMDP Q-learning update on the real record.
3. Model learning: write the record into slot $c \bmod M$ of the model entry for $(k, o)$ ($M$ slots, because outcomes are random); mark $(k, o)$ visited; store the time it was last tried.
4. Planning, $n$ times per real update: sample a visited pair $(k, o)$, sample one of its stored outcomes, and apply the same update with a Dyna-Q+ bonus $\kappa\sqrt{\text{time since last tried}}$ added to $R$.
5. Act ε-greedily on the updated table.

```
model[k, o, count[k, o] % M] = (R, tau, k_next, done)     # after each real skill
count[k, o] += 1;  last_tried[k, o] = t
for _ in range(n_plan):                                   # vectorised batch of pairs
    k, o = sample(visited_pairs)
    R, tau, k2, done = model[k, o, randint(min(count[k, o], M))]
    R_plus = R + kappa * sqrt(t - last_tried[k, o])        # Dyna-Q+
    target = R_plus + gamma**tau * (0 if done else Q_top[k2].max())
    Q_top[k, o] += alpha * (target - Q_top[k, o])
```

- Variant A, prioritized sweeping: sample pairs in proportion to their last $|\text{TD error}|$, so planning works where values just changed.
- Variant B, value iteration: build averages $\hat R$, $\hat\tau$ and next-key frequencies $\hat p(k' \mid k, o)$ for the visited pairs and run full sweeps $Q[k,o] = \hat R + \sum_{k'} \hat p(k')\,\mathbb{E}[\gamma^{\hat\tau}]\max_{o'} Q[k', o']$ over the visited graph.

## How to implement here

- Reuses everything from [[SMDP Q-Learning over Hand-Defined Skills]]: skill table, skill network (purejaxql `pqn_rnn_craftax.py`, [[PQN-RNN (Gallici 2025)]]), progress key, skill wrapper and logging; Dyna is a switch on top of it.
- New code, about 120-200 lines:
  - model arrays `model_R`, `model_tau`, `model_next`, `model_done` of shape `(N, K, M)`, plus `count` and `last_tried` of shape `(N, K)`;
  - a list of visited pair indices (append-only array with a write pointer) for uniform sampling;
  - a planning step as `jax.lax.fori_loop` over batches of sampled pairs, each batch updated with `.at[].add`.
- Extra hyperparameters: planning updates per real update $n$ (1-20 x the number of finished skills), slots $M$ (8-32), Dyna-Q+ $\kappa$ (0.001-0.01), prioritized or uniform sampling, sweeps for the value-iteration variant.
- Logging: number of visited pairs, model accuracy (share of real outcomes whose next key was already in the slots), value of the start key over time.

## Risks

- The key ignores the map, so one $(k, o)$ pair can end differently depending on unseen terrain; the slots average this, but values can be over- or under-estimated.
- Skills keep improving, so old outcomes describe an older, worse skill; small $M$ and the Dyna-Q+ bonus keep the model recent.
- Planning only helps the top level; if skill learning is the bottleneck, Dyna and plain SMDP Q-learning will score the same (a useful result in itself).
- Hash collisions corrupt the model; log collisions on a sample of keys.
- A large $\kappa$ makes the agent keep re-trying skills it already knows.

## Novelty check

- Searched (2026-10-20): Semantic Scholar citations of arXiv 2402.16801 (106 papers, filtered for Dyna, planning, graph, model terms); arXiv API `all:craftax` and `crafter AND (planning OR tree)`; web "Crafter Dyna-Q abstract model planning achievements graph", "Craftax MuZero tree search"; GitHub repository search "craftax dyna", "craftax planning".
- Closest work:
  - SEA on Crafter ([arXiv 2305.00508](https://arxiv.org/abs/2305.00508)): recovers an achievement dependency graph with a heuristic and walks it with a rule-based controller; no values, no planning for reward.
  - [[Rule Learner Planner (dargilboa 2026)]]: learns probabilistic rules and plans experiments by backward chaining, reward-free, Craftax-Classic only.
  - [[AHC (Zhu 2026)]]: an engine-informed dependency graph written by hand, used for a curriculum, not as a learned model.
  - [[SCALAR (Zabounidis 2026)]]: STRIPS planning over LLM-written operators.
  - Neural world-model Dyna on Craftax: [[Efficient MBRL (Dedieu 2025)]], [[Simulus (Cohen 2025)]], [[EASimulus (Peng 2026)]], [[ITC (Kim 2026)]], and the candidate [[Off-Policy Dyna with Imagined Data]].
  - [[Multitask Preplay (Carvalho 2025)]]: Dyna-style replay with the true simulator, not a learned model.
- Not found: a learned lookup-table model of skill outcomes used for Dyna-Q planning on the native Craftax reward.
- New here: the model is learned from experience at the skill level, planning is the book's Dyna-Q / Dyna-Q+, and it is compared head to head with the model-free version on the same skills.

## How to evaluate

- Main comparison: Dyna against [[SMDP Q-Learning over Hand-Defined Skills]] at the same budget and seeds (planning against model-free at the same abstraction).
- Craftax-1M against block D of [[Existing Implementations]] (PPO-RNN 2.3%; [[Simulus (Cohen 2025)]] 6.59% and [[EASimulus (Peng 2026)]] 7.23% as model-based references); this is where planning should help most.
- Craftax-1B against block A (PPO-RNN 15.3%, PQN-RNN 16.0%); flag the hand-written skills as prior knowledge.
- Score: native reward as % of 226, 5+ seeds; achievement success for `MAKE_IRON_PICKAXE`, `COLLECT_DIAMOND`, `ENTER_DUNGEON`, `ENTER_GNOMISH_MINES`.
- Learning speed: steps until the first and the tenth `COLLECT_DIAMOND`, with and without planning.
- Ablations: $n = 0$ (equals SMDP Q-learning), uniform against prioritized sampling, with and without the Dyna-Q+ bonus.

## References

- Sutton, R. S. "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming." ICML 1990.
- Moore, A. W., Atkeson, C. G. "Prioritized sweeping: Reinforcement learning with less data and less time." Machine Learning 13, 1993.
- Sutton, R. S., Barto, A. G. Reinforcement Learning: An Introduction, Chapter 8.
- Zhou, Z., Garg, A. "Learning Achievement Structure for Structured Exploration in Domains with Sparse Reward." ICLR 2023. https://arxiv.org/abs/2305.00508
