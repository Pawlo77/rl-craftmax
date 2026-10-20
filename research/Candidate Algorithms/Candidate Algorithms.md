---
title: "Candidate Algorithms"
type: index
parent: "[[wut_murl_craftax]]"
tags:
  - index
  - craftax
  - candidates
  - reinforcement-learning
created: 2026-10-20
---
# Candidate Algorithms

Algorithms the project could try on Craftax beyond the published agents in [[Existing Implementations]]. Each builds on the PPO or PPO-RNN baseline ([[Craftax Baselines]]) and on textbook ideas from [[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]. Novelty was checked on 2026-10-20 (method at the end).

## Top 5 recommendations

For a one-semester student project: a small team, a Mac plus maybe one GPU, building on PPO-RNN.

- Why 5: each pick attacks a different Craftax difficulty (exploration, rare successes, sparse crafting chain, partial observability, the floor wall), so together they cover every truly promising direction without overlap.
- Left out: [[Per-Achievement Value Heads]] (cheap and novel, but a smaller critic tweak; next in line), [[Bandit-Tuned Exploration]] and [[Population-Based Training]] (tune other methods rather than add an idea), and the suggested-by notes that are taken or beyond student level.

| # | Candidate | Craftax difficulty it attacks | Lines of code | First experiment | Success means |
|---|---|---|---|---|---|
| 1 | [[Count-Based Progress Bonus]] | Exploration bonuses that chase new scenery | 60-100 | Craftax-1M on the Mac, 5 seeds | Above RND/E3B (2.2%) at 1M |
| 2 | [[Self-Imitation Learning]] | Rare one-time successes | 120-180 | Craftax-Classic, then Craftax-1M, 5 seeds | Above PPO-RNN (2.3%) at 1M |
| 3 | [[Tech-Tree Potential-Based Reward Shaping]] | Long reward-free crafting chain | 80-120 | Craftax-1M on the Mac, 5 seeds | Above PPO-RNN (2.3%) at 1M on native reward |
| 4 | [[Explicit Map Memory]] | Seeing 4% of the floor; dark floors | 130-180 | Craftax-1M on the Mac, 5 seeds | Above PPO-RNN (2.3%) at 1M, fair-position variant |
| 5 | [[Go-Explore with Achievement Cells]] | No agent passes the floor-2 wall | 150-250 | 200M steps on one GPU | Repeatable `ENTER_GNOMISH_MINES`; above 15.3% at 1B |

### 1. [[Count-Based Progress Bonus]]

- Most interesting: it tests the Craftax paper's own explanation of why exploration bonuses failed (the bonus looks at scenery). A bonus over game progress only is a direct, clean test of that claim.
- Feasible: no new network, about 60-100 lines, negligible compute; a textbook idea (UCB-style bonus, [[02 Multi-armed bandits]]).
- Minimal first experiment: Craftax-1M on the Mac, PPO-RNN with and without the bonus, β in {0.01, 0.03, 0.1}, 5 seeds each; log distinct progress keys and floor entries.
- Success: beat our own PPO-RNN rerun and RND/E3B (2.2% in block D) at 1M with non-overlapping seed spread; on a GPU, beat PPO-RNN's 15.3% at 1B (block A).

### 2. [[Self-Imitation Learning]]

- Most interesting: Craftax's deep achievements are rare one-time events; SIL turns each lucky success into many training examples. No valid SIL result on Craftax exists.
- Feasible: one extra loss and a replay buffer, about 120-180 lines; the main pitfall (buffer memory) is known and avoidable.
- Minimal first experiment: feed-forward PPO plus SIL on Craftax-Classic (fast) and then Craftax-1M, 5 seeds, buffer of 50k steps in `float16`.
- Success: higher reward than PPO-RNN at 1M (above 2.3%), and higher success on rare achievements (`MAKE_IRON_PICKAXE`, `ENTER_DUNGEON`) without losing `DEFEAT_ZOMBIE`.

### 3. [[Tech-Tree Potential-Based Reward Shaping]]

- Most interesting: the crafting chain (wood, table, pickaxes, furnace, iron, diamonds) pays nothing between achievements; a potential computed from the recipe table fills those gaps, and the potential form cannot be farmed in a loop, unlike the hand-made bonuses agents exploited before.
- Feasible: a reward wrapper of about 80-120 lines, PPO unchanged, negligible compute; a textbook idea (policy-invariant shaping).
- Minimal first experiment: Craftax-1M on the Mac, potential scale in {0.1, 0.3, 1.0}, 5 seeds; ablate depth-only versus full tech-tree potential versus none.
- Success: native (unshaped) reward above PPO-RNN (2.3%) at 1M and higher success along `MAKE_STONE_PICKAXE`, `MAKE_IRON_PICKAXE`, `MAKE_DIAMOND_PICKAXE`; the only prior use (maskedPPO-RS) mixed it with masking and extra rewards, so an isolated effect is new.

### 4. [[Explicit Map Memory]]

- Most interesting: the agent sees a 9 x 11 window of a 48 x 48 floor and dark floors blank unlit tiles; a hand-written map gives long spatial memory (water, table, ladder) without a transformer. Not found on Craftax.
- Feasible: a wrapper of about 100-150 lines plus a small CNN of about 30 lines; about 21 MB of map memory for 1024 environments.
- Minimal first experiment: Craftax-1M on the Mac, 5 seeds, crop-only versus pooled-map versus both; run the dead-reckoning (non-privileged) position variant for the fair comparison.
- Success: above PPO-RNN (2.3%) at 1M with the fair variant, fewer deaths from thirst, higher `ENTER_DUNGEON` success; privileged-position results reported separately.

### 5. [[Go-Explore with Achievement Cells]]

- Most interesting: it attacks the floor wall (no 1B agent reliably reaches the Gnomish Mines) and uses Craftax's unique feature: the whole game state is one JAX object you can save and reload.
- Feasible: the change sits in the reset wrapper, about 150-250 lines; it reuses the progress key of the count bonus. The most ambitious pick: do it last.
- Minimal first experiment: PPO-RNN with reset probability $p = 0.25$ to archived cells, 200M steps on one GPU (or Craftax-Classic on the Mac); evaluate on normal resets only.
- Success: non-zero, repeatable `ENTER_GNOMISH_MINES` success on normal resets and reward above PPO-RNN's 15.3% at 1B (block A); stretch goal: above PPO-GTrXL's 18.3%.

## Candidates

| Candidate | One-line idea | Craftax mechanic it exploits | Difficulty | Compute | Novelty status |
|---|---|---|---|---|---|
| [[Count-Based Progress Bonus]] | Bonus $\beta/\sqrt{N}$ over a summary of floor, tools and inventory | Tech tree, inventory, floors | low | Mac (1M); GPU for 1B | Not found on Craftax; only learned bonuses (RND, ICM, E3B, NovelD, C-TeC) tried |
| [[Self-Imitation Learning]] | Replay own better-than-expected actions, per achievement | Rare one-time achievements, long recipes | low | Mac (1M); GPU for 1B | Not published; one autonomous attempt crashed (out of memory) |
| [[Go-Explore with Achievement Cells]] | Restart episodes from archived progress states | Floor wall; JAX state save and load | medium | Mac small tests; GPU for 1B | Partial overlap (Anchor-and-Harvest, SCALAR); one failed attempt |
| [[Tech-Tree Potential-Based Reward Shaping]] | Shaping $\gamma\Phi(s')-\Phi(s)$ from the recipe graph | Crafting dependencies, ladders | low | Mac | Partial: depth-only potential plus hand bonuses and masking (maskedPPO-RS) |
| [[Per-Achievement Value Heads]] | One value head per reward part, summed advantages | Reward = 67 one-time terms + health | low-medium | Mac | Not found |
| [[Explicit Map Memory]] | Hand-written per-floor map as extra input | 9 x 11 view, dark floors, ladders | medium | Mac | Not found |
| [[Population-Based Training]] | Evolve hyperparameter schedules across a vmapped population | Parallel JAX envs; phase-dependent needs | medium | GPU preferred | Not found (ES on Craftax done) |
| [[Bandit-Tuned Exploration]] | UCB picks the bonus weight per episode | Bonus-weight sensitivity | medium | Mac | Not found |

- Mac compute: an issue reporter measured about 2,250 environment steps per second on CPU for Craftax-Classic with 256 environments ([[Craftax performance]]). Derived: 1M steps take minutes plus learner time; 1B steps would take days, so 1B runs need a GPU (about 1 hour on an RTX 4090, [[Craftax Baselines compute and wall-clock]]).
- Common pitfall: `info["Achievements/..."]` is non-zero only on the step an episode ends ([[Craftax Baselines benchmark protocol]]); read `env_state.achievements` for anything that needs unlocks during an episode.

## Suggested by existing work

Future-work ideas from the sources behind [[Existing Implementations]] (Craftax paper arXiv 2402.16801, PQN paper and purejaxql README, transformerXL_PPO_JAX README, Simulus, Efficient MBRL, JaxUED README). The Craftax and Craftax_Baselines READMEs contain no future-work suggestions.

| Algorithm | Suggested by | Where | Student-level feasible | Tried on Craftax since |
|---|---|---|---|---|
| [[Count-Based Progress Bonus]] | [[PQN-RNN (Gallici 2025)]], [[PPO (Matthews 2024)]] | PQN §5.2 (ε-greedy too simple); Craftax §6 | yes | Learned bonuses only; counts not found |
| [[Go-Explore with Achievement Cells]] | [[PPO (Matthews 2024)]] | Craftax abstract and §6 ("deep exploration") | yes (medium) | Partial: Anchor-and-Harvest progress signatures, SCALAR frontier checkpointing |
| [[Population-Based Training]] | [[PQN-RNN (Gallici 2025)]], [[PPO-GTrXL (Hamon 2024)]] | purejaxql README, Craftax section; transformerXL_PPO_JAX README, Results on Craftax | yes (GPU helps) | No PBT found; ES done (EGGROLL) |
| [[PLR with a Recurrent Policy]] | [[PLR (Matthews 2024)]], [[Domain Randomisation (Matthews 2024)]] | JaxUED README ("coming soon"); Craftax §4.5 | yes (medium) | Not found |
| [[In-Context Adaptation Memory]] | [[PPO-RNN (Matthews 2024)]] | Craftax §3.2 (shuffled potions) | yes | Not found |
| [[Language-Conditioned Policy]] | [[PPO (Matthews 2024)]] | Craftax App. D.4 (text renderer) | partly (fixed instruction list only) | Yes: CrafText, LLM instruction relabelling, LEO goal-conditioning, LLM curricula |
| [[Plasticity-Preserving Training]] | [[PPO (Matthews 2024)]], [[PPO-RNN (Matthews 2024)]] | Craftax §6 ("continual learning"); App. F.2 | yes | Yes: Plasticine, Adam-Rel |
| [[Achievement Skill Options]] | [[PPO (Matthews 2024)]] | Craftax §6 ("skill acquisition") | partly (hand-written options) | Yes, with LLMs: SCALAR; Unsupervised Hierarchical Skill Discovery |
| [[Off-Policy Dyna with Imagined Data]] | [[Efficient MBRL (Dedieu 2025)]] | §3.4 and §5 | no (needs a transformer world model; 8 H100 in the paper) | Prioritised world-model replay: Simulus; off-policy mixing: not found |
| [[World-Model Tokenizer Improvements]] | [[Simulus (Cohen 2025)]], [[Efficient MBRL (Dedieu 2025)]] | Simulus §5; Efficient MBRL §5 | no (full world-model agent; pretrained-model variant uses a foundation model) | Simulus-A tokenizer, no Craftax results |

- Craftax §6 also names "generalisation" and "long term reasoning"; they map to [[PLR with a Recurrent Policy]] and to the memory notes ([[Explicit Map Memory]], [[In-Context Adaptation Memory]]).

## Suggested order

1. Reproduce PPO-RNN on Craftax-1M (Mac) and Craftax-Classic; this is the control for everything below.
2. [[Count-Based Progress Bonus]]: cheapest; its progress key is reused by steps 4, 6 and 8.
3. [[Self-Imitation Learning]]: independent of the bonus; can be combined with it later.
4. [[Tech-Tree Potential-Based Reward Shaping]]: shares the recipe table with the progress key.
5. [[Per-Achievement Value Heads]]: a critic change, orthogonal to 2-4.
6. [[Go-Explore with Achievement Cells]]: first 1B-scale GPU experiment.
7. [[Explicit Map Memory]] or [[In-Context Adaptation Memory]]: memory track.
8. [[Bandit-Tuned Exploration]]: needs a working bonus from step 2.
9. [[Population-Based Training]]: only with steady GPU access.

## Rejected ideas

- Action masking of impossible actions: done on Craftax and Craftax-Classic ("Overcoming Valid Action Suppression", [arXiv 2603.09090](https://arxiv.org/abs/2603.09090)); also in maskedPPO-GTrXL ([repository](https://github.com/Doguhan58/craftax-maskedPPO-reward-shaping), [Craftax PR #52](https://github.com/MichaelTMatthews/Craftax/pull/52)).
- Achievement-conditioned policies with hindsight relabelling: done by LEO on goal-conditioned Craftax with a HER baseline ([arXiv 2605.23551](https://arxiv.org/abs/2605.23551)) and by LLM instruction relabelling ([arXiv 2506.20061](https://arxiv.org/abs/2506.20061)).
- Curriculum over start states or floors: done by Anchor-and-Harvest Curriculum ([arXiv 2610.15328](https://arxiv.org/abs/2610.15328)), DiCode (ICML 2026, [code](https://github.com/konstantinosmitsides/dreaming-in-code)), SCALAR frontier checkpointing ([arXiv 2603.09036](https://arxiv.org/abs/2603.09036)) and PLR/ACCEL in the Craftax paper ([[PLR (Matthews 2024)]], [[ACCEL (Matthews 2024)]]).
- Evolution strategies: done (OpenES and EGGROLL on Craftax-Classic and Craftax-Symbolic, [arXiv 2511.16652](https://arxiv.org/abs/2511.16652)).
- Options and macro-actions as our own pick: done with LLM-defined skills (SCALAR); kept only as the suggested-by note [[Achievement Skill Options]].
- Return-conditioned supervised learning (upside-down RL, Decision-Transformer style): overlaps with SIL and LEO, and is known to fail in stochastic environments ([arXiv 2205.15967](https://arxiv.org/abs/2205.15967)); Craftax is stochastic (mobs, loot, potions).
- Simple episodic memory or episodic novelty: E3B already covers episodic novelty ([[E3B (Matthews 2024)]]); replaced by [[Explicit Map Memory]].
- Next-achievement prediction as an auxiliary loss: done on Crafter, which has Craftax-Classic's rules (Achievement Distillation, [arXiv 2307.03486](https://arxiv.org/abs/2307.03486)).
- Decision-time rollouts or MCTS with the real simulator: uses the full hidden state and spends environment steps on planning, so scores are not comparable with the 1B and 1M challenges.
- More learned curiosity (RND, ICM, E3B, NovelD, temporal-contrastive C-TeC): done ([[RND (Matthews 2024)]], [[ICM (Matthews 2024)]], [[E3B (Matthews 2024)]], [arXiv 2603.02008](https://arxiv.org/abs/2603.02008)).
- Exploration from retrying (ReMax PPO): done on Craftax ([arXiv 2606.00151](https://arxiv.org/abs/2606.00151)).
- Scaling the number of parallel environments: done ([arXiv 2603.06009](https://arxiv.org/abs/2603.06009)).
- Empowerment maximisation: shown on Crafter ([arXiv 2610.13002](https://arxiv.org/abs/2610.13002)); heavier theory than the project targets.

## How novelty was checked

- Semantic Scholar: all 105 indexed papers citing the Craftax paper (arXiv 2402.16801), titles and abstracts, read 2026-10-20; the index may miss recent papers.
- Full text of the closest papers: Anchor-and-Harvest Curriculum, LEO, SCALAR, ReMax PPO, Adam-Rel, EGGROLL.
- Web searches per candidate ("Craftax" or "Crafter" plus the method name).
- GitHub: all repositories matching "craftax" (about 70), including the auto-craftax autonomous research report ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md)), which tried Go-Explore, PBRS and SIL without valid results.
- Craftax pull requests #52 (maskedPPO with reward shaping) and #54 (PPO-LSTM).
