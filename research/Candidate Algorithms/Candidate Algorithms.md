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
updated: 2026-10-20
---
# Candidate Algorithms

Algorithms the project could try on Craftax beyond the published agents in [[Existing Implementations]], built on textbook ideas from [[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]. Two groups: PPO add-ons (a bonus, loss, reward wrapper, input or reset strategy on the PPO or PPO-RNN baseline, [[Craftax Baselines]]) and different algorithm families (a learner that is not PPO). Novelty was checked on 2026-10-20 (method at the end).

## Proposed final method set

Four methods: the two best different-family algorithms and the two best PPO add-ons. Controls are our own reruns of PPO-RNN and PQN-RNN at Craftax-1M and Craftax-1B.

| # | Method | Group | Why |
|---|---|---|---|
| 1 | [[Count-Based Progress Bonus]] on PPO-RNN and on PQN-RNN | PPO add-on (also run on the value-based learner) | Simplest Top 5 pick; its progress key is reused by methods 3 and 4; the same bonus on both learners tests whether exploration help depends on the learner |
| 2 | [[Self-Imitation Learning]] on PPO-RNN | PPO add-on | Turns rare one-time achievements into many training examples; independent of method 1 |
| 3 | [[SMDP Q-Learning over Hand-Defined Skills]] | Hierarchical RL, value-based | Book-level answer to the long crafting chain: about 10 skill decisions instead of thousands of key presses, learned by a plain Q-table |
| 4 | [[Dyna-Q over a Skill-Level Progress Model]] | Model-based planning | Adds the book's Dyna-Q to method 3 with about 150 lines; a lookup-table model of skill outcomes avoids the screen-prediction problems of the world-model agents |

- Value-based against policy gradient: PPO-RNN against PQN-RNN, each with and without the method-1 bonus.
- Model-free against planning: method 3 against method 4, same skills, same seeds (planning updates $n = 0$ against $n > 0$).
- Flat against hierarchical: PQN-RNN against method 3, whose skills are trained by the same PQN-RNN learner.
- Base against add-on: PPO-RNN against PPO-RNN plus method 1 and PPO-RNN plus method 2.
- Prior knowledge: methods 3 and 4 use hand-written skills, so they are also reported apart from block A of [[Existing Implementations]], as maskedPPO is.
- Reserve: [[Actor-Critic with Eligibility Traces]] if the team wants a fifth, knowledge-free algorithm comparison (batch against online updates).

## Top 5 PPO add-ons

For a one-semester student project: a small team building on PPO-RNN. Picks are ranked by promise and by conceptual and implementation simplicity; each is evaluated at both Craftax-1M and Craftax-1B.

- Why 5: each pick attacks a different Craftax difficulty (exploration, rare successes, sparse crafting chain, partial observability, the floor wall), so together they cover every truly promising direction without overlap. Judged on conceptual and implementation complexity alone, no left-out candidate is clearly more promising than these five.
- Left out: [[Per-Achievement Value Heads]] (simple and novel, but a smaller critic tweak; next in line), [[Bandit-Tuned Exploration]] and [[Population-Based Training]] (tune other methods rather than add an idea), [[Off-Policy Dyna with Imagined Data]] and [[World-Model Tokenizer Improvements]] (partly student-level, but each needs a full world-model agent with research-grade debugging and targets only Craftax-1M sample efficiency), and the suggested-by notes already taken by others.

| # | Candidate | Craftax difficulty it attacks | Lines of code | First experiment | Success means |
|---|---|---|---|---|---|
| 1 | [[Count-Based Progress Bonus]] | Exploration bonuses that chase new scenery | 60-100 | Craftax-1M and Craftax-1B, 5 seeds | Above RND/E3B (2.2%) at 1M; above PPO-RNN (15.3%) at 1B |
| 2 | [[Self-Imitation Learning]] | Rare one-time successes | 120-180 | Craftax-Classic, then Craftax-1M and Craftax-1B, 5 seeds | Above PPO-RNN at 1M (2.3%) and 1B (15.3%) |
| 3 | [[Tech-Tree Potential-Based Reward Shaping]] | Long reward-free crafting chain | 80-120 | Craftax-1M and Craftax-1B, 5 seeds | Native reward above PPO-RNN at 1M (2.3%) and 1B (15.3%) |
| 4 | [[Explicit Map Memory]] | Seeing 4% of the floor; dark floors | 130-180 | Craftax-1M and Craftax-1B, 5 seeds | Above PPO-RNN at 1M (2.3%) and 1B (15.3%), fair-position variant |
| 5 | [[Go-Explore with Achievement Cells]] | No agent passes the floor-2 wall | 150-250 | Craftax-1B, 5 seeds, $p \in \{0, 0.25, 0.5\}$ | Repeatable `ENTER_GNOMISH_MINES`; above 15.3% at 1B |

### 1. [[Count-Based Progress Bonus]]

- Most interesting: it tests the Craftax paper's own explanation of why exploration bonuses failed (the bonus looks at scenery). A bonus over game progress only is a direct, clean test of that claim.
- Feasible: no new network, about 60-100 lines; a textbook idea (UCB-style bonus, [[02 Multi-armed bandits]]).
- Minimal first experiment: Craftax-1M and Craftax-1B, PPO-RNN with and without the bonus, β in {0.01, 0.03, 0.1}, 5 seeds each; log distinct progress keys and floor entries.
- Success: beat our own PPO-RNN rerun and RND/E3B (2.2% in block D) at 1M with non-overlapping seed spread, and beat PPO-RNN's 15.3% at 1B (block A).

### 2. [[Self-Imitation Learning]]

- Most interesting: Craftax's deep achievements are rare one-time events; SIL turns each lucky success into many training examples. No valid SIL result on Craftax exists.
- Feasible: one extra loss and a replay buffer, about 120-180 lines; the main pitfall (buffer memory) is known and avoidable.
- Minimal first experiment: feed-forward PPO plus SIL on Craftax-Classic, then PPO-RNN plus SIL on Craftax-1M and Craftax-1B, 5 seeds, buffer of 50k steps in `float16`; ablate per-achievement slots against one shared buffer.
- Success: higher reward than PPO-RNN at 1M (above 2.3%) and 1B (above 15.3%), and higher success on rare achievements (`MAKE_IRON_PICKAXE`, `ENTER_DUNGEON`) without losing `DEFEAT_ZOMBIE`.

### 3. [[Tech-Tree Potential-Based Reward Shaping]]

- Most interesting: the crafting chain (wood, table, pickaxes, furnace, iron, diamonds) pays nothing between achievements; a potential computed from the recipe table fills those gaps, and the potential form cannot be farmed in a loop, unlike the hand-made bonuses agents exploited before.
- Feasible: a reward wrapper of about 80-120 lines, PPO unchanged; a textbook idea (policy-invariant shaping).
- Minimal first experiment: Craftax-1M and Craftax-1B, potential scale in {0.1, 0.3, 1.0}, 5 seeds; ablate depth-only versus full tech-tree potential versus none.
- Success: native (unshaped) reward above PPO-RNN at 1M (2.3%) and 1B (15.3%), and higher success along `MAKE_STONE_PICKAXE`, `MAKE_IRON_PICKAXE`, `MAKE_DIAMOND_PICKAXE`; the only prior use (maskedPPO-RS) mixed it with masking and extra rewards, so an isolated effect is new.

### 4. [[Explicit Map Memory]]

- Most interesting: the agent sees a 9 x 11 window of a 48 x 48 floor and dark floors blank unlit tiles; a hand-written map gives long spatial memory (water, table, ladder) without a transformer. Not found on Craftax.
- Feasible: a wrapper of about 100-150 lines plus a small CNN of about 30 lines; about 21 MB of map memory for 1024 environments.
- Minimal first experiment: Craftax-1M and Craftax-1B, 5 seeds, crop-only versus pooled-map versus both; run the dead-reckoning (non-privileged) position variant for the fair comparison.
- Success: above PPO-RNN at 1M (2.3%) and 1B (15.3%) with the fair variant, fewer deaths from thirst, higher `ENTER_DUNGEON` success; privileged-position results reported separately.

### 5. [[Go-Explore with Achievement Cells]]

- Most interesting: it attacks the floor wall (no 1B agent reliably reaches the Gnomish Mines) and uses Craftax's unique feature: the whole game state is one JAX object you can save and reload.
- Feasible: the change sits in the reset wrapper, about 150-250 lines; it reuses the progress key of the count bonus. The most ambitious pick: do it last.
- Minimal first experiment: PPO-RNN with reset probability $p \in \{0.25, 0.5\}$ to archived cells against $p = 0$, Craftax-1B, 5 seeds, after debugging the archive on Craftax-Classic; evaluate on normal resets only, and also report Craftax-1M.
- Success: non-zero, repeatable `ENTER_GNOMISH_MINES` success on normal resets and reward above PPO-RNN's 15.3% at 1B (block A); stretch goal: above PPO-GTrXL's 18.3%.

## PPO add-ons

All rows below keep PPO (or PPO-RNN) as the learner and change one part around it.

| Candidate | One-line idea | Craftax mechanic it exploits | Difficulty | Novelty status |
|---|---|---|---|---|
| [[Count-Based Progress Bonus]] | Bonus $\beta/\sqrt{N}$ over a summary of floor, tools and inventory | Tech tree, inventory, floors | low | Not found on Craftax; only learned bonuses (RND, ICM, E3B, NovelD, C-TeC) tried |
| [[Self-Imitation Learning]] | Replay own better-than-expected actions, per achievement | Rare one-time achievements, long recipes | low | Not published; one autonomous attempt crashed (out of memory) |
| [[Go-Explore with Achievement Cells]] | Restart episodes from archived progress states | Floor wall; JAX state save and load | medium | Partial overlap (Anchor-and-Harvest, SCALAR); one failed attempt |
| [[Tech-Tree Potential-Based Reward Shaping]] | Shaping $\gamma\Phi(s')-\Phi(s)$ from the recipe graph | Crafting dependencies, ladders | low | Partial: depth-only potential plus hand bonuses and masking (maskedPPO-RS) |
| [[Per-Achievement Value Heads]] | One value head per reward part, summed advantages | Reward = 67 one-time terms + health | low-medium | Not found |
| [[Explicit Map Memory]] | Hand-written per-floor map as extra input | 9 x 11 view, dark floors, ladders | medium | Not found |
| [[Population-Based Training]] | Evolve hyperparameter schedules across a vmapped population | Parallel JAX envs; phase-dependent needs | medium | Not found (ES on Craftax done) |
| [[Bandit-Tuned Exploration]] | UCB picks the bonus weight per episode | Bonus-weight sensitivity | medium | Not found |

- Common pitfall: `info["Achievements/..."]` is non-zero only on the step an episode ends ([[Craftax Baselines benchmark protocol]]); read `env_state.achievements` for anything that needs unlocks during an episode.

## Different algorithm families

Second pass (2026-10-20): whole algorithm families whose learner is not PPO, at the level of the textbook. Every family evaluated has a note; families already covered by a note in this folder or in [[Existing Implementations]] link that note instead. Difficulty is conceptual and implementation complexity only.

| Candidate | Family | Book chapter | Craftax mechanic exploited | Implementation difficulty | Novelty status |
|---|---|---|---|---|---|
| [[SMDP Q-Learning over Hand-Defined Skills]] | Hierarchical RL: SMDP Q-learning table over progress keys, Q(λ) skills | [[14.4 Elevator dispatching]], [[06 Temporal-difference learning]], [[07b Eligibility-trace control and off-policy]] | Tech-tree chains, survival stats, floors; skill end tests from the JAX state | medium | Candidate. Partial overlap: SCALAR (LLM skills, STRIPS planner, PPO), SEA on Crafter (rule-based controller); a learned tabular SMDP top level not found |
| [[Dyna-Q over a Skill-Level Progress Model]] | Model-based planning: Dyna-Q, Dyna-Q+, prioritized sweeping | [[08a Models, planning, Dyna]], [[04 Dynamic programming]] | Recipe graph learned as a lookup-table model; long chains | medium (on top of the SMDP note) | Candidate. Partial overlap: SEA graph (Crafter), Rule Learner Planner (reward-free, Classic), AHC hand-written graph; Dyna-Q on a learned skill-outcome model not found |
| [[Actor-Critic with Eligibility Traces]] | Online actor-critic with traces (and its Q(λ) twin) | [[11 Policy approximation (actor-critic)]], [[07a n-step TD and TD(λ)]] | 100,000-step episodes, dense health reward, parallel envs to average per-step updates | medium | Candidate. Not found on Craftax or Crafter |
| [[Monte Carlo Control]] | Monte Carlo control | [[05a Monte Carlo prediction and control]] | - | low | Rejected: the λ = 1 setting of PQN and GAE; huge variance over 100,000-step episodes |
| [[Sarsa-Lambda with Function Approximation]] | On-policy TD control | [[07b Eligibility-trace control and off-policy]], [[09b Control with approximation, bootstrapping]] | - | low | Rejected: the family is done on Craftax by PQN-RNN (Q(λ)); a one-line target ablation |
| [[Synchronous Advantage Actor-Critic (A2C)]] | Batch actor-critic | [[11 Policy approximation (actor-critic)]] | - | low | Rejected: a special case of PPO (one epoch, no clip); not found on Craftax |
| [[REINFORCE with Baseline]] | Monte Carlo policy gradient | [[11 Policy approximation (actor-critic)]], [[05a Monte Carlo prediction and control]] | - | low | Rejected: variance over long episodes; used inside Simulus's imagination |
| [[V-trace Off-Policy Actor-Critic]] | Off-policy actor-critic (IMPALA) | [[05b Off-policy Monte Carlo]], [[07b Eligibility-trace control and off-policy]] | - | medium | Rejected: no actor lag in the JAX pipeline, so it equals A2C; IMPALA used on Crafter by SEA |
| [[Advantage-Weighted Regression]] | Advantage-weighted policy improvement | not in the book | - | medium | Rejected: not textbook; overlaps [[Self-Imitation Learning]] |
| [[Average-Reward Differential Q-Learning]] | Average-reward control (R-learning) | [[11 Policy approximation (actor-critic)]] | - | low | Rejected: chaining episodes makes dying and re-earning easy achievements profitable |
| [[Distributional Q-Learning]] | Distributional values (C51, QR-DQN) | not in the book | - | medium | Rejected: not textbook; distributional ensembles on Crafter (Jiang et al. 2023), HL-Gauss values in Simulus |
| [[Successor Features and GPI over Achievements]] | Successor features, generalized policy improvement | [[03b MDPs, value functions, optimality]] | - | high | Rejected: not textbook; reduces to [[Per-Achievement Value Heads]] plus skills; all-goals learning done by Multitask Preplay and LEO |
| [[Linear Sarsa-Lambda with Tile Coding]] | Linear function approximation | [[09a Function approximation and linear methods]], [[09b Control with approximation, bootstrapping]] | - | low | Rejected: a linear function of the view cannot navigate; the table part is used in the SMDP note |
| [[Least-Squares Policy Iteration]] | Least-squares TD (LSTD, LSPI) | [[09a Function approximation and linear methods]], [[04 Dynamic programming]] | - | medium | Rejected: same representation limit; on progress keys it equals the value-iteration variant of the Dyna note |
| [[MCTS with a Learned Model]] | Decision-time planning (MuZero style) | [[08b Trajectory sampling, search, MCTS]] | - | high | Rejected: beyond student level; not found on Craftax; at the skill level value iteration is exact |
| [[Option-Critic]] | Learned options | not in the book | - | high | Rejected: not textbook, option collapse; a hobby project on Craftax without numbers |
| [[PQN-RNN (Gallici 2025)]] | Q(λ) with a network over parallel envs | [[07b Eligibility-trace control and off-policy]] | - | - | Covered: done on Craftax (16.0% at 1B); used as a learner in the final set |
| [[DLSD (Rajaraman 2025)]], [[PQN-RNN (Gallici 2025)]] | DQN with a replay buffer | [[06 Temporal-difference learning]] | - | - | Covered: DQN on Craftax-Classic (DLSD); replay-buffer PQN on Craftax (same score, slower) |
| [[Off-Policy Dyna with Imagined Data]], [[Efficient MBRL (Dedieu 2025)]], [[Simulus (Cohen 2025)]], [[DreamerV3 (Hafner 2023)]] | Dyna with a neural world model | [[08a Models, planning, Dyna]] | - | - | Covered: done on Craftax and Craftax-Classic |
| [[Achievement Skill Options]], [[SCALAR (Zabounidis 2026)]] | Options with a PPO top level | [[03a MDP setting and returns]] | - | - | Covered: done with LLM skills (SCALAR) |
| [[Rule Learner Planner (dargilboa 2026)]] | Learned rules plus backward-chaining planner | [[08b Trajectory sampling, search, MCTS]] | - | - | Covered: reward-free, Craftax-Classic |
| [[RePPO (Nishimori 2026)]] | Policy gradient for a retry (max-return) objective | [[11 Policy approximation (actor-critic)]] | - | - | Covered: done on Craftax-1B |
| [[DLSD (Rajaraman 2025)]] | Unsupervised skill discovery (DIAYN) | not in the book | - | - | Covered: done on Craftax-Classic |
| [[Sable Inference Search (Chalumeau 2025)]] | Decision-time search with the real simulator | [[08b Trajectory sampling, search, MCTS]] | - | - | Covered: done; not comparable (see Rejected ideas) |
| Evolution strategies, goal-conditioned RL with hindsight relabelling | ES (OpenES, EGGROLL); HER (LEO) | [[01 Introduction]] (evolutionary search) | - | - | Covered: done on Craftax; see Rejected ideas |

- Each rejected family is done on Craftax, beyond textbook level, or a special case of a method already in the set.
- The progress key of [[Count-Based Progress Bonus]] is the shared state abstraction of the SMDP and Dyna candidates.

## Suggested by existing work

Future-work ideas from the sources behind [[Existing Implementations]] (Craftax paper arXiv 2402.16801, PQN paper and purejaxql README, transformerXL_PPO_JAX README, Simulus, Efficient MBRL, JaxUED README). The Craftax and Craftax_Baselines READMEs contain no future-work suggestions.

| Algorithm | Suggested by | Where | Student-level feasible | Tried on Craftax since |
|---|---|---|---|---|
| [[Count-Based Progress Bonus]] | [[PQN-RNN (Gallici 2025)]], [[PPO (Matthews 2024)]] | PQN §5.2 (ε-greedy too simple); Craftax §6 | yes | Learned bonuses only; counts not found |
| [[Go-Explore with Achievement Cells]] | [[PPO (Matthews 2024)]] | Craftax abstract and §6 ("deep exploration") | yes (medium) | Partial: Anchor-and-Harvest progress signatures, SCALAR frontier checkpointing |
| [[Population-Based Training]] | [[PQN-RNN (Gallici 2025)]], [[PPO-GTrXL (Hamon 2024)]] | purejaxql README, Craftax section; transformerXL_PPO_JAX README, Results on Craftax | yes | No PBT found; ES done (EGGROLL) |
| [[PLR with a Recurrent Policy]] | [[PLR (Matthews 2024)]], [[Domain Randomisation (Matthews 2024)]] | JaxUED README ("coming soon"); Craftax §4.5 | yes (medium) | Not found |
| [[In-Context Adaptation Memory]] | [[PPO-RNN (Matthews 2024)]] | Craftax §3.2 (shuffled potions) | yes | Not found |
| [[Language-Conditioned Policy]] | [[PPO (Matthews 2024)]] | Craftax App. D.4 (text renderer) | partly (fixed instruction list only) | Yes: CrafText, LLM instruction relabelling, LEO goal-conditioning, LLM curricula |
| [[Plasticity-Preserving Training]] | [[PPO (Matthews 2024)]], [[PPO-RNN (Matthews 2024)]] | Craftax §6 ("continual learning"); App. F.2 | yes | Yes: Plasticine, Adam-Rel |
| [[Achievement Skill Options]] | [[PPO (Matthews 2024)]] | Craftax §6 ("skill acquisition") | partly (hand-written options) | Yes, with LLMs: SCALAR; Unsupervised Hierarchical Skill Discovery |
| [[Off-Policy Dyna with Imagined Data]] | [[Efficient MBRL (Dedieu 2025)]] | §3.4 and §5 | partly (small MLP world model with PQN; a faithful transformer world model is research-grade, code not released) | Prioritised world-model replay: Simulus; off-policy mixing: not found |
| [[World-Model Tokenizer Improvements]] | [[Simulus (Cohen 2025)]], [[Efficient MBRL (Dedieu 2025)]] | Simulus §5; Efficient MBRL §5 | partly (integer tokenizer inside the open Simulus code, a large system to learn; pretrained-model variant excluded as a foundation model) | Simulus-A tokenizer, no Craftax results |

- Student-level feasibility is judged on conceptual and implementation complexity only.
- Craftax §6 also names "generalisation" and "long term reasoning"; they map to [[PLR with a Recurrent Policy]] and to the memory notes ([[Explicit Map Memory]], [[In-Context Adaptation Memory]]).

## Suggested order

1. Reproduce PPO-RNN on Craftax-1M, Craftax-1B and Craftax-Classic; this is the control for everything below.
2. [[Count-Based Progress Bonus]]: simplest; its progress key is reused by steps 4, 6 and 8.
3. [[Self-Imitation Learning]]: independent of the bonus; can be combined with it later.
4. [[Tech-Tree Potential-Based Reward Shaping]]: shares the recipe table with the progress key.
5. [[Per-Achievement Value Heads]]: a critic change, orthogonal to 2-4.
6. [[Go-Explore with Achievement Cells]]: the most ambitious change; reuses the progress key from step 2.
7. [[Explicit Map Memory]] or [[In-Context Adaptation Memory]]: memory track.
8. [[Bandit-Tuned Exploration]]: needs a working bonus from step 2.
9. [[Population-Based Training]]: last, to tune whichever of steps 2-8 works best.

Different-family track, in parallel with steps 2-9:

1. Reproduce PQN-RNN (purejaxql) next to PPO-RNN; it is the value-based control and the skill learner of step 2.
2. [[SMDP Q-Learning over Hand-Defined Skills]] on Craftax-Classic, then Craftax; reuses the progress key of [[Count-Based Progress Bonus]].
3. [[Dyna-Q over a Skill-Level Progress Model]]: a switch on top of the SMDP agent.
4. [[Actor-Critic with Eligibility Traces]]: independent; reserve.

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

Second pass (algorithm families, 2026-10-20):

- Semantic Scholar: the 106 papers citing arXiv 2402.16801, titles and abstracts filtered for option, hierarchical, SMDP, Dyna, planning, MCTS, successor, eligibility, trace, Sarsa, Q-learning, actor-critic, A2C, IMPALA, V-trace, average reward, distributional, least-squares, linear, tabular, abstraction, streaming and skill terms.
- arXiv API: `all:craftax` (30 papers), `crafter AND (hierarchical OR options OR planning OR tree)`, `crafter AND (successor OR distributional OR eligibility OR sarsa OR q-learning)`.
- Full text (arXiv HTML) of the closest papers: SEA (2305.00508), H²RL (2603.06565), Streaming Deep RL (2410.14606), Scalable Option Learning (2509.00338).
- Web searches per family ("Craftax" or "Crafter" plus the method name) and GitHub repository search for "craftax" plus each family name (no hits); GitHub code search stopped by the API rate limit.

First pass (PPO add-ons):

- Semantic Scholar: all 105 indexed papers citing the Craftax paper (arXiv 2402.16801), titles and abstracts, read 2026-10-20; the index may miss recent papers.
- Full text of the closest papers: Anchor-and-Harvest Curriculum, LEO, SCALAR, ReMax PPO, Adam-Rel, EGGROLL.
- Web searches per candidate ("Craftax" or "Crafter" plus the method name).
- GitHub: all repositories matching "craftax" (about 70), including the auto-craftax autonomous research report ([report](https://github.com/roger-creus/auto-craftax/blob/main/report/report.md)), which tried Go-Explore, PBRS and SIL without valid results.
- Craftax pull requests #52 (maskedPPO with reward shaping) and #54 (PPO-LSTM).
