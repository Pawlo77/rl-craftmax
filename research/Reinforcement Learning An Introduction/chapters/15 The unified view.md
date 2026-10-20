---
title: "Prospects: the unified view"
parent: "[[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]"
type: book-chapter
chapter: 15
section_range: "15.1"
authors:
  - Richard S. Sutton
  - Andrew G. Barto
venue: "Book, Reinforcement Learning: An Introduction, 2nd edition (draft), Chapter 15"
pdf: "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf"
tags:
  - book
  - reinforcement-learning
  - chapter
created: 2026-10-14
---
# The unified view (Sutton & Barto, Chapter 15)

> [!abstract] TL;DR
> Every reinforcement learning (RL) method in the book estimates value functions, by backing up values along actual or possible state trajectories, inside the loop of generalized policy iteration (GPI); methods differ along a set of dimensions. Two dimensions, the width and the depth of a backup, span a plane in which dynamic programming (DP), temporal-difference (TD) learning, Monte Carlo (MC) methods and exhaustive search are the corners, and a third, function approximation, runs perpendicular to it.

## Problem

- **What the closing view answers**: the book presents RL as a coherent set of ideas cutting across methods, not as a list of methods; each idea is a dimension along which methods vary, and together the dimensions span a large space of possible methods.
- **Why work at the level of dimensions**: it gives the broadest and most lasting understanding. The analysis of on-policy versus off-policy learning together with bootstrapping and function approximation involved three dimensions, and many other dimensions were found to be irrelevant to it, which simplified the analysis and made its conclusion more significant (see [[09b Control with approximation, bootstrapping]]).

## Conceptual framework

### What every method in the book shares

These three ideas circumscribe the subject of the book, and are proposed as organizing principles potentially relevant to any model of intelligence.

- **Value-function estimation**: the objective of all the methods is to estimate value functions (defined in [[03b MDPs, value functions, optimality]]).
- **Backing up values**: all operate by backing up values along actual or possible state trajectories. A backup updates the value estimate of a state (or state-action pair) from the values of what follows it.
- **Generalized policy iteration (GPI)**: all maintain an approximate value function and an approximate policy, and keep improving each on the basis of the other. [[04 Dynamic programming]] introduces it.

### Terms used by the dimensions

- **Sample versus full backup**: a sample backup uses one sample trajectory; a full backup uses the whole distribution of possible trajectories. Full backups need a model of the environment (something that predicts the next state and reward from a state and action, see [[01 Introduction]]); sample backups work with or without one, so with-model versus without-model is itself a dimension of variation ([[08a Models, planning, Dyna]] compares the two).
- **Depth of a backup**: how far a backup reaches before it stops and substitutes an estimate; this is the degree of bootstrapping, that is, of updating an estimate from other estimates.

## Approach

The dimensions come in two groups: the four the section treats as the most important, then a list of further ones.

### The four main dimensions

1. **Width of backup (sample versus full)**: the horizontal axis of the plane in the next subsection.
2. **Depth of backup (degree of bootstrapping)**: the vertical axis of that plane.
3. **Function approximation**: a spectrum orthogonal to the plane, visualized as perpendicular to the page; its range is listed in the subsection on function approximation.
4. **On-policy versus off-policy**: a binary distinction.
   - An on-policy method learns the value function of the policy it is currently following.
   - An off-policy method learns the value function of the policy it currently thinks is best.
   - The two policies often differ because the agent must explore.

### The plane of width and depth

The book's figure is a square with width of backup along the horizontal axis and depth along the vertical one. Three of its four corners are the three primary ways of estimating values:

| Corner | Method | Backup it uses | Developed in |
|--------|--------|----------------|--------------|
| upper left | TD learning | one-step sample backup | [[06 Temporal-difference learning]] |
| lower left | Monte Carlo | sample backup of the full return, down to a terminal state | [[05a Monte Carlo prediction and control]], [[05b Off-policy Monte Carlo]] |
| upper right | DP | one-step full backup | [[04 Dynamic programming]] |
| lower right | exhaustive search | full backups so deep they reach terminal states (in a continuing task, until discounting makes further rewards negligible) | the limiting case of the limited-depth search described below, [[08b Trajectory sampling, search, MCTS]] |

- **Left edge (all sample backups)**: ranges from one-step TD to full-return MC, with a spectrum between them.
  - That spectrum holds $n$-step backups and mixtures of them, such as the $\lambda$-backups implemented by eligibility traces ([[07a n-step TD and TD(λ)]], [[07b Eligibility-trace control and off-policy]]).
- **Intermediate depth on the full-backup side**: heuristic search and related methods, which search and back up to a limited depth, perhaps selectively ([[08b Trajectory sampling, search, MCTS]]).
- **Intermediate width**: methods that mix full and sample backups, and the possibility of methods that mix samples and distributions within one backup.
- **Interior**: filled in to represent all such intermediate methods.

### Function approximation as a third axis

- **The spectrum**: tabular methods at one extreme, then state aggregation, a variety of linear methods, and finally a diverse set of nonlinear methods.
- **Where developed**: [[09a Function approximation and linear methods]] and [[09b Control with approximation, bootstrapping]].

### Further dimensions identified in the book

| Dimension | The question it asks | Related notes (this vault's links; the section names no chapter) |
|-----------|----------------------|--------------|
| Definition of return | Is the task episodic or continuing, discounted or undiscounted? | [[03a MDP setting and returns]] |
| Action values vs. state values vs. afterstate values | What kind of values to estimate? Estimating only state values requires either a model or a separate policy, as in actor-critic methods, for action selection | [[03b MDPs, value functions, optimality]], [[06 Temporal-difference learning]] (afterstates), [[11 Policy approximation (actor-critic)]] |
| Action selection and exploration | How to trade off exploration and exploitation? Only the simplest ways were considered: $\varepsilon$-greedy, softmax action selection and optimistic initialization of values | [[02 Multi-armed bandits]] |
| Synchronous vs. asynchronous | Are the backups for all states done simultaneously, or one by one in some order? | [[04 Dynamic programming]] |
| Replacing vs. accumulating traces | If eligibility traces are used, which kind is most appropriate? | [[07a n-step TD and TD(λ)]] |
| Real vs. simulated | Back up real experience or simulated experience, and if both, how much of each? | [[08a Models, planning, Dyna]] |
| Location of backups | Which states or state-action pairs to back up? Model-free methods can choose only among those actually encountered; model-based methods can choose arbitrarily, which leaves many potent possibilities | [[08a Models, planning, Dyna]], [[08b Trajectory sampling, search, MCTS]] |
| Timing of backups | Should backups be done as part of selecting actions, or only afterward? | [[08b Trajectory sampling, search, MCTS]] |
| Memory for backups | How long to retain backed-up values: permanently, or only while computing an action selection, as in heuristic search? | [[08b Trajectory sampling, search, MCTS]] |

### How the dimensions combine

- **Neither exhaustive nor mutually exclusive**: the listed dimensions are not a complete taxonomy; individual algorithms also differ in many other ways, and many lie at several places along several dimensions.
- **Dyna as an example of mixing**: Dyna methods use real and simulated experience to affect the same value function ([[08a Models, planning, Dyna]]).
- **Several value functions**: it is also sensible to maintain multiple value functions computed in different ways or over different state and action representations.
- **Overall claim**: the dimensions are a coherent set of ideas for describing and exploring a wide space of possible methods.
