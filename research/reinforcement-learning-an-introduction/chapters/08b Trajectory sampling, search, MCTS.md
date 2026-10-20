---
title: "Planning and Learning with Tabular Methods: trajectory sampling, heuristic search, Monte Carlo tree search"
parent: "[[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]"
type: book-chapter
chapter: 8
section_range: "8.6-8.9"
authors:
  - Richard S. Sutton
  - Andrew G. Barto
venue: "Book, Reinforcement Learning: An Introduction, 2nd edition (draft), Chapter 8"
pdf: "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf"
tags:
  - book
  - reinforcement-learning
  - chapter
created: 2026-10-14
---
# Trajectory sampling, heuristic search, MCTS (Sutton & Barto, Chapter 8 part b)

> [!abstract] TL;DR
> Where planning spends its backups matters as much as how many it does: trajectory sampling backs up the states a simulated run of the current policy visits, and heuristic search concentrates backups on the tree of futures below the current state. Following the on-policy distribution plans faster early but slower later, and deep search wins not through multistep backups but through focus on what lies just downstream. Models, planning, Dyna and full versus sample backups are in [[08a Models, planning, Dyna]].

## Problem

- **Where to put backups**: with a model, backups can be applied to any state or state-action pair in any order, and the choice decides how soon the value function becomes useful. A **backup** is defined in [[08a Models, planning, Dyna]]: an update of a value estimate toward a value computed from successor estimates.
- **Exhaustive sweeps**: the classical dynamic-programming way is to sweep the whole state (or state-action) space, backing up each item once per sweep (see [[04 Dynamic programming]]).
  - On large tasks there may be no time to finish even one sweep.
  - Most states are irrelevant, visited only under very poor policies or with very low probability, yet a sweep gives every part of the space equal time.
- **Sweeps are not required**: backups can be distributed any way one likes; to assure convergence every state or state-action pair must be visited infinitely often in the limit.
- **Uniform sampling is no better**: picking pairs uniformly at random, as the Dyna-Q agent of [[08a Models, planning, Dyna]] does, shares the problem of sweeps.
- **Acting needs a decision now**: at each real step the agent must choose an action, and the backups that matter most are those that sharpen that decision.

## Examples in the chapter

| Example | Setting / decision | What it demonstrates |
|---------|--------------------|----------------------|
| Random tasks, uniform versus on-policy backups | undiscounted episodic tasks with 1000 or 10,000 states, two actions, random successors | on-policy focusing plans faster at first and slower in the long run, with a long-lasting gain on large, narrow tasks |
| Chess and checkers | which position to study, how to choose a move | learn from positions that arise in real play; spend computation and memory on the current position |
| TD-Gammon | backgammon move selection with lookahead search | deeper search gives better moves but slower play |

## Conceptual framework

- **Sweep**: one pass through the whole state (or state-action) space, backing up each item once. It is the classical dynamic-programming distribution of backups.
- **On-policy distribution**: the distribution of states (or state-action pairs) encountered while following the current policy. It changes whenever the policy changes, and computing it exactly takes about as much work as a complete policy evaluation.
- **Trajectory sampling**: generating experience by simulating explicit trajectories with the model and backing up the pairs met along the way.
  - Transitions and rewards come from the model; actions come from the current policy.
  - An episodic task starts in the start state (or from the starting-state distribution) and runs to the terminal state; a continuing task starts anywhere and keeps simulating.
  - It yields the on-policy distribution without ever representing it.
- **Heuristic search**: the state-space planning method (see [[08a Models, planning, Dyna]]) predominant in artificial intelligence. For each state met, it grows a large tree of possible continuations, applies an approximate (hence "heuristic") value function at the leaf nodes and backs the values up to the root.
  - The **root** is the current state and the **leaves** are the states at the tree's frontier; backing up stops at the root's state-action nodes.
  - Unlike the planning of [[08a Models, planning, Dyna]], it does not change the value function; it only improves action selection given the current one. It is planning as part of computing a policy.
- **Max-backup**: a backup that takes the maximum over actions, as for $v_*$ and $q_*$ in [[03b MDPs, value functions, optimality]]. The backups inside a search tree are exactly these.
- **Search depth** $k$: the number of steps of lookahead.
- **Afterstate value function**: a value function over the position that results right after the agent's move; TD-Gammon learned one with TD($\lambda$) (see [[07a n-step TD and TD(λ)]]).

## Approach

### Trajectory sampling

1. **Why on-policy focusing looks good**: it is easy to generate (just interact with the model under the current policy) and it is hard to see another efficient way to follow that distribution.
   - An explicit representation of the on-policy distribution would allow weighting a sweep by it, but that keeps all the costs of sweeps.
   - Even if sampling pairs from the distribution could be done efficiently, the book asks what it would gain over simulating trajectories, and the distribution is not known explicitly anyway.
   - Intuition: a chess learner studies positions that arise in real games, not random arrangements of pieces; those may be valid, but valuing them is a different skill.
2. **A possible downside**: the same familiar part of the space may be backed up over and over.
3. **The experiment**: to isolate the effect of the distribution, every backup is a one-step full tabular backup (defined in [[08a Models, planning, Dyna]]).
   - **Uniform**: cycle through all state-action pairs, backing up each in place.
   - **On-policy**: simulate episodes under the current $\varepsilon$-greedy policy ($\varepsilon = 0.1$; see [[05a Monte Carlo prediction and control]]) and back up each pair that occurs.
   - **Tasks**: undiscounted and episodic, generated at random. Each of $|\mathcal{S}|$ states has two actions; each action leads to one of $b$ next states, equally likely, with its own random choice of $b$ states per pair, and $b$ is the same for all pairs. Every transition also has probability $0.1$ of going to the terminal state.
   - **Measure**: the true value of the start state $s_0$ under the greedy policy $\tilde\pi$ for the current $Q$, computed exhaustively, as a stand-in for how well the agent would do on a new episode acting greedily (assuming the model is correct).
4. **Results**: averaged over 200 sample tasks with 1000 states and $b$ of 1, 3 and 10, plotted against the number of full backups completed.
   - On-policy sampling gave faster planning at first and slower planning in the long run, in all cases.
   - The effect was stronger, and the initial advantage longer, at smaller $b$.
   - The effects also strengthened as the number of states grew: for $b = 1$ with 10,000 states the advantage of on-policy focusing was large and long-lasting.
5. **Why**:
   - **Short term**: on-policy sampling concentrates on near descendants of the start state; with many states and small $b$ this effect is large and lasts long.
   - **Long run**: the commonly occurring states already have correct values, so sampling them is useless, whereas other states still hold useful work. Hence the exhaustive, unfocused approach does better in the long run, at least on small problems.
   - **Implication**: on large problems where only a small subset of the state-action space is visited on-policy, focusing can be a great advantage.
6. **Link to function approximation**: the on-policy distribution turns out to have significant advantages when function approximation is used (see [[09a Function approximation and linear methods]]).

### Heuristic search

1. **Action selection by lookahead**: from the current state, grow a tree of continuations, evaluate leaves with the approximate value function, back values up to the root's state-action nodes with max-backups, take the best action, then discard every backed-up value.
   - In conventional heuristic search no backed-up value is saved; the value function is typically designed by people and never changed by search.
   - It is natural to let the value function improve over time, using the backed-up values or any other method of this book.
2. **An extension of the greedy policy**: computing a greedy action from a model and state-value function already means looking ahead over each action and each next state, backing up rewards and values and picking the best, without saving them. Heuristic search extends that beyond one step.
3. **Why search deeper**: with a perfect model and an imperfect action-value function, deeper search usually gives better policies (the draft notes exceptions exist).
   - Search to the end of the episode removes the effect of the imperfect value function, so the chosen action is optimal.
   - Search deep enough that $\gamma^k$ is very small gives near-optimal actions.
   - The price is more computation and slower response.
4. **TD-Gammon as the example**: it learned an afterstate value function by TD($\lambda$) over many games against itself and used a form of heuristic search to move.
   - Its model was prior knowledge of dice-roll probabilities plus the assumption that the opponent always picks the action TD-Gammon rates best for it.
   - Deeper search gave better moves but slower ones; backgammon has a large branching factor and moves must come within a few seconds, so only selective search a few steps ahead was feasible, and it still improved action selection significantly.
5. **Selective trees**: the tree is grown deeper along some lines (actions that seem most likely best) and shallower along others. Applying the same idea to distributing backups outside search, for example preferentially updating pairs whose values appear close to the maximum available from the state, is stated to be unexplored.
6. **Focus on the current state**: much of heuristic search's effectiveness comes from a tree tightly focused on what might immediately follow the current state, which is where an accurate value function is most urgently needed.
   - Memory is focused too: chess programs cannot store a value for every position, but can store distinct estimates for the millions of positions met looking ahead from one position.
7. **Search as one-step backups**: run the search to build a tree, then perform individual one-step backups from the bottom up (the ordering of a selective depth-first search). With a table-lookup representation this achieves exactly the same backup as heuristic search.
   - So any state-space search can be viewed as a piecing together of many one-step backups.
   - The gain from deeper search is therefore not due to multistep backups as such, but to the concentration of backups on states and actions immediately downstream of the current state.

### Dimensions of variation among planning methods

- **Distribution of backups**: one of the most important dimensions, the focus of search.
  - Prioritized sweeping (see [[08a Models, planning, Dyna]]) focuses on predecessors of states whose values recently changed.
  - Heuristic search focuses, among other things, on successors of the current state.
  - Trajectory sampling focuses on the on-policy distribution.
  - All can significantly speed planning and are current research topics.
- **Size of backups**: the smaller the backups, the more incremental planning can be; among the smallest are one-step sample backups, which one study suggests are preferable on very large problems.
- **Depth of backups**: in many cases a deep backup can be implemented as a sequence of shallow ones.
- **Combining processes**: planning, acting and model-learning feed each other in a circle, so they can run asynchronously in parallel, and when they share computation the split can be almost arbitrary.

## Math

**Notation.** Symbols from earlier notes ($S_t$, $A_t$, $\gamma$, $\varepsilon$, $b$, $Q$, $\hat p$, $q_*$, $v_*$) keep their meaning.

| Symbol | Meaning |
|--------|---------|
| $b$ | branching factor: number of possible next states per state-action pair |
| $\mathcal{S}$, $\lvert\mathcal{S}\rvert$ | the state set and its size |
| $s_0$ | the start state of the episodic tasks |
| $\tilde\pi$ | the greedy policy for the current action-value estimate $Q$ |
| $v_{\tilde\pi}(s_0)$ | true value of the start state under $\tilde\pi$, the quality measure of the experiment |
| $\varepsilon$ | exploration probability of the $\varepsilon$-greedy policy ($0.1$ in the experiment) |
| $\hat v$ | approximate value function applied at the leaves of a search tree |
| $k$ | search depth |
| $\hat p(s',r \mid s,a)$ | estimated dynamics of the model (distribution model, as in [[08a Models, planning, Dyna]]) |
| $\hat q_j(s,a)$, $\hat v_j(s)$ | depth-$j$ backed-up action and state values of a search tree (formalized here) |

```mermaid
flowchart LR
  P["current policy"] -->|"action"| M["model"]
  M -->|"sampled reward and next state"| T["simulated trajectory"]
  T -->|"back up each visited pair"| Q["Q table"]
  Q -->|"epsilon-greedy"| P
```

**One-step full backup used in the trajectory-sampling experiment** (the same backup as in [[08a Models, planning, Dyna]]):

$$
Q(s,a) \leftarrow \sum_{s',r} \hat p(s',r \mid s,a) \Big[ r + \gamma \max_{a'} Q(s',a') \Big]
$$

- Uniform and on-policy runs apply exactly this backup (with $\gamma = 1$ in the undiscounted tasks); they differ only in which pairs get backed up and in what order.

**Depth-$k$ lookahead of heuristic search** (formalized here; the draft describes the procedure in words, with max-backups at every level):

$$
\hat v_0(s) = \hat v(s), \qquad
\hat q_j(s,a) = \sum_{s',r} \hat p(s',r \mid s,a)\big[ r + \gamma\, \hat v_{j-1}(s') \big], \qquad
\hat v_j(s) = \max_a \hat q_j(s,a)
$$

- The action taken in the current state $S_t$ is $\arg\max_a \hat q_k(S_t,a)$; all $\hat q_j,\hat v_j$ are then discarded.
- Leaf values enter the root scaled by $\gamma^k$, which is why a deep enough search is near optimal whatever the quality of $\hat v$.

> [!example] Worked intuition
> Formalized here. Suppose the leaf value function is wrong by at most 1 in every state, $\gamma = 0.9$, and the model is perfect. The leaves' error reaches the root's backed-up values scaled by $\gamma^k$: at most $0.9$ for $k = 1$, about $0.35$ for $k = 10$ and about $0.005$ for $k = 50$. Deeper search shrinks the influence of the imperfect value function, at the price of a tree whose size grows roughly like the branching factor to the power $k$.

## Dataset example

One row per simulated step of one trajectory in the random task of the experiment; illustrative rows (the draft prints no data). The task has 2 actions per state and $b = 3$: each pair reaches each of 3 listed successors with probability $0.3$ and the terminal state with probability $0.1$ (total $0.9 + 0.1 = 1$). The behavior policy is $\varepsilon$-greedy with $\varepsilon = 0.1$.

| step $t$ | $S_t$ | $A_t$ (how chosen) | successors in model (probability) | sampled $S_{t+1}$ | pair given a full backup |
|----------|-------|--------------------|-----------------------------------|-------------------|--------------------------|
| 0 | $s_0$ | $a_2$ (greedy) | $s_{402}, s_{88}, s_{731}$ (0.3 each); terminal (0.1) | $s_{402}$ | $(s_0, a_2)$ |
| 1 | $s_{402}$ | $a_1$ (greedy) | $s_{12}, s_{88}, s_{950}$ (0.3 each); terminal (0.1) | $s_{88}$ | $(s_{402}, a_1)$ |
| 2 | $s_{88}$ | $a_2$ (exploratory) | $s_{5}, s_{402}, s_{617}$ (0.3 each); terminal (0.1) | $s_{617}$ | $(s_{88}, a_2)$ |
| 3 | $s_{617}$ | $a_1$ (greedy) | $s_{3}, s_{240}, s_{88}$ (0.3 each); terminal (0.1) | terminal | $(s_{617}, a_1)$, then the episode ends |

- **Column to symbol**: $S_t$, $A_t$ and the sampled $S_{t+1}$ form the simulated trajectory; the successor column is the model's $\hat p$ used inside the full backup of the last column.
- **Derived**: each row's backup sums over all $b$ successors and the terminal outcome, weighted by $\hat p$, so the sampling decides only which pair is backed up next, not the backup's value. A uniform run would instead cycle through all $2\,\lvert\mathcal{S}\rvert$ pairs in a fixed order.
- **Absent**: rewards, because the draft does not state them for this experiment; and any pair off the trajectory, which on-policy sampling never touches until a later episode reaches it.

## Where it matters in practice

### Signs a problem fits

- **Trajectory sampling**: a very large state space of which only a small part is reached under reasonable policies, so sweeps would waste almost all their time; the benefit is larger with many states and a small branching factor.
- **Heuristic search**: a model good enough to look ahead and a time limit per decision; with a large branching factor, as in backgammon, search can only be done selectively a few steps ahead.

### What you need to use it

- **A model**: for trajectory sampling a sample model suffices to simulate (model types in [[08a Models, planning, Dyna]]); the experiment's full backups need a distribution model. Heuristic search needs one that predicts reward and next state (in TD-Gammon: dice probabilities and an assumed opponent policy).
- **A value function or table**: an approximate value function for the leaves of a search, a $Q$ table for backups along trajectories.
- **Compute and memory focused on the present**: enough to expand a tree below each decision, and storage for the values of the positions that tree meets.

### When not to trust it

- **Long runs on small problems**: on-policy focusing can fall behind uniform backups, as the experiment showed.
- **Tight response deadlines**: deeper search gives better actions only if there is time to finish it.

## Assumptions & limitations

- **Trajectory-sampling evidence is narrow**: it comes from randomly generated tasks with equally likely successors and one-step full backups, and the draft calls the results not conclusive.
- **Monte Carlo tree search** (MCTS) is not developed in this part of the book, so nothing about it is stated in this note.
- **Scope**: tabular, one-step-backup methods; combining these ideas with function approximation is left to later chapters.
