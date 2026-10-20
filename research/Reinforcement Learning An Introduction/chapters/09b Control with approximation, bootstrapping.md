---
title: "On-policy approximation of action values: control and bootstrapping"
parent: "[[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]"
type: book-chapter
chapter: 9
section_range: "9.4-9.6"
authors:
  - Richard S. Sutton
  - Andrew G. Barto
venue: "Book, Reinforcement Learning: An Introduction, 2nd edition (draft), Chapter 9"
pdf: "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf"
tags:
  - book
  - reinforcement-learning
  - chapter
created: 2026-10-14
---
# Control with function approximation and bootstrapping (Sutton & Barto, Chapter 9 part b)

> [!abstract] TL;DR
> Prediction with a parameterized value function extends to control by approximating action values and acting greedily or $\varepsilon$-greedily on them, giving linear gradient-descent Sarsa($\lambda$) (an on-policy control method based on temporal-difference learning, [[06 Temporal-difference learning]]) and a Watkins's Q($\lambda$) variant with binary features. Bootstrapping (building estimates from other estimates) has weaker guarantees than not bootstrapping, yet in practice it usually performs much better, and why is left open.

Part a, [[09a Function approximation and linear methods]], defines the weight vector, feature vectors, gradient-descent temporal-difference (TD) learning and linear methods (tile coding, Kanerva coding); this note uses them and adds only what control and the bootstrapping question need.

## Problem

- **From prediction to control**: part a estimated a state-value function $v_\pi$ for a fixed policy; control must also improve the policy, so it needs action values.
- **Exploration still has to be ensured**: as in the tabular case, either an on-policy or an off-policy approach is used.
- **Eligibility traces break**: a trace is kept per state in the tabular case, but with function approximation each weight serves many states, so the replacing-trace rule (reset a state's trace to 1) has no direct form.
- **Large or continuous action sets**: picking the best action from an approximate action-value function is easy for a few discrete actions and unresolved otherwise.
- **Should we bootstrap?** Methods without bootstrapping have stronger guarantees with function approximation, so bootstrapping needs a justification.

## Examples in the chapter

| Example | Setting / decision | What it demonstrates |
|---------|--------------------|----------------------|
| Mountain car | continuous control: drive an underpowered car up a steep slope with three throttle actions | Sarsa($\lambda$) with tile-coded binary features learns a near-optimal policy; optimistic initial values drive exploration even with $\varepsilon = 0$; effect of $\alpha$, $\lambda$ and trace type |
| Random walk, mountain car, puddle world, cart and pole (a sweep of $\lambda$) | prediction on a random walk, control on the other three | performance becomes much worse as $\lambda \to 1$ in every task |

## Conceptual framework

- **Approximate action-value function** $\hat q(s, a, \mathbf{w}) \approx q_\pi(s, a)$: a parameterized form with weight vector $\mathbf{w}$, as in part a, but taking a state-action pair as input. $q_\pi$ is defined in [[03b MDPs, value functions, optimality]].
- **Training example**: a pair $S_t, A_t \mapsto Q_t$, where the target $Q_t$ is any approximation of $q_\pi(S_t, A_t)$, such as the full Monte Carlo (MC) return or the one-step Sarsa-style return (see the Math section). Here $Q_t$ is a target, not the tabular estimate of [[07b Eligibility-trace control and off-policy]].
- **Greedy action**: for a discrete, not too large action set, compute $\hat q(S_t, a, \mathbf{w})$ for every available $a$ and take the maximizer.
- **Estimation policy versus acting policy**:
  - The estimation policy is the one whose values are being learned.
  - On-policy methods use one policy for both and make it a soft approximation of greedy, such as $\varepsilon$-greedy ([[05a Monte Carlo prediction and control]]).
  - Off-policy methods make the estimation policy greedy and let any policy choose the actions.
- **Binary features and feature sets** $\mathcal{F}_a$: a binary feature is 0 or 1 (as from tile coding); $\mathcal{F}_a$ is the set of indices of features present for the current state and action $a$.
- **Accumulating versus replacing traces**: the trace of a feature is incremented by 1 (accumulating) or set to 1 (replacing) each time the feature is present; both are defined for tabular methods in [[07b Eligibility-trace control and off-policy]].
- **Clearing nonselected actions**: an optional rule that sets to 0 the traces of the features of the state paired with actions not chosen.
- **Bootstrapping**: updating toward a target that contains the current estimate, as TD learning does, instead of toward a complete return ([[07b Eligibility-trace control and off-policy]]). In TD($\lambda$) with traces, $\lambda = 0$ is pure bootstrapping and $\lambda = 1$ is none.
- **Sarsa and TD error**: Sarsa updates a pair's value from the quintuple $S, A, R, S', A'$; the TD error is the reward plus the discounted next estimate minus the current estimate ([[06 Temporal-difference learning]]).
- **Root-mean-squared error (RMSE)**: the square root of the mean squared difference between estimated and true values; a prediction metric, not a measure of policy quality.

## Approach

1. **Extend prediction to action values**: replace the state-value training pair (state to value target) by $S_t, A_t \mapsto Q_t$ and apply the same gradient-descent update, with $\hat q$ in place of the state-value approximator.
2. **Couple with policy improvement and action selection**: for each action in the current state compute $\hat q$, find the greedy action, then follow the estimation-policy / acting-policy split above. This is generalized policy iteration (GPI, [[05a Monte Carlo prediction and control]]) with the evaluation step done by gradient descent.
3. **Gradient-descent Sarsa($\lambda$)**: the backward-view update analogous to TD($\lambda$), driven by a TD error and a trace vector $\mathbf{e}$ with one component per weight.
   - For a constant policy it converges like TD($\lambda$), with the same kind of error bound (Math).
4. **Linear versions with binary features**: the linear algorithms in Math work with $\mathcal{F}_a$ instead of vectors, so cost scales with the number of present features.
   - **Sarsa($\lambda$) (on-policy)**: picks the next action $A'$ from an $\varepsilon$-greedy policy before updating and uses it in the target.
   - **Watkins's Q($\lambda$) (off-policy)**: acts $\varepsilon$-greedily but targets the greedy value of the next state, with the trace cut described in [[07b Eligibility-trace control and off-policy]].
   - If each action's value is a separate linear function of the same features, the index sets $\mathcal{F}_a$ coincide across actions, which simplifies the computation.
5. **Replacing traces with function approximation**: treat each feature as if it were a state, so the trace of every present feature is set to 1.
   - With state-action traces, optionally clear the traces of the features for the nonselected actions first, then set to 1 those of the selected action.
   - As in the tabular case, this may or may not be the best choice.
6. **Mountain car**: Sarsa($\lambda$) with replacing traces and clearing, state variables position and velocity turned into binary features by ten $9 \times 9$ grid tilings (each offset by a random fraction of a tile width), learned a near-optimal policy within 100 episodes.
   - Initial action values of 0 are optimistic because every true value is negative, so even with $\varepsilon = 0$ the agent leaves each visited region (valued worse after real, negative rewards) for unvisited ones.
   - Early in learning, before one episode has ended, the car oscillates in the valley along circular trajectories in state space while all frequently visited states are valued worse than unexplored ones.
7. **Bootstrap or not**: compare TD methods with traces while varying $\lambda$ from 0 to 1 on all four tasks in the examples table.
   - Results: performance gets much worse as $\lambda$ approaches 1, for both trace types where both are shown; on mountain car, accumulating traces degrade earlier than replacing traces, and in a study of $\alpha$, $\lambda$ and trace type (steps per episode over the first 20 episodes, averaged over 30 runs, with five $9 \times 9$ tilings instead of the ten used above) intermediate $\lambda$ gave the fewest steps.
   - In the random-walk prediction task, with RMSE averaged over the first episodes, $\lambda = 1$ must be best asymptotically but is much worse short of the asymptote.
   - Nonbootstrapping methods do reduce RMSE better, but RMSE is not the goal: adding 1000 to every true action value ruins RMSE and leaves the optimal policy unchanged. Nothing that simple explains bootstrapping, yet it "does something right".
8. **Overall picture**: any supervised function approximator can be used by treating each backup as a training example, and gradient descent extends all earlier techniques, traces included; backpropagation in multilayer networks is the nonlinear case, and linear methods work well given appropriate features.
   - Choosing the features is one of the most important ways to add prior domain knowledge.
   - For the on-policy case the extension is mostly straightforward; the off-policy case is subtler and postponed to a later chapter.

## Math

**Notation.** Symbols follow [[07b Eligibility-trace control and off-policy]] and part a; the draft's own symbols are kept, except that its typographical slips in formulas are corrected silently (for instance the one-step return below).

| Symbol | Meaning |
|--------|---------|
| $S_t, A_t, R_{t+1}$ | state, action and next reward at time $t$ |
| $S, A, R, S', A'$ | current and next state, action and reward in the linear algorithms below |
| $\mathbf{w}, \mathbf{w}_t$ | weight vector (of the approximation), one component per feature |
| $\hat q(s, a, \mathbf{w})$ | approximate action value, $\approx q_\pi(s, a)$ |
| $\nabla \hat q(s, a, \mathbf{w})$ | gradient of $\hat q$ with respect to $\mathbf{w}$ |
| $Q_t$ | training target for $\hat q(S_t, A_t, \mathbf{w}_t)$ |
| $G_t$ | complete return |
| $\delta_t$ | TD error |
| $\mathbf{e}_t, e_i$ | eligibility-trace vector and its component for weight $i$ |
| $\mathcal{F}_a$ | indices of the binary features present for the current state and action $a$ |
| $Q_a$ | in the linear algorithms, $\sum_{i \in \mathcal{F}_a} w_i$, the linear value of action $a$ |
| $A^*$ | greedy action, $\arg\max_a Q_a$ |
| $\mathcal{A}(s)$ | the actions available in state $s$ |
| $\alpha, \gamma, \lambda, \varepsilon$ | step size, discount, trace decay, exploration probability |
| $p_t, \dot p_t$ | mountain-car position and velocity |

```mermaid
flowchart LR
  A["features of S and each action a"] --> B["q-hat for each a"]
  B --> C["epsilon-greedy action A"]
  C --> D["observe R and S'"]
  D --> E["TD error delta"]
  E --> F["update weights w by alpha, delta and traces e"]
  F --> A
```

**Gradient-descent update for action values:**

$$
\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \Big[ Q_t - \hat q(S_t, A_t, \mathbf{w}_t) \Big] \nabla \hat q(S_t, A_t, \mathbf{w}_t)
$$

- Move the weights to shrink the error between the target $Q_t$ and the current estimate for the visited pair.
- The target $Q_t$ may be the full return $G_t$, or the one-step Sarsa-style return $R_{t+1} + \gamma \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_t)$.

**Gradient-descent Sarsa($\lambda$), backward view:**

$$
\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \mathbf{e}_t
$$

$$
\delta_t = R_{t+1} + \gamma \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat q(S_t, A_t, \mathbf{w}_t)
$$

$$
\mathbf{e}_t = \gamma \lambda \mathbf{e}_{t-1} + \nabla \hat q(S_t, A_t, \mathbf{w}_t), \qquad \mathbf{e}_0 = \mathbf{0}
$$

- It estimates $q_\pi$ of the policy generating $A_{t+1}$ (the $\varepsilon$-greedy policy with respect to $\hat q$ in the control version).
- Every weight moves in proportion to its trace, which accumulates the gradients of recently visited pairs and fades by $\gamma\lambda$ per step.

**Greedy action and policy improvement:**

$$
A^*_t = \arg\max_a \hat q(S_t, a, \mathbf{w}_t)
$$

- Estimation policy: greedy (off-policy) or $\varepsilon$-greedy (on-policy); acting policy: the same $\varepsilon$-greedy policy (on-policy) or any policy (off-policy).

**Linear Sarsa($\lambda$) with binary features** (one step; the algorithm is the book's, the numbered layout is formalized here):

1. For $i \in \mathcal{F}_A$: $e_i \leftarrow e_i + 1$ (accumulating) or $e_i \leftarrow 1$ (replacing).
2. Take $A$, observe $R, S'$; $\delta \leftarrow R - \sum_{i \in \mathcal{F}_A} w_i$.
3. If $S'$ is terminal: $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \mathbf{e}$ and end the episode.
4. For all $a$ in $S'$: $\mathcal{F}_a \leftarrow$ features of $(S', a)$, $Q_a \leftarrow \sum_{i \in \mathcal{F}_a} w_i$; choose $A'$ ($\varepsilon$-greedy).
5. $\delta \leftarrow \delta + \gamma Q_{A'}$; $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \mathbf{e}$; $\mathbf{e} \leftarrow \gamma\lambda \mathbf{e}$; $S \leftarrow S'$, $A \leftarrow A'$.

- With binary features, $\nabla \hat q$ is 1 on present features and 0 elsewhere, so the trace rule adds 1 to exactly the features in $\mathcal{F}_A$.
- Weights are initialized, e.g., to $\mathbf{0}$, and $\mathbf{e} = \mathbf{0}$ at each episode start.

**Watkins's Q($\lambda$) with binary features** differs from the Sarsa steps in four places:

- The action is chosen before the update: $A^* \leftarrow \arg\max_a Q_a$ and $A \leftarrow A^*$ with probability $1 - \varepsilon$, else a random action in $\mathcal{A}(S)$.
- If $A \neq A^*$, set $\mathbf{e} = \mathbf{0}$ (the traces of $\mathcal{F}_A$ are incremented or reset later in the step, after the action is taken).
- The target uses the greedy next value: $\delta \leftarrow \delta + \gamma \max_{a \in \mathcal{A}(S')} Q_a$.
- There is no $A'$ step: the next action is chosen at the top of the loop.

**Mountain-car dynamics:**

$$
\dot p_{t+1} = \text{bound}\Big[\dot p_t + 0.001 A_t - 0.0025 \cos(3 p_t)\Big], \qquad p_{t+1} = \text{bound}\Big[p_t + \dot p_{t+1}\Big]
$$

- $A_t \in \{-1, 0, +1\}$ is the throttle; $\cos(3p)$ enters the gravity term, which on the steep slope is stronger than the engine, so full throttle alone cannot climb.
- The bound keeps $-1.2 \le p_{t+1} \le 0.5$ and $-0.07 \le \dot p_{t+1} \le 0.07$; at the left bound $\dot p_{t+1}$ is reset to 0, and reaching the right bound is the goal and ends the episode.
- Reward is $-1$ on every step until the goal; each episode starts at a position and velocity drawn uniformly from the ranges.
- Parameters of the learned run: $\lambda = 0.9$, $\varepsilon = 0$, $\alpha = 0.05$.

**Error bound for on-policy bootstrapping with a linear approximator:**

$$
\text{MSE}(\mathbf{w}_\infty) \le \frac{1 - \gamma\lambda}{1 - \gamma} \, \text{MSE}_{\min}
$$

- The limit weights have mean-squared error (MSE) at most the factor $\frac{1-\gamma\lambda}{1-\gamma}$ times the smallest error any weight vector of the linear form can reach; here $\mathbf{w}_\infty$ and $\text{MSE}_{\min}$ are names introduced in this note for the limit solution and the minimum possible error.
- Formalized here: the factor is 1 at $\lambda = 1$ (no bootstrapping loses nothing) and $\frac{1}{1-\gamma}$ at $\lambda = 0$.

> [!example] Worked intuition
> Formalized here with illustrative numbers. Take mountain car with ten tilings, so ten binary features are present for the current pair, all weights 0, all traces 0, replacing traces, $\alpha = 0.05$. After the first step $R = -1$, and suppose $\gamma Q_{A'} = 0$ (the next pair's features are untouched). Then $\delta = -1 - 0 + 0 = -1$, each of the ten present weights becomes $0 + 0.05 \cdot (-1) \cdot 1 = -0.05$, and $\hat q$ for that pair becomes $-0.5$. All unvisited pairs still read 0, which is better than $-0.5$, so a greedy agent now prefers them: the optimism mechanism behind exploration with $\varepsilon = 0$.

## Dataset example

One row per time step of one mountain-car episode, the state variables at $t$, the chosen action and the transition produced by the dynamics; illustrative rows (the chapter prints no data), computed by hand from the dynamics above and rounded to four decimals.

| step type | $p_t$ | $\dot p_t$ | $A_t$ | $R_{t+1}$ | $\dot p_{t+1}$ | $p_{t+1}$ |
|-----------|-------|------------|-------|-----------|----------------|-----------|
| start, valley floor | -0.5000 | 0.0000 | +1 | -1 | 0.0008 | -0.4992 |
| swinging left | -0.9000 | -0.0200 | -1 | -1 | -0.0187 | -0.9187 |
| left bound hit | -1.1900 | -0.0300 | -1 | -1 | 0.0000 (reset; unbounded value -0.0287) | -1.2000 (clipped from -1.2187) |
| goal passed (terminal) | 0.4900 | 0.0600 | +1 | -1 (assumed) | 0.0607 | 0.5 (clipped from 0.5507) |

- **Column to symbol**: $p_t, \dot p_t$ are the continuous state, $A_t$ the throttle, $R_{t+1}$ the reward; the two continuous values are what tile coding turns into the binary features $\mathcal{F}_a$ used in the linear algorithms.
- **What is derived**: for each present feature the trace and weight updates; at the terminal row, step 3 of the Sarsa steps applies and there is no $A'$.
- **Reward on the last row**: the book says $-1$ on all steps until the goal is passed; whether the crossing step itself yields $-1$ is not stated, so that value is an assumption.
- **Absent**: the features (tile indices), the weights, and the explicit action values.

## Where it matters in practice

### Signs a problem fits

- Continuous or very large state space, so a table is impossible, with a discrete action set small enough to evaluate every action.
- Tasks where progress requires first getting worse by the reward measure (the car must move away from the goal), which many control methods handle poorly without a designer's help.

### What you need to use it

- **Features**: binary features, for example from tile coding, that encode what is known about the task.
- **A scalar reward and episodes or steps of experience** to compute TD errors.
- **Per-step cost proportional to the number of present features**, with traces kept for every weight.
- **Tuning**: step size $\alpha$ and trace decay $\lambda$ interact with the kind of trace, so they need joint tuning.

### When not to trust it

- **$\lambda$ near 1 chosen to avoid bootstrapping**: with limited data, the nonbootstrapping end did much worse in every task tested.
- **Value error as a yardstick**: a method that wins on RMSE need not give the better policy.
- **Large or continuous action sets**: there is no settled technique for improvement and action selection.

## Assumptions & limitations

- **Scope of the guarantee**: the error bound is stated for on-policy bootstrapping with linear gradient-descent approximation; more bootstrapping loosens it, though practical results favor bootstrapping.
- **No convergence proof for the Sarsa control method**; the Q-learning-style control method is known to be unsound and to diverge on some problems, and there are proofs only for special cases such as state aggregation (one value per group of states).
- **Nonbootstrapping methods have the stronger theory**: they work with function approximation more reliably and over a broader range of conditions, reach a lower asymptotic error even under the on-policy distribution, and with traces at $\lambda = 1$ can run on-line step by step.
- **Open question**: why partial bootstrapping beats none is unexplained; it may learn faster, or it may learn something better; this is unresolved.
- **Off-policy approximation** is not covered here and is harder.
