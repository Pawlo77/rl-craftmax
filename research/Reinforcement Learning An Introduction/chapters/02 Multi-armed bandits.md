---
title: "Multi-arm Bandits"
parent: "[[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]"
type: book-chapter
chapter: 2
authors:
  - Richard S. Sutton
  - Andrew G. Barto
venue: "Book, Reinforcement Learning: An Introduction, 2nd edition (draft), Chapter 2"
pdf: "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf"
tags:
  - book
  - reinforcement-learning
  - chapter
created: 2026-10-14
---
# Multi-armed bandits (Sutton & Barto, Chapter 2)

> [!abstract] TL;DR
> The n-armed bandit is the simplest reinforcement learning (RL) problem: repeatedly pick one of $n$ actions, observe only the reward of the chosen one, and balance trying new actions (exploring) against using the best-looking one (exploiting). Simple estimate-and-select rules (sample averages with $\varepsilon$-greedy, optimistic starting values, upper-confidence-bound selection) and a preference-based gradient method all beat always-exploit; which works best depends on the task, and the upper-confidence-bound rule does best in the chapter's parameter study.

## Problem

- **Evaluative feedback**: RL differs from other learning in that its training signal evaluates the action taken instead of instructing which action was correct.
  - Evaluative feedback says how good the chosen action was, not whether it was the best or worst possible, so the learner must search by trial and error.
  - Instructive feedback names the correct action independently of the action taken; it underlies supervised learning (see [[01 Introduction]] for the contrast).
- **The bandit task**: face the same choice among $n$ actions again and again; each action pays a numerical reward drawn from a fixed (stationary) distribution that depends only on the action.
  - The goal is to maximize the expected total reward over a fixed number of steps, for example 1000 action selections.
  - The name comes from a slot machine ("one-armed bandit") with $n$ levers; a doctor choosing among experimental treatments for a series of patients is the other analogy.
- **Why it isolates evaluative feedback**: there is only one situation, so nothing has to be associated with different circumstances, and an action affects only its own immediate reward. This removes most of the complexity of the full RL problem (see [[03a MDP setting and returns]] for that problem).
- **Exploration versus exploitation**: a single selection cannot both explore and exploit, so the two conflict.
  - Exploiting is right for the next step; exploring can pay more in the long run, for instance when several actions look nearly as good as the greedy one but are uncertain, and many steps remain.
  - Which is better depends in a complicated way on the estimates, their uncertainties and the number of remaining steps.
- **Why not sophisticated balancing**: methods with guarantees of optimality or bounded loss assume stationarity and prior knowledge that are violated or unverifiable in applications and in the full RL problem. The chapter asks only that exploration and exploitation be balanced at all, with simple methods.

## Examples in the chapter

| Example                                       | Setting / decision                                                | What it demonstrates                                                                                   |
| --------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Slot machine, doctor choosing treatments      | pull a lever / pick a treatment, receive a payoff                 | the bandit setup and the idea of an action's value                                                     |
| 10-armed testbed                              | a suite of randomly generated 10-armed tasks, 1000 steps each     | the comparison bed for every method; learning curves of average reward and fraction of optimal choices |
| Testbed with true values shifted to mean $+4$ | a variant whose true values are drawn with mean $+4$ instead of 0 | a reward baseline makes the gradient method indifferent to the reward level                            |
| Slot machine that changes its display color   | several bandit tasks chosen at random, each signaled by a clue    | associative search: learning a policy from clue to action                                              |

## Conceptual framework

- **Action value**: $q(a)$, the mean reward when action $a$ is selected. Knowing all values makes the task trivial (always pick the highest); the learner only has estimates $Q_t(a)$.
- **Greedy action, exploiting, exploring**: a greedy action is one whose estimated value is highest at the moment.
  - Choosing it is exploiting current knowledge; choosing any other action is exploring, which improves that action's estimate.
- **Stationary versus nonstationary**: in a stationary task the reward distributions never change; in a nonstationary one the true values drift over time.
  - Effective nonstationarity is the usual case in RL: even if the task is fixed, the learner's own improving behavior changes what it faces.
- **Nonassociative versus associative**: a nonassociative task has one situation, so the learner looks for one best action (stationary case) or tracks it (nonstationary case).
  - An associative task (a contextual bandit) has several situations and the goal is a policy, a mapping from situations to the best action in each.
  - It sits between the bandit and the full RL problem: it needs a policy, but actions still affect only the immediate reward, not the next situation.
- **Step size**: the fraction $\alpha$ of the way an estimate moves toward a new observation; it may be constant or vary with the step.
- **Bias from initial estimates**: a method whose result depends on the starting values $Q_1(a)$ is biased by them. For sample averages the bias disappears once every action has been tried; with constant $\alpha$ it remains, shrinking over time.

## Approach

Most bandit methods in the chapter are built from two interchangeable parts: an **estimating rule**, which turns the rewards received so far into a value estimate $Q_t(a)$ for each action, and a **selecting rule**, which uses those estimates to pick the next action. One method is one choice of each. The gradient bandit replaces both parts with learned preferences, and associative search extends any method to several situations.

### Estimating rules (how $Q_t(a)$ is formed)

1. **Sample-average estimate**: $Q_t(a)$ is the mean of the rewards received so far from $a$, with a default such as $Q_1(a) = 0$ before the first pull.
   - By the law of large numbers (sample means converge to the true mean) it converges to $q(a)$ as $N_t(a) \to \infty$.
   - It is computed incrementally: keep only $Q_k$ and the count $k$, and fold in each new reward with constant work, instead of storing all rewards.
2. **Constant step size for nonstationary tasks**: replace the shrinking $\tfrac{1}{k}$ with a fixed $\alpha$.
   - The estimate becomes an exponential recency-weighted average: recent rewards weigh most and the weight of older ones decays geometrically.
3. **Optimistic initial values**: start every $Q_1(a)$ far above any plausible reward (the chapter uses $+5$ when true values have unit spread).
   - Whichever action is tried first pays less than the estimate, so the learner is "disappointed" and switches; all actions get tried several times even with purely greedy selection.
   - The drive to explore fades as estimates settle. It is therefore a trick for stationary tasks and cannot help when a change renews the need to explore.
   - Any method that treats the start of time specially, sample averages included, shares this weakness, since the beginning occurs only once.

### Selecting rules (how the next action is chosen from the estimates)

4. **Greedy selection**: always pick an action maximizing $Q_t(a)$, breaking ties arbitrarily. It never samples apparently inferior actions to check them.
5. **$\varepsilon$-greedy selection**: act greedily, but with small probability $\varepsilon$ pick uniformly among all actions, whatever the estimates.
   - A random pick can land on any action, so no action is ever abandoned: every action is sampled infinitely often, all $Q_t(a)$ converge to $q(a)$, and the greedy part then picks the best action. In the limit the best action is chosen with probability $1 - \varepsilon + \varepsilon/n$, a little above $1 - \varepsilon$ ($n$ is the number of actions).
   - This is only an asymptotic guarantee and says little about practical speed; the fraction $\varepsilon$ of random picks is also never given up.
6. **Upper-confidence-bound (UCB) selection**: pick the action with the highest estimate plus an uncertainty bonus.
   - $\varepsilon$-greedy explores indiscriminately; UCB prefers actions that are near-greedy or poorly known.
   - The bonus shrinks each time the action is picked (its count grows) and grows slowly, through the logarithm, each time another action is picked, so every action is eventually selected but worse or well-sampled ones less often.
   - An action never tried counts as a maximizing action.
   - The parameter $c$ sets the degree of exploration.

### Combinations the chapter tests

| Name | Estimating rule | Selecting rule |
|------|-----------------|----------------|
| greedy | sample average | greedy |
| $\varepsilon$-greedy | sample average | $\varepsilon$-greedy |
| $\varepsilon$-greedy on a nonstationary task | constant step size | $\varepsilon$-greedy |
| optimistic greedy | constant step size with optimistic initial values | greedy |
| UCB | sample average | UCB |

### Preferences and several situations

7. **Gradient bandit**: learn a numerical preference $H_t(a)$ per action instead of a value, so it has neither an estimating nor a selecting rule of the kind above.
   - Only differences in preference matter: adding a constant to all preferences leaves the probabilities unchanged, and the soft-max turns preferences into action probabilities.
   - After each reward, the chosen action's preference rises if the reward beat a baseline (the running average of rewards) and falls if it did not; the other actions move the opposite way.
   - It is stochastic gradient ascent, that is, gradient ascent on expected reward with the exact gradient replaced by a sample of it (shown in Math), so it inherits the robust convergence behavior of that family.
8. **Associative search**: when a clue identifies which bandit task is active, learn one set of estimates per clue, giving a policy such as "if red, play arm 1; if green, play arm 2".
   - Without the clue, the changing task looks like one nonstationary bandit and the methods above work poorly unless values change slowly.

### The 10-armed testbed and what it shows

- **Setup**: 2000 randomly generated tasks with $n = 10$; each task's true values $q(a)$ are drawn from a normal distribution with mean 0 and variance 1.
  - The reward at step $t$ is $q(A_t)$ plus a normal noise term with mean 0 and variance 1; curves are averaged over the tasks and plotted over 1000 steps.
  - The best possible average reward on this testbed is about 1.55.
- **Greedy versus $\varepsilon$-greedy**: greedy improves slightly faster at first, then levels off near a reward of about 1, because it gets stuck on suboptimal actions.
  - It finds the optimal action in only about one-third of the tasks; in the rest its first samples of the optimal action disappointed and it never returned.
- **Choosing $\varepsilon$**: $\varepsilon = 0.1$ explores more and finds the optimal action earlier but never picks it more than about 91% of the time; $\varepsilon = 0.01$ is slower but eventually better on both reward and optimal-action fraction. Reducing $\varepsilon$ over time can combine both advantages.
- **How the advantage depends on the task**: noisier rewards (variance 10 instead of 1) need more exploration, so $\varepsilon$-greedy gains over greedy.
  - With zero reward variance, greedy knows each value after one trial and may win, unless the task is nonstationary, where exploring is needed even without noise.
- **Optimistic greedy versus realistic $\varepsilon$-greedy** (both with $\alpha = 0.1$): the optimistic method is worse at first because it explores more, then better because its exploration decays.
- **UCB versus $\varepsilon$-greedy** (UCB with $c = 2$, $\varepsilon = 0.1$): UCB generally does better, except for the first $n$ plays, when it picks randomly among the not-yet-tried actions. It is harder to extend beyond bandits: nonstationary tasks need more than the constant-step fix, and with large state spaces and function approximation no practical use of the UCB idea is known.
- **Baseline matters**: on the testbed shifted to mean $+4$, the gradient method with the average-reward baseline is unaffected by the shift, while without the baseline it degrades significantly.
- **Parameter study**: each method's average reward over the first 1000 steps, plotted against its own parameter on a log scale (factors of two), shows an inverted U.
  - Every method does best at an intermediate parameter value and performs well over a range of about an order of magnitude, so all are fairly insensitive.
  - Overall on this problem UCB looks best.

## Math

**Notation.** The chapter indexes by time step $t$ for selections and by $k$ for the $k$th reward of one action; it writes the true value as $q(a)$ (a function of the action only, because a bandit has a single situation; later notes use $q_\pi$ for action values under a policy) and the policy as $\pi_t(a)$ only for the gradient method.

| Symbol | Meaning |
|--------|---------|
| $n$ | number of actions |
| $A_t$, $R_t$ | action selected and reward received at step $t$ |
| $q(a)$ | true value of action $a$: its mean reward |
| $Q_t(a)$ | estimate of $q(a)$ at step $t$; $Q_1(a)$ is the initial estimate |
| $N_t(a)$ | number of times $a$ was chosen before step $t$ |
| $A_t^*$ | a greedy action at step $t$, one with $Q_t(A_t^*) = \max_a Q_t(a)$ |
| $R_k$, $Q_k$ | $k$th reward received from one action, and that action's estimate after $k-1$ rewards (the same action is implicit) |
| $\alpha$, $\alpha_k(a)$ | step size: constant, or the value used for the reward after the $k$th selection of $a$ |
| $\varepsilon$ | probability of a random action in $\varepsilon$-greedy |
| $c$ | exploration degree in UCB |
| $H_t(a)$ | preference for action $a$ in the gradient bandit |
| $\pi_t(a)$ | probability of choosing $a$ at step $t$ under the soft-max |
| $\bar R_t$ | average of all rewards up to and including step $t$ (baseline) |
| $\mathbb{I}_{a=b}$ | 1 if $a = b$, else 0 |
| $\mathbb{E}$ | expectation |

**Sample-average action-value estimate:**

$$
Q_t(a) = \frac{R_1 + R_2 + \cdots + R_{N_t(a)}}{N_t(a)}
$$

- The mean of the rewards actually received after choosing $a$; defined by a default value when $N_t(a) = 0$.

**Greedy action selection:**

$$
A_t = \operatorname*{argmax}_a Q_t(a)
$$

- Picks a greedy action $A_t^*$, a value-maximizing one (ties broken arbitrarily).

**Incremental sample average:**

$$
Q_{k+1} = Q_k + \frac{1}{k}\big[R_k - Q_k\big]
$$

- Equals the average of $k$ rewards, using memory only for $Q_k$ and $k$; it holds for $k = 1$ for any $Q_1$, giving $Q_2 = R_1$.
- Derivation: write $Q_{k+1} = \tfrac{1}{k}\big(R_k + (k-1)Q_k\big)$ and add and subtract $Q_k$ inside the bracket.

**General update form:**

$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize}\,\big[\text{Target} - \text{OldEstimate}\big]
$$

- The bracket is an error in the estimate; each update steps toward the target, which may be noisy. The same shape recurs through the book, for example in the tic-tac-toe value update of [[01 Introduction]].
- Here the target is the $k$th reward and the step size is $\tfrac{1}{k}$, informally written $\alpha = \tfrac{1}{k}$.

**Constant-step update and its weights:**

$$
Q_{k+1} = Q_k + \alpha\big[R_k - Q_k\big] = (1-\alpha)^k Q_1 + \sum_{i=1}^{k} \alpha (1-\alpha)^{k-i} R_i
$$

- With $\alpha \in (0, 1]$ the weights on $Q_1$ and on all rewards sum to 1, and reward $R_i$ gets weight $\alpha(1-\alpha)^{k-i}$, decaying exponentially with how many rewards ago it arrived.
- At $\alpha = 1$ all the weight falls on the last reward $R_k$.

**Convergence conditions on the step-size sequence:**

$$
\sum_{k=1}^{\infty} \alpha_k(a) = \infty \qquad \text{and} \qquad \sum_{k=1}^{\infty} \alpha_k^2(a) < \infty
$$

- Together they guarantee convergence with probability 1: the first makes steps large enough to overcome initial conditions and random fluctuations, the second makes them eventually small enough to settle.
- Both hold for $\alpha_k(a) = \tfrac{1}{k}$; the second fails for constant $\alpha$, so those estimates keep tracking recent rewards.
- Sequences meeting both conditions often converge slowly or need much tuning, so they are seldom used in applications.

**Upper-confidence-bound action selection:**

$$
A_t = \operatorname*{argmax}_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

- The square-root term measures the uncertainty in $Q_t(a)$, so the bracket is an upper bound on the plausible true value; $c > 0$ sets the confidence level.

**Soft-max action probabilities:**

$$
\Pr\{A_t = a\} = \frac{e^{H_t(a)}}{\sum_{b=1}^{n} e^{H_t(b)}} = \pi_t(a)
$$

- Initially all preferences are equal (for example $H_1(a) = 0$), so all actions are equally likely.

**Gradient-bandit preference update:**

$$
\begin{aligned}
H_{t+1}(A_t) &= H_t(A_t) + \alpha\big(R_t - \bar R_t\big)\big(1 - \pi_t(A_t)\big), \\
H_{t+1}(a) &= H_t(a) - \alpha\big(R_t - \bar R_t\big)\pi_t(a), \qquad \forall a \ne A_t
\end{aligned}
$$

- With $\alpha > 0$. A reward above the baseline raises the chosen action's probability and lowers the others'; a reward below does the reverse.
- The baseline $\bar R_t$ is computed incrementally like any average (by the incremental sample average, or with constant $\alpha$ if the task is nonstationary).

**Gradient-ascent view of the update.** Exact gradient ascent on expected reward would set

$$
H_{t+1}(a) = H_t(a) + \alpha \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}, \qquad \mathbb{E}[R_t] = \sum_b \pi_t(b)\, q(b)
$$

and the update above equals this in expectation, so the algorithm is stochastic gradient ascent (the true $q(b)$ are unknown, so the exact version cannot be run).

- **Step 1, add a baseline for free:** the probabilities sum to 1, so $\sum_b \partial \pi_t(b) / \partial H_t(a) = 0$ and any scalar $X_t$ not depending on $b$ can be subtracted from $q(b)$:

$$
\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \sum_b \big(q(b) - X_t\big)\frac{\partial \pi_t(b)}{\partial H_t(a)}
$$

- **Step 2, turn the sum into an expectation:** multiply and divide by $\pi_t(b)$, read the sum over $b$ as an expectation over $A_t$, choose $X_t = \bar R_t$ and replace $q(A_t)$ by $R_t$ (allowed because $\mathbb{E}[R_t] = q(A_t)$ given $A_t$):

$$
\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E}\!\left[\big(R_t - \bar R_t\big)\frac{\partial \pi_t(A_t)}{\partial H_t(a)} \Big/ \pi_t(A_t)\right]
$$

- **Step 3, derivative of the soft-max:** by the quotient rule and $\partial e^x / \partial x = e^x$,

$$
\frac{\partial \pi_t(b)}{\partial H_t(a)} = \pi_t(b)\big(\mathbb{I}_{a=b} - \pi_t(a)\big)
$$

- **Step 4, substitute:** the factor $\pi_t(A_t)$ cancels, leaving a sampleable quantity whose use as the increment gives exactly the update above:

$$
\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E}\!\left[\big(R_t - \bar R_t\big)\big(\mathbb{I}_{a=A_t} - \pi_t(a)\big)\right]
$$

- **What the baseline may be**: anything that does not depend on the selected action, such as 0 or 1000. It does not change the expected update, but it changes the update's variance and so the convergence rate.

> [!example] Worked intuition
> Formalized here with illustrative numbers. With $\alpha = 0.5$ after $k = 3$ rewards, the weights in the constant-step average are $0.5$ on $R_3$, $0.25$ on $R_2$, $0.125$ on $R_1$ and $0.125$ on $Q_1$, which sum to 1. A reward two steps old counts a quarter as much as the newest, while under $\tfrac{1}{k}$ averaging each of the three rewards would count equally.

## Dataset example

One row per time step of a single illustrative bandit run with $n = 3$ actions, $Q_1(a) = 0$ for all $a$, sample-average estimates and $\varepsilon$-greedy selection; the rewards are made up (the chapter prints no data). $k$ is the count of the chosen action including the current reward, so $\alpha = 1/k$.

| $t$ | $A_t$ | selection | $N_t(A_t)$ | $Q_t(A_t)$ before | $R_t$ | $k$ | $Q_{t+1}(A_t)$ after |
|-----|-------|-----------|------------|-------------------|-------|-----|----------------------|
| 1 | 2 | greedy (all tied at 0) | 0 | 0 | 1 | 1 | 1 |
| 2 | 1 | exploratory | 0 | 0 | 0.5 | 1 | 0.5 |
| 3 | 2 | greedy | 1 | 1 | 3 | 2 | 2 |
| 4 | 2 | greedy | 2 | 2 | 2 | 3 | 2 |
| 5 | 2 | greedy | 3 | 2 | -1 | 4 | 1.25 |

- **Column to symbol**: the columns are $t$, $A_t$, $N_t(A_t)$, $Q_t(A_t)$, $R_t$ and $Q_{t+1}(A_t)$; the last column applies the incremental sample-average update with $k = N_t(A_t) + 1$.
- **Hand check**: step 5 gives $2 + \tfrac{1}{4}(-1 - 2) = 1.25$, equal to the plain mean of $1, 3, 2, -1$ for action 2; step 1 shows that the first reward sets the estimate exactly, whatever $Q_1$ was.
- **Deliberately absent**: the true values $q(a)$, which the learner never sees; and the other actions' rewards, which evaluative feedback never reveals (action 3 is never tried here, so its estimate is still $Q_1(3) = 0$).

## Where it matters in practice

### Signs a problem fits

- The same choice among a modest set of discrete options recurs many times, and each choice returns a numerical reward.
- Feedback arrives only for the option chosen, so options must be tried to be evaluated.
- Each choice affects only its own reward, not later situations.
- A clue that identifies the situation is available, which turns it into an associative (contextual) search problem.

### What you need to use it

- **A numerical reward after every choice** and the ability to repeat choices many times.
- **A small amount of memory per action**: an estimate and a count, with constant work per reward (or a preference per action for the gradient method).
- **A parameter chosen by the user**: $\varepsilon$, $\alpha$, $c$ or the initial value.
- **For nonstationary tasks**, a constant step size (or another recency weighting) rather than plain averages.

### When not to trust it

- **Convergence statements are asymptotic**: they say little about performance within a finite budget of steps.
- **Testbed rankings are task-specific**: UCB's lead comes from one synthetic family of tasks with normally distributed values and noise, and the chapter itself calls the whole question of the best method difficult to answer in general.
- **The simple methods are not a final answer**: the chapter calls them state of the art because the more sophisticated alternatives are impractical for the full RL problem, yet far from a satisfactory solution to the balance.

## Assumptions & limitations

- **Stationarity for the plain averages**: sample averages and optimistic starts assume the true values stay put; the chapter handles nonstationarity only with constant steps and leaves anything more elaborate open.
- **Gittins indices**: the classical optimal solution for a more general class of bandit problems, but it assumes the prior distribution of possible problems is known, and neither its theory nor its computation appears to carry over to full RL.
- **Bayes-optimal balancing**: assuming a known distribution over problem instances, each reward updates a posterior distribution over action values (the information state), and one can search the tree of all possible action-and-reward sequences over the horizon.
  - The tree is intractable: with two actions, two rewards and 1000 plays it has $2^{2000}$ leaves.
  - Exact computation is infeasible, though efficient approximations may exist; this turns the bandit into a full RL problem and is left to research.
