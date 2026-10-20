---
title: "Reinforcement Learning: An Introduction (second edition, draft)"
type: book
authors:
  - Richard S. Sutton
  - Andrew G. Barto
venue: "Book, 2nd edition (draft)"
pdf: "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf"
tags:
  - book
  - reinforcement-learning
  - moc
created: 2026-10-14
---
# Reinforcement Learning: An Introduction (Sutton & Barto, 2nd edition draft)

> [!abstract] Scope
> The book develops reinforcement learning (RL), learning what to do from interaction so as to maximize a numerical reward, in three parts: tabular solution methods (bandits, Markov decision processes, dynamic programming, Monte Carlo, temporal-difference learning, eligibility traces, planning), approximate solution methods (function approximation, policy approximation) and frontiers (psychology, neuroscience, applications, prospects). It is a draft: several chapters are placeholders, and chapter and section numbers differ from the final book. Each chapter note below was written from the PDF; the notebooks implement the chapter's algorithms and check them against known truth.

## Chapters

Pages are the printed page numbers of the draft. A dash in the last two columns means the draft has no content to note.

| Chapter | Topic | Pages | What it settles | Note | Notebook |
|---------|-------|-------|-----------------|------|----------|
| 1 | The reinforcement learning problem | 1-26 | what RL is, its four elements (policy, reward, value function, model), and a tic-tac-toe learner that shows value learning at work | [[01 Introduction]] | [notebook](notebooks/01%20Introduction.ipynb) |
| *Part I: tabular solution methods* | | | | | |
| 2 | Multi-arm bandits | 31-52 | balancing exploration and exploitation with a single state: sample averages, optimistic values, upper-confidence-bound selection, gradient bandits | [[02 Multi-armed bandits]] | [notebook](notebooks/02%20Multi-armed%20bandits.ipynb) |
| 3 (3.1-3.5) | Agent-environment setting and returns | 53-66 | the interaction loop, goals as rewards, episodic and continuing returns, the Markov property | [[03a MDP setting and returns]] | [notebook](notebooks/03a%20MDP%20setting%20and%20returns.ipynb) |
| 3 (3.6-3.10) | Markov decision processes, value functions, optimality | 67-88 | the dynamics of a finite MDP, the Bellman equations for a policy and for the optimum | [[03b MDPs, value functions, optimality]] | [notebook](notebooks/03b%20MDPs%2C%20value%20functions%2C%20optimality.ipynb) |
| 4 | Dynamic programming | 89-112 | solving an MDP exactly when its model is known: policy evaluation, policy and value iteration, generalized policy iteration | [[04 Dynamic programming]] | [notebook](notebooks/04%20Dynamic%20programming.ipynb) |
| 5 (5.1-5.4) | Monte Carlo prediction and control | 113-127 | learning values and policies from sampled episodes without a model, with exploring starts or ε-soft policies | [[05a Monte Carlo prediction and control]] | [notebook](notebooks/05a%20Monte%20Carlo%20prediction%20and%20control.ipynb) |
| 5 (5.5-5.9) | Off-policy Monte Carlo | 127-142 | learning about one policy from another's episodes with ordinary and weighted importance sampling | [[05b Off-policy Monte Carlo]] | [notebook](notebooks/05b%20Off-policy%20Monte%20Carlo.ipynb) |
| 6 | Temporal-difference learning | 143-166 | one-step bootstrapped updates from raw experience: TD(0), Sarsa, Q-learning | [[06 Temporal-difference learning]] | [notebook](notebooks/06%20Temporal-difference%20learning.ipynb) |
| 7 (7.1-7.4) | n-step TD and TD(λ) | 167-183 | the family between one-step TD and Monte Carlo: n-step returns, the λ-return, eligibility traces | [[07a n-step TD and TD(λ)]] | [notebook](notebooks/07a%20n-step%20TD%20and%20TD%28%CE%BB%29.ipynb) |
| 7 (7.5-7.10) | Eligibility-trace control and off-policy | 183-194 | traces for control: Sarsa(λ), Watkins's Q(λ), variable λ | [[07b Eligibility-trace control and off-policy]] | [notebook](notebooks/07b%20Eligibility-trace%20control%20and%20off-policy.ipynb) |
| 8 (8.1-8.5) | Models, planning and Dyna | 195-213 | planning and learning as one activity, Dyna-Q, a wrong model, prioritized sweeping, full versus sample backups | [[08a Models, planning, Dyna]] | [notebook](notebooks/08a%20Models%2C%20planning%2C%20Dyna.ipynb) |
| 8 (8.6-8.9) | Trajectory sampling, search | 213-222 | where to spend backups: on-policy trajectories and heuristic search (the Monte Carlo tree search section is empty in the draft) | [[08b Trajectory sampling, search, MCTS]] | - |
| *Part II: approximate solution methods* | | | | | |
| 9 (9.1-9.3) | Function approximation and linear methods | 225-242 | a parameterized value function trained by gradient descent; linear features and their construction | [[09a Function approximation and linear methods]] | - |
| 9 (9.4-9.6) | Control with approximation, bootstrapping | 241-254 | approximate action values for control, mountain car, whether to bootstrap | [[09b Control with approximation, bootstrapping]] | - |
| 10 | Off-policy approximation of action values | 255-256 | two-page placeholder in the draft | - | - |
| 11 | Policy approximation | 257-264 | actor-critic methods, the average-reward setting and R-learning | [[11 Policy approximation (actor-critic)]] | - |
| *Part III: frontiers* | | | | | |
| 12 | Psychology | 269-270 | two-page placeholder in the draft | - | - |
| 13 | Neuroscience | 271-272 | two-page placeholder in the draft | - | - |
| 14.1 | TD-Gammon | 273-278 | TD learning with a neural network, by self-play, reaches strong backgammon | [[14.1 TD-Gammon]] | - |
| 14.2 | Samuel's checkers player | 279-281 | minimax search with a learned linear evaluation, and why it lacked an anchor | [[14.2 Samuel's checkers player]] | - |
| 14.3 | The Acrobot | 282-285 | Sarsa(λ) with tile coding learns a swing-up control task | [[14.3 The Acrobot]] | - |
| 14.4 | Elevator dispatching | 286-290 | continuous-time Q-learning with a network minimizes passenger waiting | [[14.4 Elevator dispatching]] | - |
| 14.5 | Dynamic channel allocation | 291-294 | afterstate values with linear TD reduce blocked calls in a cellular network | [[14.5 Dynamic channel allocation]] | - |
| 14.6 | Job-shop scheduling | 295-301 | a TD-trained network guides iterative schedule repair | [[14.6 Job-shop scheduling]] | - |
| 15 (15.1) | The unified view | 303-305 | every method as a point in a space of backup dimensions | [[15 The unified view]] | - |
| 15 (15.2-15.5) | Prospects: frontier topics | 306 | one-line stubs in the draft | - | - |

## Crucial parts

- **The problem and its vocabulary**: policy, reward signal, value function, model, and the exploration-exploitation dilemma: [[01 Introduction]], [[02 Multi-armed bandits]].
- **Markov decision processes and returns**: states, actions, rewards, the Markov property, discounting: [[03a MDP setting and returns]].
- **Bellman equations**: the recursive relations for $v_\pi$, $q_\pi$ and the optimal values, and greedy action selection from them: [[03b MDPs, value functions, optimality]].
- **Policy and value iteration, generalized policy iteration**: evaluation and improvement as two interacting processes: [[04 Dynamic programming]].
- **Monte Carlo estimation**: averaging sampled returns, exploring starts and ε-soft policies: [[05a Monte Carlo prediction and control]]; importance sampling for off-policy learning: [[05b Off-policy Monte Carlo]].
- **TD(0), Sarsa and Q-learning**: bootstrapping from raw experience, on- and off-policy control: [[06 Temporal-difference learning]].
- **n-step returns, the λ-return and eligibility traces**: the continuum between TD and Monte Carlo: [[07a n-step TD and TD(λ)]]; traces for control: [[07b Eligibility-trace control and off-policy]].
- **Planning with a model**: Dyna and prioritized sweeping: [[08a Models, planning, Dyna]]; trajectory sampling and search: [[08b Trajectory sampling, search, MCTS]].
- **Function approximation**: gradient-descent value estimation, linear methods, tile coding: [[09a Function approximation and linear methods]]; control with approximation and the bootstrapping question: [[09b Control with approximation, bootstrapping]].
- **Actor-critic and average reward**: [[11 Policy approximation (actor-critic)]].
- **The map of methods**: backup width, backup depth and function approximation as dimensions: [[15 The unified view]].
- **The methods at work**: [[14.1 TD-Gammon]], [[14.2 Samuel's checkers player]], [[14.3 The Acrobot]], [[14.4 Elevator dispatching]], [[14.5 Dynamic channel allocation]], [[14.6 Job-shop scheduling]].

## How the parts fit

- **One state to many**: bandits have a single state; Markov decision processes add states and delayed consequences, and the Bellman equations tie their values together.
- **Three ways to estimate values**: dynamic programming uses a model and full backups; Monte Carlo uses sampled whole episodes; temporal-difference learning samples one step and bootstraps. The width-and-depth plane in [[15 The unified view]] places all three, plus exhaustive search, as its corners.
- **Traces bridge TD and Monte Carlo**: n-step returns and eligibility traces give every setting between the two.
- **Learning and planning share machinery**: a model lets the same backups run on simulated experience, so the question becomes where to spend them.
- **Tables to functions**: function approximation replaces the table, so one update generalizes to many states, and bootstrapping then loses some of its guarantees; policies can also be represented separately, as in actor-critic methods.
- **Every method sits inside generalized policy iteration**: an estimate of values and a policy that is greedy (or nearly) with respect to it keep improving each other.
