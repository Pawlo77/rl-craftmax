---
title: "Achievement Skill Options"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - hierarchical-rl
  - options
  - skills
origin: "Sutton, Precup, Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in RL. Artificial Intelligence 112, 1999. https://doi.org/10.1016/S0004-3702(99)00052-1"
suggested-by:
  - "[[PPO (Matthews 2024)]]"
created: 2026-10-20
updated: 2026-10-20
---
# Achievement Skill Options

## Idea in plain words

- An option is a small policy that does one job and then stops, for example "get wood until I have 5" or "make an iron pickaxe".
- Train one option per job with its own simple reward (did the job get done?).
- A top-level policy picks which option to run next, so it plans in jobs, not in single key presses.
- Craftax's achievement list gives a ready-made list of jobs.

## Why it fits Craftax

- Achievements form chains of prerequisites ([[Craftax crafting and placing]], [[Craftax achievements]]); deep goals need dozens of jobs in the right order.
- A flat policy must discover the whole chain from one-step actions; with options, the top level makes only a few decisions per floor.
- Success tests for jobs are easy to write from the state (inventory counts, floor, kills).

## How it works

Builds on the MDP framework ([[03a MDP setting and returns]]) and the actor-critic ([[11 Policy approximation (actor-critic)]]).

1. Define $K$ options from the achievement list, each with a start condition, a success test and a step limit.
2. Train all options as one goal-conditioned policy (option id as input) on their own success reward.
3. Train a top-level PPO policy whose actions are option ids; it gets the native reward collected while the option runs.
4. Alternate or train both levels together.

## How to implement here

- Base: `ppo_rnn.py` for both levels ([[Craftax Baselines]]); an option wrapper that runs the low-level policy until success or step limit. About 300-500 lines.
- Extra hyperparameters: option list, step limits, low-level reward, how often the top level decides.
- Student level: partly. Hand-written options with hand-written success tests are feasible; learning the options or their definitions automatically (or with an LLM) is research-level.

## Risks

- Risk: a lot of hand design; options that fail silently (step limit) stall the top level.
- Risk: the result depends on the option list more than on the algorithm.

## Novelty check

- Searched: web "Craftax options macro-actions hierarchical", Semantic Scholar citations of arXiv 2402.16801, GitHub repositories named "craftax".
- Tried since the suggestion:
  - SCALAR: LLM-specified skills with RL-trained options; 9.1% Gnomish Mines entry ([arXiv 2603.09036](https://arxiv.org/abs/2603.09036)).
  - Unsupervised Hierarchical Skill Discovery, evaluated on Craftax ([arXiv 2601.23156](https://arxiv.org/abs/2601.23156)).
  - Achievement-oriented tasks in the Anchor-and-Harvest Curriculum ([arXiv 2610.15328](https://arxiv.org/abs/2610.15328)).
  - An Option-Critic hobby project on Craftax ([repository](https://github.com/13felixblakemore/CraftaxExperiment)), no published numbers.
- Status: explored with LLMs; a non-LLM, hand-defined option set with a learned top level was not found.

## Suggested by existing work

- Craftax paper §6 Conclusion lists "skill acquisition" among the research areas Craftax should enable ([[PPO (Matthews 2024)]]).

## How to evaluate

- Native reward % of max on Craftax-1M and Craftax-1B against blocks D and A of the [[Existing Implementations]] table; Craftax-Classic first to debug the option list.
- Success rate per option; deepest floor reached.

## References

- Sutton, R. S., Precup, D., Singh, S. "Between MDPs and semi-MDPs." Artificial Intelligence 112, 1999.
- "SCALAR." 2026. https://arxiv.org/abs/2603.09036
- "Unsupervised Hierarchical Skill Discovery." 2026. https://arxiv.org/abs/2601.23156
