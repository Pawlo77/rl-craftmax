---
title: "Option-Critic"
type: candidate
status: rejected
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - hierarchical-rl
  - options
origin: "Bacon, Harb, Precup. The Option-Critic Architecture. AAAI 2017. https://arxiv.org/abs/1609.05140"
created: 2026-10-20
updated: 2026-10-20
---
# Option-Critic

## Idea in plain words

- Learn the options themselves, end to end: each option has its own policy and its own termination function ("when to stop").
- A policy over options picks which option runs; all parts are trained by policy gradients on the task reward.
- No hand-written skill list.

## Verdict

- Rejected: not in the textbook, and known to be hard to make work.
- Learned terminations tend to collapse: options stop after one step or one option does everything, which needs extra regularisers to avoid.
- On Craftax: an Option-Critic hobby project exists without published numbers ([repository](https://github.com/13felixblakemore/CraftaxExperiment)); unsupervised skill hierarchies are studied by HiSD ([arXiv 2601.23156](https://arxiv.org/abs/2601.23156)).
- The student-level form of options is the hand-defined skill set of [[SMDP Q-Learning over Hand-Defined Skills]].

## Why it fits Craftax

- Craftax's jobs (gather, craft, descend) are natural options; the question is only whether they can be learned without a list.

## How it works

1. Intra-option Q-learning for the critic over (state, option).
2. Policy gradient for each option's policy; termination gradient pushes termination up where the option's value is below the best option's.

## How to implement here

- A multi-head network (one policy head and one termination head per option) on top of `training/ppo.py`.

## Risks

- Option collapse; many interacting losses.

## Novelty check

- Searched: web "Craftax options hierarchical", Semantic Scholar citations of arXiv 2402.16801 (106 papers), GitHub (the hobby project above, found in the earlier pass of [[Achievement Skill Options]]).
- Partly tried on Craftax (hobby project, no numbers); learned skill hierarchies studied by HiSD.

## How to evaluate

- Not planned.

## References

- Bacon, P.-L., Harb, J., Precup, D. "The Option-Critic Architecture." AAAI 2017. https://arxiv.org/abs/1609.05140
