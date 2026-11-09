---
title: "Checkpoint meeting"
type: handbook
domain: vault
parent: "[[wut_murl_craftax]]"
tags: [handbook, course, checkpoint]
created: 2026-11-08
updated: 2026-11-08
---
# Checkpoint meeting

The meeting with the instructor falls between 23 and 27 November 2026 (10 points, criteria in [[Course project requirements]]). By then the experiments are finished, so the checkpoint shows the whole study, not a first result. One block per scored item:

| Points | Show | Where it is |
|---|---|---|
| 2 | Problem, objective and an accessible repository | Craftax-Symbolic at 1M and 1B steps, three baselines, four methods and their combination under one protocol ([[2026-10-20-craftax-project]]); the repository is public: https://github.com/Pawlo77/rl-craftmax |
| 4 | A working minimal experiment, from the environment to a measured score | `make install`, then `make train-ppo ARGS="--env Craftax-Classic-Symbolic-v1 --steps 1e6"` for a short PPO run, or open `mlruns.db` through `make mlflow`; the full runs are `make queue PHASE=final-1m` and the phases of [[2026-10-20-craftax-compute-budget]] |
| 2 | An initial result and its limits | [[Final results]] (T1 and the figures of [[Figures 1M]] and [[Figures 1B]]); limits: three seeds at 1B, no tuning at 1B, two GPU classes at 1B, laptop and cluster stacks (checked, no effect), the PPO-RNN gap above the published score ([[I1 - PPO-RNN 1B above the reproduction band]]) |
| 2 | A plan for further experiments and the division of work | below |

## Plan from the checkpoint on

- The 1B ablations (`ablate-1b`, running on the cluster) fill the last empty figure; the report and the Hugging Face release are updated with them ([[2026-10-20-craftax-report]], [[2026-10-20-craftax-huggingface-release]]).
- The three videos are rendered and need an upload ([[2026-10-20-craftax-visualisation-production]]).
- Candidate extensions, each a new phase file: tune the methods at 1B, run more 1B seeds so close methods can be ordered, replace the hand-chosen progress summary by a learned one, add the RND and E3B baselines of the Craftax paper.
- Division of work: the repository and the report list one author, Paweł Pozorski; add a line here if the course team has more members and who owns which part.

## Questions to expect

| Question | Answer, with its note |
|---|---|
| Why Craftax at two budgets? | Both scoreboard budgets are comparable with published baselines ([[Experiment decisions]], 2026-10-20) |
| Is the 1M result significant? | No method significantly beats its control; the PQN-RNN family and the skill methods are significantly below the PPO family ([[Final results]]) |
| Does 1M predict 1B? | Only loosely: the rank correlation is in `## 1M vs 1B` of [[Final results]] |
| Could the software stack explain it? | The bridge runs say no ([[Experiment decisions]], 2026-11-06) |
