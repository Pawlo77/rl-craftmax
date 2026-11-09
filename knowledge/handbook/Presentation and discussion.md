---
title: "Presentation and discussion"
type: handbook
domain: vault
parent: "[[wut_murl_craftax]]"
tags: [handbook, course, presentation]
created: 2026-11-08
updated: 2026-11-08
---
# Presentation and discussion

25 January 2027, 14 points ([[Course project requirements]]): 15 minutes plus 7 of questions, as a talk or as the YouTube video shown in the team's slot; the team attends in person either way and joins the discussion. Six points are for the clarity of the explanation, six for the understanding shown in the questions, two for materials and time.

## Talk outline (15 minutes)

| Min | Part | Material |
|---|---|---|
| 1.5 | Craftax and the question | the first scenes of the 12-minute video, [[Craftax scoring and evaluation]] |
| 3 | The three baselines and the four methods, one idea each | [[Experiment decisions]], the method notes under `knowledge/methods/` |
| 1.5 | The protocol: budgets, seeds, score, statistics | the report's Section 4 |
| 2 | Results at 1M | T1, F1, F2 of [[Figures 1M]] |
| 2 | Results at 1B and the 1M vs 1B comparison | [[Figures 1B]], [[Figures SCALE]] |
| 2 | What the agents do and why they fail | [[Decision-tree interpretation]], the per-method training analysis notes |
| 1.5 | Conclusion, limits, release | the report's Section 7, the Space and the dataset |
| 1.5 | Spare | |

The 12-minute video (`outputs/visualisation/final/long_land.mp4`) covers about 12 minutes of this; showing it leaves 3 minutes for a live introduction and the limits. The videos' voice is synthetic: say so.

## Discussion: everyone explains their part

Be able to answer, with the note that holds the evidence:

- **What is the score and why a training score?** The share of the 226 points over the last 2% of the budget; the fixed-seed evaluation is reported beside it ([[Craftax scoring and evaluation]]).
- **Why does count bonus help nothing?** Its gain is inside the seed noise at 1M and vanishes at 1B ([[Final results]]); the SimHash variant overflows its table at 1B ([[Experiment decisions]], 2026-11-08).
- **Why do the skill methods lose?** The skills are hand-written and the high-level table sees few visits; their obstacle detectors fire at 1B ([[Dyna-Q training analysis]], [[SMDP skills training analysis]]).
- **Why is PPO-RNN above the published score?** Not traced; see [[I1 - PPO-RNN 1B above the reproduction band]] for what is ruled out.
- **Why only three seeds at 1B?** One 1B run is 3.5 to 5.6 GPU hours; the 1B results are indicative and the tables give intervals only ([[2026-10-20-craftax-compute-budget]]).
- **What do the decision trees show?** They agree with the networks on more states at 1B but keep only a small part of the score ([[Decision-tree interpretation]]).
- **What was done with AI help?** The text, code, figures and narration were written with Claude Code under the author's direction; the disclosure sits in the report.
