---
title: "Language-Conditioned Policy"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - language
  - instructions
  - goal-conditioned
origin: "Matthews et al. Craftax, App. D.4 (text renderer). ICML 2024. https://arxiv.org/abs/2402.16801; CrafText Benchmark. 2025. https://arxiv.org/abs/2505.11962"
suggested-by:
  - "[[PPO (Matthews 2024)]]"
created: 2026-10-20
updated: 2026-10-20
---
# Language-Conditioned Policy

## Idea in plain words

- Craftax can describe the agent's view and inventory as text ("troll on torch on path", "DIAMOND: 2").
- The repository also has a text tutorial of the game.
- Idea: give the policy a short instruction ("make a stone pickaxe") and reward it for following it.
- A student-level version uses a fixed list of instructions with simple word embeddings, no large language model.

## Why it fits Craftax

- The text renderer exists in the code (`render_craftax_text`, [[Craftax API and wrappers]]).
- The tech tree gives natural instructions: one per recipe, per floor, per creature ([[Craftax crafting and placing]], [[Craftax floors and progression]]).
- Instructions let you train skills separately and then chain them, which targets the long recipe chains.

## How it works

Builds on function approximation with extra input features ([[09a Function approximation and linear methods]]) and the actor-critic ([[11 Policy approximation (actor-critic)]]).

1. Make a list of instructions, each with a success test on the state (for example "inventory stone pickaxe > 0").
2. At episode start (or after each success), sample an instruction.
3. Encode it: bag of words or a small learned embedding of the instruction id.
4. Feed it to the policy with the observation; reward +1 on success, plus the native reward if desired.
5. Evaluate on the native task by issuing instructions in tech-tree order.

## How to implement here

- Base: `ppo_rnn.py`; an instruction wrapper with success tests, about 150-250 lines; an embedding layer, about 20 lines.
- Extra hyperparameters: instruction list, sampling rule, mix of instruction reward and native reward.
- Student level: partly. A fixed instruction list is feasible; free-form language with a language model is beyond scope (the project excludes large foundation models as the core method).

## Risks

- Risk: this is close to goal-conditioned RL, which has been done on Craftax (see below); a new angle is needed.
- Risk: the native score needs a fixed instruction schedule, which is hand design.

## Novelty check

- Searched: Semantic Scholar citations of arXiv 2402.16801, web "Craftax language-conditioned".
- Tried since the suggestion:
  - CrafText: an instruction-following benchmark on Craftax ([arXiv 2505.11962](https://arxiv.org/abs/2505.11962)).
  - LLM instruction relabelling on Craftax ([arXiv 2506.20061](https://arxiv.org/abs/2506.20061)).
  - Goal-conditioned Craftax with 512 goals, LEO ([arXiv 2605.23551](https://arxiv.org/abs/2605.23551)).
  - LLM-informed curricula over 256 textual goals in Craftax ([arXiv 2610.04604](https://arxiv.org/abs/2610.04604)).
- Status: already explored; low novelty for a student project.

## Suggested by existing work

- Craftax paper App. D.4: with the text tutorial, the text renderer "could be use for language-conditioned learning" ([[PPO (Matthews 2024)]]).

## How to evaluate

- Instruction success rate per instruction.
- Native reward % of max with a tech-tree instruction schedule on Craftax-1M and Craftax-1B, against blocks D and A of the [[Existing Implementations]] table.

## References

- Matthews, M. et al. "Craftax." ICML 2024, App. D.4. https://arxiv.org/abs/2402.16801
- "CrafText Benchmark." 2025. https://arxiv.org/abs/2505.11962
- "Learning Instruction-Following Policies through Open-Ended Instruction Relabeling with Large Language Models." 2025. https://arxiv.org/abs/2506.20061
- "Goal-Conditioned Agents that Learn Everything All at Once." 2026. https://arxiv.org/abs/2605.23551
