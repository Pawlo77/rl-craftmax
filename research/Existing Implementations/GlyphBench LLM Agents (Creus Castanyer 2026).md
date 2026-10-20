---
title: "GlyphBench LLM Agents (Creus Castanyer 2026)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Roger Creus Castanyer
  - Marc-Alexandre Côté
  - Matthew James Sargent
  - Augustine N. Mavor-Parker
  - Glen Berseth
  - Pablo Samuel Castro
year: 2026
venue: "arXiv preprint (v1, Sep 2026)"
paper: "https://arxiv.org/abs/2609.34214"
code: "https://github.com/VmaxAI/glyphbench"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - llm-agent
  - zero-shot
  - benchmark
created: 2026-10-20
---
# GlyphBench LLM Agents (Creus Castanyer 2026)

## TL;DR

- Zero-shot language-model agents (no RL training) playing full Craftax through a text "glyph" interface: a Unicode grid with a legend and a HUD, one native action per turn ([paper](https://arxiv.org/abs/2609.34214) §3.1, App. C).
- Best standard result: 11.4% of max reward for the Sol model with the Guided Memory harness on glyphs at high reasoning effort, against 3.7% on pixels and 0.3% on Craftax's native text; 15.8% at maximum effort; 5 episodes per condition ([paper](https://arxiv.org/abs/2609.34214) §5.1, Fig. 4).
- One exploratory GPT-6-Astra episode reached 51.4% ([paper](https://arxiv.org/abs/2609.34214) Fig. 4, App. C.2).
- No step budget in the RL sense, 5 episodes, LLM agents: not comparable with any RL block of [[Existing Implementations]].

## Method

- Observation formats compared: pixels, Craftax's native text renderer, and glyphs (2D Unicode grid plus legend plus HUD) ([paper](https://arxiv.org/abs/2609.34214) App. C.1).
- Harnesses: Basic (instructions plus history) and Guided Memory (extended guide, persistent memory of plan, landmarks and recent actions, floor-specific guidance, separate memory-update calls every 5 steps or on events) ([paper](https://arxiv.org/abs/2609.34214) App. C.2).
- Models: gpt-5.6-luna, gpt-5.6-terra, gpt-5.6-sol, and one GPT-6-Astra episode; reasoning effort varied ([paper](https://arxiv.org/abs/2609.34214) §4, §5.1).
- The paper's RL part (Qwen3.5-4B post-training) uses bounded Craftax-like GlyphBench tasks, not standard Craftax, and is not covered here ([paper](https://arxiv.org/abs/2609.34214) §5.3).

## Training setup and cost

- No training for the Craftax results (zero-shot evaluation).
- Environment: full Craftax with native reward through the wrapper `glyphbench/craftaxfull-v0`, which drives a pinned Craftax fork with `CraftaxSymbolicEnvNoAutoReset` (repo `src/glyphbench/envs/craftaxfull/fork.py`).
- Five shared environment seeds per condition ([paper](https://arxiv.org/abs/2609.34214) §4).
- Episode cap: not stated in the paper; repository default `max_turns` 10000.
- Tokens per Craftax episode: not given; across all standard tasks, mean total tokens per episode fall from 2.15M to 596k across models (Fig. 3).
- Hardware, wall-clock and API cost: not reported.

## Results

| Model, harness, observation, effort | Metric | Value | Episodes | Source |
|---|---|---|---|---|
| Sol, Guided Memory, glyphs, high | reward, % of max (226) | 11.4 | 5 | [paper](https://arxiv.org/abs/2609.34214) §5.1, Fig. 4a |
| Sol, Guided Memory, pixels, high | same | 3.7 | 5 | §5.1, Fig. 4a |
| Sol, Guided Memory, native text, high | same | 0.3 | 5 | §5.1, Fig. 4a |
| Sol, Basic, glyphs, effort sweep | same | levels off "near 11" | 5 | Fig. 4c text |
| Sol, Guided Memory, glyphs, max effort | same | 15.8 | 5 | Fig. 4c text |
| GPT-6-Astra, Guided Memory, glyphs, max effort | same | 51.4 (after 2,304 steps) | 1 (exploratory) | Fig. 4, App. C.2 |

- Values are means of 5 episodes with 95% t-intervals shown in the figure; the intervals and the Luna and Terra values are figure only ([paper](https://arxiv.org/abs/2609.34214) Fig. 4).

## Problems and limitations

- Glyph abstraction and bounded horizons; harness comparisons bundle instructions, memory and extra calls, so ablations are needed; model comparisons do not isolate size or architecture ([paper](https://arxiv.org/abs/2609.34214) App. L).
- 5 episodes per condition and one Astra episode: small samples.
- Proprietary models, so results cannot be reproduced exactly.

## Code

- Repository: https://github.com/VmaxAI/glyphbench (Craftax runs: `eval/run_craftax.sh` with task `glyphbench/craftaxfull-v0`); commit read in this pass: `6cc45a4` (2026-09-14); no issues.
- Framework: Python (NumPy ports of tasks); the full-Craftax adapter uses JAX Craftax.
- Licence: MIT.
- Same first author as [[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]].
