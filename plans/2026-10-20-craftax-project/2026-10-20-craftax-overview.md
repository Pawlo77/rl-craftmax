---
title: "Craftax project plan overview"
type: plan
domain: training
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, overview, summary]
---
# Craftax project plan overview

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a one-page summary of the whole project plan for anyone joining the team; the executable steps live in [[2026-10-20-craftax-project]] and its linked files.

**Architecture:** seven agents (three controls, four methods) trained on the same game with the same code, compared on the same metrics, then explained, filmed, written up and published.

**Tech Stack:** JAX, craftax, MLflow, ffmpeg, LaTeX, Hugging Face.

---

## The idea in five sentences

- Craftax is a fast, open-ended survival and crafting game with 67 achievements across 9 floors ([[Craftax]], [[Game]]); no published agent gets far past the second floor.
- We train three published agents as controls (PPO, PPO-RNN, PQN-RNN) and four textbook-level methods that each attack a different difficulty of the game ([[Methods]]).
- Every agent runs at the two scoreboard budgets, 1M and 1B environment steps, so our numbers sit next to [[Existing Implementations]].
- Beyond the score we measure learning stability, hardware use and twelve in-game behaviours (who unlocks what first, in which order, how deep, how they die, how they explore, fight and survive), and we distil each final policy into a decision tree to see how complex its rules are.
- Results become GIFs and YouTube videos, a 4-page report, and a public Hugging Face Space and dataset.

## The four methods in plain words

| Method | One sentence | Difficulty it attacks |
|---|---|---|
| Count-Based Progress Bonus | Reward the agent for reaching combinations of floor, tools and items it has rarely reached | exploration that chases new scenery instead of progress |
| Self-Imitation Learning | Replay the agent's own lucky successes so it learns them faster | rare one-time achievements |
| SMDP Q-Learning over Hand-Defined Skills | A table chooses the next skill ("get wood", "make a pickaxe", "descend"); a network learns to execute the skills | the long crafting chain |
| Dyna-Q over a Skill-Level Progress Model | The same skills, plus a learned table of what each skill does, used to plan ahead | slow value propagation along the chain |

## How the work flows

```mermaid
flowchart LR
  I["Infrastructure: resumable MLflow runner"] --> C["Controls reproduced at 1M"]
  C --> T["Methods tuned at 1M (coarse, then fine)"]
  T --> F1["Finals: 10 seeds at 1M"]
  T --> F2["Finals: 10 seeds at 1B"]
  F1 --> X["Best two methods combined"]
  F2 --> X
  F1 --> A["Analysis: hard scenarios, obstacles, comparison, decision trees"]
  F2 --> A
  X --> A
  A --> V["GIFs and videos"]
  A --> R["4-page report"]
  V --> H["Hugging Face Space and dataset"]
  R --> H
```

## Phases and where they run

| Phase | What happens | Runs | Laptop estimate |
|---|---|---|---|
| 0 | Infrastructure, metrics, controls reproduced | smoke + control tuning | part of the 112 h below |
| 1 | Each method implemented, tested, tuned at 1M | 494 coarse + 138 fine | about 112 h for phases 0-2 at 1M |
| 2 | Finals at 1M and 1B, 10 seeds each | 150 + 90 | 1B: about 128 GPU hours on CUDA, or about 11,300 laptop hours |
| 3 | Best two methods combined | 20 | included above |
| 4 | Analysis | offline | small |
| 5-7 | Videos, report, Hugging Face release | offline | small |

- Full numbers: [[2026-10-20-craftax-compute-budget]].

## How to run it

- `make queue PHASE=<phase>` starts or continues a phase; Ctrl-C stops it safely; the next call resumes.
- `make status PHASE=<phase>` shows what is done; `make mlflow` shows the results.
- Laptop and cluster run the same code and run names; their stores merge into one `mlruns.db` ([[2026-10-20-craftax-infrastructure]]).

## What we will be able to say at the end

- Which family learns Craftax best at 1M and at 1B: policy gradient or value-based, flat or hierarchical, model-free or planning ([[2026-10-20-craftax-comparison-and-results]]).
- Which scenarios were hard for each agent and how it eventually solved them ([[2026-10-20-craftax-training-analysis]]).
- How complex each agent's decision rules are ([[2026-10-20-craftax-interpretation-and-ruleset]]).
- How each agent plays, in numbers and on video ([[2026-10-20-craftax-visualisation-production]]).

## Key decisions

- Benchmarks, seeds, tuning, metrics, run and release: [[Experiment decisions]].
- Videos and figures: [[Visualisation decisions]].
- Fastest machine setup: [[Craftax runtime setup]].
