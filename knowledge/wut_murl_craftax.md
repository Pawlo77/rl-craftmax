---
title: "wut_murl_craftax"
type: map
domain: index
tags: [moc, wut_murl_craftax]
created: 2026-01-01
---
# wut_murl_craftax

Reinforcement learning on Craftax

Start here: [[Dashboard]]. Fixed bugs: [[Fixed issues]].

## Plan

- [[2026-10-20-craftax-project]]: the execution plan (phases, methods, metrics, videos, report, release).
- [[Methods]]: the three controls and four methods with their design notes.

## Craftax

- [[Craftax]]: the environment framework (API, observations, achievements, scoring, Apple GPU setup).
- [[Game]]: the full game: stats, resources, crafting, creatures, combat, magic, floors, boss, achievements, complexity.

## Visualisation

- [[Visualisation]]: tested pipeline for YouTube videos and GIFs of gameplay highlights and of the learning process.

## Research

- [[Research]]: main note of the research folder.
- [[Craftax Baselines]]: the official baselines repository, protocol and published scores.
- [[Existing Implementations]]: every known Craftax agent with results and training cost; the table to benchmark against.
- [[Candidate Algorithms]]: simple, not-yet-tried algorithm ideas for Craftax, with the top 5 recommendations.
- [[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]: RL textbook notes and notebooks.

## Packages

| Package | Responsibility |
|---|---|
| [[training]] | PPO, PPO-RNN and PQN-RNN controls, trainer registry, checkpoints, resume state, training records |
| [[visualisation]] | gameplay highlights and learning-process videos, GIFs, ffmpeg cuts |
| [[orchestration]] | fast JAX runtime (compile cache, CPU shards, device roles, textures), run queue, resumable runner, MLflow tracking |
| [[evaluation]] | game tables, per-step events, online episode metrics, reproduction check |

## Concepts

## Decisions

- [[Experiment decisions]]: benchmarks, seeds, tuning, metrics, evaluation, run/resume, report and publication, method selection.
- [[Visualisation decisions]]: video formats, footage, audio, review, metrics, code location.

## Handbook
