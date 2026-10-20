---
title: "Research"
type: index
parent: "[[wut_murl_craftax]]"
tags: [index, research]
created: 2026-10-20
---
# Research

Literature and prior work behind the project: the RL textbook, the official Craftax baselines, every published Craftax agent and the algorithm ideas we may try. Framework and game facts live in [[Craftax]] and [[Game]].

| Folder | Main note | What it holds |
|---|---|---|
| `Reinforcement Learning An Introduction/` | [[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]] | Chapter notes and runnable notebooks for the Sutton & Barto textbook (2nd ed draft): the RL foundations every other folder builds on |
| `Craftax Baselines/` | [[Craftax Baselines]] | The official baselines repository: how to run it, algorithms and hyperparameters, benchmark protocol, published scores, compute, reproduction pitfalls |
| `Existing Implementations/` | [[Existing Implementations]] | One note per published Craftax or Craftax-Classic agent (method, training cost, results, problems) and the comparison table we benchmark against |
| `Candidate Algorithms/` | [[Candidate Algorithms]] | Simple algorithm ideas not yet tried on Craftax, ideas from existing works' future-work sections, and the top 5 recommendations |

## Reading order

1. [[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]] for the methods (TD, Q-learning, actor-critic, function approximation).
2. [[Craftax Baselines]] for the benchmark protocol and the scores to beat.
3. [[Existing Implementations]] for what has been tried and how well it worked.
4. [[Candidate Algorithms]] for what to try next.
