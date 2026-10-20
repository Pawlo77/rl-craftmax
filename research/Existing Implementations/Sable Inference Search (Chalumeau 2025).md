---
title: "Sable Inference Search (Chalumeau 2025)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Felix Chalumeau
  - Daniel Rajaonarivonivelomanantsoa
  - Ruan de Kock
  - Claude Formanek
  - Sasha Abramowitz
  - Oumayma Mahjoub
  - Wiem Khlifi
  - Simon Du Toit
  - Louay Ben Nessir
  - Refiloe Shabe
  - Noah De Nicola
  - Arnol Fokam
  - Siddarth Singh
  - Ulrich Mbou Sob
  - Arnu Pretorius
year: 2025
venue: "NeurIPS 2025"
paper: "https://arxiv.org/abs/2505.21236"
code: "Google Drive files linked from https://sites.google.com/view/inference-strategies-rl (not downloaded)"
envs:
  - Craftax (env id and observation type not stated)
tags:
  - implementation
  - craftax
  - sequence-model
  - inference-time-search
  - model-free
created: 2026-10-20
---
# Sable Inference Search (Chalumeau 2025)

## TL;DR

- A multi-agent paper on inference-time strategies; App. C.1 adds a single-agent check on Craftax: a Sable policy (retention-based sequence model) trained for 1B steps, then a 30-second inference-time search with stochastic sampling that keeps the best of many attempts ([paper](https://arxiv.org/abs/2505.21236) App. C.1).
- Craftax-1B: 18.8% of max reward zero-shot (95% CI 18.1-19.5) and 25.8% (25.7-26.1) with the 30-second search ([paper](https://arxiv.org/abs/2505.21236) App. C.1, Table 1).
- The 25.8% is best-of-attempts with a resettable simulator at test time, so it is not comparable with single-episode scores; the zero-shot 18.8% is the comparable number, but its protocol differs from the training-episode returns of block A in [[Existing Implementations]].

## Method

- Base policy: Sable (Mahjoub et al. 2025), a retention-based sequence model, trained on Craftax with hyperparameters from that work ([paper](https://arxiv.org/abs/2505.21236) App. C.1, App. D).
- Inference strategy: stochastic policy sampling at temperature 1.0, batched attempts within a 30-second budget, keep the best ([paper](https://arxiv.org/abs/2505.21236) §3.2, App. C.1).
- Requires a simulator that can be reset to the same instance at execution time.

## Training setup and cost

- Training budget: 1B steps ("Craftax-1B", App. C.1); observation type not stated.
- Hardware: one NVIDIA A100-SXM4-80GB for all experiments ([paper](https://arxiv.org/abs/2505.21236) §4).
- Craftax-specific hyperparameters, parameter count and wall-clock: not reported (Tables 8-10 cover the multi-agent tasks only).
- Seeds: not stated for Craftax; the paper says "we always run 128 independent seeds" ([paper](https://arxiv.org/abs/2505.21236) §4).

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, zero-shot (no search) | reward, % of max, mean (95% CI) | 18.8 (18.1, 19.5) | not stated | [paper](https://arxiv.org/abs/2505.21236) App. C.1, Table 1 |
| Craftax-1B, stochastic sampling, 30 s budget | same | 25.8 (25.7, 26.1) | not stated | App. C.1, Table 1 |

- The text claims a 37% gain from search, of the same size as the last 300M training steps ([paper](https://arxiv.org/abs/2505.21236) App. C.1).
- Inconsistency: the text says the base policy "matched" the state of the art at 18.3% (the scoreboard value of [[PPO-GTrXL (Hamon 2024)]]), while Table 1 shows 18.8.

## Problems and limitations

- The authors call the Craftax results "initial (but limited) evidence" ([paper](https://arxiv.org/abs/2505.21236) §5, Limitations and future work).
- Inference strategies need a simulator and a time and compute budget at execution ([paper](https://arxiv.org/abs/2505.21236) §5, Limitations).
- Craftax training details are missing, so the base policy cannot be reproduced from the paper alone.

## Code

- No GitHub repository; the project site https://sites.google.com/view/inference-strategies-rl links "Source Code" and "Model Checkpoints" as Google Drive files (not downloaded; licence unknown).
- Framework: JAX (Mava, QDax, Flax) ([paper](https://arxiv.org/abs/2505.21236) App. E, experimental stack).
