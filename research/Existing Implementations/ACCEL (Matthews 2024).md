---
title: "ACCEL (Matthews 2024)"
type: implementation
parent: "[[Existing Implementations]]"
authors:
  - Michael Matthews
  - Michael Beukman
  - Benjamin Ellis
  - Mikayel Samvelyan
  - Matthew Jackson
  - Samuel Coward
  - Jakob Foerster
year: 2024
venue: "ICML 2024 (PMLR 235)"
paper: "https://arxiv.org/abs/2402.16801"
code: "https://github.com/DramaCow/jaxued/blob/main/examples/craftax/craftax_plr.py"
envs:
  - Craftax-Symbolic-v1
tags:
  - implementation
  - craftax
  - ued
  - curriculum
  - ppo
created: 2026-10-20
---
# ACCEL (Matthews 2024)

## TL;DR

- ACCEL (Parker-Holder et al. 2022) on Craftax-1B via JaxUED, with three level-mutation operators: Noise, Swap, Restricted Swap (RSwap).
- RSwap and Noise perform roughly like DR and Robust PLR; unrestricted Swap is the worst UED method ([paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7).
- Exact reward: not reported (curves only); 10 runs on 20 fixed evaluation levels.

## Method

- As [[PLR (Matthews 2024)]], plus randomly mutated buffer levels as new candidates, so level complexity can grow by editing ([paper](https://arxiv.org/abs/2402.16801) §4.2).
- Gradient updates on DR levels but not on mutated levels ([paper](https://arxiv.org/abs/2402.16801) App. G.3); "All ACCEL-methods use exploratory gradient updates" (Table 14).
- Mutation Noise: add uniform noise to the Perlin-noise angle vectors that generate the overworld ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Mutation Swap: swap two tiles, one inside the central 16 x 16 area near the spawn ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Mutation RSwap: swap only compatible tiles (e.g. stone with ores, grass with trees) for more coherent levels ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- JaxUED flags: `--use_accel --accel_mutation {swap,swap_restricted,noise} --num_edits N` ([JaxUED craftax_plr.py](https://github.com/DramaCow/jaxued/blob/main/examples/craftax/craftax_plr.py)).
- Policy: feed-forward MLP; no recurrence.

## Training setup and cost

- PLR and PPO hyperparameters: as [[PLR (Matthews 2024)]] ([paper](https://arxiv.org/abs/2402.16801) Table 14), replay rate 0.5.
- Number of mutations: Swap 10, RSwap 200, Noise 100 (tuned over {10, 100, 200}) ([paper](https://arxiv.org/abs/2402.16801) Table 14, App. G.3).
- JaxUED default `--num_edits 30` matches none of these; pass the Table 14 value.
- Budget: 1B environment steps.
- Hardware, wall-clock, parameters, memory: not reported.

## Results

| Variant | Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|---|
| ACCEL (RSwap) | Craftax-1B, Symbolic | evaluation reward on 20 fixed levels, % of max (226) | not reported as a number; about DR level | 10 | [paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7 |
| ACCEL (Noise) | Craftax-1B, Symbolic | same | not reported as a number; about DR level | 10 | [paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7 |
| ACCEL (Swap) | Craftax-1B, Symbolic | same | not reported as a number; lowest UED curve | 10 | [paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 7 |

- Swap collects diamonds in more than 30% of replay episodes but 7% on DR levels ([paper](https://arxiv.org/abs/2402.16801) §4.5, Figs. 32-33).
- RSwap reaches the Gnomish Mines in 1% of replay episodes, almost never on DR levels ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Both swap variants earn "significantly higher returns during replay", which may mean the generated levels are easier ([paper](https://arxiv.org/abs/2402.16801) §4.5, Fig. 8).

- Third-party rerun (2026-10-20 pass): [[PACE (Yuan 2026)]] reruns ACCEL (mutation variant not stated) with a JaxUED PPO student on Craftax-Symbolic for 1.07B steps and evaluates the final checkpoints on 20 held-out levels: mean episodic reward 22.661 (sd across levels 4.005), i.e. 10.0% of 226 over 10 seeds; IQM of min-max normalised reward 0.364 [0.283, 0.458] ([PACE paper](https://arxiv.org/abs/2605.01358) Tables 4, 5). Same kind of protocol as the table above but different levels, hyperparameters and budget; in that rerun DR ranks above PLR, unlike the Craftax paper.

## Problems and limitations

- Unrestricted Swap produces levels visually very different from the generator's ([paper](https://arxiv.org/abs/2402.16801) App. G.2).
- Replay-level gains do not show up on DR levels (diamonds 30% vs 7% above), and evaluation is on normally generated levels.
- Mutations act only on the overworld (floor 0) ([paper](https://arxiv.org/abs/2402.16801) §4.5).
- Evaluation metric differs from the training-episode return of the scoreboard baselines.

## Code

- Repository: https://github.com/DramaCow/jaxued, `examples/craftax/craftax_plr.py`, `examples/craftax/mutators.py`.
- Framework: JAX. Licence: Apache-2.0 (declared in `pyproject.toml`).
- Citation: Matthews et al., ICML 2024, arXiv:2402.16801; method: Parker-Holder et al., "Evolving Curricula with Regret-Based Environment Design", ICML 2022.
