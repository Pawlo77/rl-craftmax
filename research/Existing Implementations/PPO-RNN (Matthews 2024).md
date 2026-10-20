---
title: "PPO-RNN (Matthews 2024)"
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
code: "https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py"
envs:
  - Craftax-Symbolic-v1
  - Craftax-Classic-Symbolic-v1
tags:
  - implementation
  - craftax
  - ppo
  - recurrent
  - model-free
created: 2026-10-20
---
# PPO-RNN (Matthews 2024)

## TL;DR

- PPO with a GRU; the strongest baseline of the Craftax paper.
- Craftax-1B (Symbolic): 15.3% of max reward; Craftax-1M (Symbolic): 2.3% ([README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard)); 10 seeds ([paper](https://arxiv.org/abs/2402.16801) Fig. 3, Fig. 6).
- Craftax-Classic (Symbolic) at 1B steps: about 90% of max reward ([paper](https://arxiv.org/abs/2402.16801) abstract, §1, App. F.3).
- How to run it: [[Craftax Baselines]].

## Method

- Network (`ppo_rnn.py` `ActorCriticRNN`): Dense(512) + ReLU embedding, then a GRU with hidden size equal to the layer size, then actor and critic heads of 2 x Dense(512) + ReLU each.
- Hidden state is reset to zeros where an episode ends (`ScannedRNN`).
- Same PPO loss and hyperparameters as [[PPO (Matthews 2024)]]; GRU hidden size equal to the MLP layer size ([paper](https://arxiv.org/abs/2402.16801) App. E.1).
- No intrinsic reward.

## Training setup and cost

- Hyperparameters 1B: those of PPO, [paper](https://arxiv.org/abs/2402.16801) Table 7 (1024 workers, 64 steps, 4 epochs, 8 minibatches, lr 0.0002, γ 0.99, GAE λ 0.8, clip 0.2, entropy 0.01, width 512, annealed LR).
- Hyperparameters 1M: not reported separately for PPO-RNN (PPO 1M values are in [paper](https://arxiv.org/abs/2402.16801) Table 11).
- `ppo_rnn.py` defaults: `--total_timesteps 1e9`, `--num_envs 1024`, `--lr 2e-4`, `--max_grad_norm 1.0`, optimistic resets on with ratio 16.
- Hardware: single RTX 4090 used for the paper's speed tests ([paper](https://arxiv.org/abs/2402.16801) App. B).
- Wall-clock: Craftax-Classic "solved (to within 90% of maximum return)" in 51 minutes ([paper](https://arxiv.org/abs/2402.16801) §1, pointing to App. F.3); Craftax-1B wall-clock for PPO-RNN: not reported.
- Adding the RNN "only slightly reduced steps per second" ([Craftax issue #34](https://github.com/MichaelTMatthews/Craftax/issues/34)).
- Parameters: not reported. GPU memory: not reported.

## Results

| Benchmark | Metric | Value | Seeds | Source |
|---|---|---|---|---|
| Craftax-1B, Symbolic | reward, % of max (226) | 15.3 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [paper](https://arxiv.org/abs/2402.16801) Fig. 3; also quoted in [PQN paper](https://arxiv.org/abs/2407.04811) §5.3 |
| Craftax-1M, Symbolic | reward, % of max (226) | 2.3 | 10 | [README scoreboard](https://github.com/MichaelTMatthews/Craftax#-scoreboard); [paper](https://arxiv.org/abs/2402.16801) Fig. 6 |
| Craftax-1M, Symbolic (third-party rerun) | reward, % of max (226) | 2.13 | 10 | [Simulus paper](https://arxiv.org/abs/2502.11537) Fig. 1; per-run data in [Simulus repo](https://github.com/leor-c/Simulus/blob/main/results/data/craftax/craftax_baselines.json) |
| Craftax 10B steps, Symbolic | reward, % of max (226) | not reported as a number ("barely improves") | 4 | [paper](https://arxiv.org/abs/2402.16801) App. F.2, Fig. 22 |
| Craftax-Classic 1B, Symbolic | reward, % of max (22) | about 90 | 10 | [paper](https://arxiv.org/abs/2402.16801) abstract, App. F.3, Fig. 23 |

- Standard deviations: not reported; figures show ±1 standard error.
- Metric in code: mean return of episodes that finish inside each training batch (`ppo_rnn.py`).
- Memory helps push into the Intermediate achievement category ([paper](https://arxiv.org/abs/2402.16801) §4.3).
- Enters the dungeon (floor 1) significantly more often than the other baselines, which drives its higher reward ([paper](https://arxiv.org/abs/2402.16801) §4.3, Fig. 5).
- Craftax-Classic: all achievements except `COLLECT_DIAMOND` and `EAT_PLANT` are reliably achieved ([paper](https://arxiv.org/abs/2402.16801) App. F.3).

Third-party reruns found in the 2026-10-20 pass (protocols differ; details in the linked notes):

| Source | Benchmark | Metric | Value | Seeds |
|---|---|---|---|---|
| [[Feasibility Classification (Zabounidis 2026)]] (unmasked PPO, GRU 512) | Craftax 1B, Symbolic; Craftax-Classic 1B, Symbolic | raw return, 10,240 stochastic evaluation episodes per seed, mean ± sd | 36.7 ± 0.1 (16.2%); Classic 19.8 ± 0.0 (90.0%) | 4 |
| Singh, Ghasemi, Crowley 2026, [arXiv:2608.14642](https://arxiv.org/abs/2608.14642) §6.1, Table 2 (unconstrained PPO-GRU, hidden 512) | Craftax-Symbolic-v1, 1e9 steps | raw return, 64 environments x 4096 steps of evaluation (about 2,400 episodes across seeds), ± type not stated | 37.05 ± 0.78 (16.4%); about 3 h per run on one H100 | 3 |
| Park, Cho, Ahn 2025 (CrafterDojo), [arXiv:2508.13530](https://arxiv.org/abs/2508.13530) Table 1, "Expert Behavior Generator" | Craftax-Classic-Symbolic, 10B steps, Crafter reward plus a −10 survival penalty after all achievements | return % of max / Crafter score %, 100 episodes | 98.4 ± 0.04 / 97.5 (Table 4 of the same paper lists 1e9 total timesteps) | not stated |
| [[Auto-Craftax PPO-GRU (Creus Castanyer 2026)]] (PyTorch PPO-LSTM port) | Craftax-1B, Symbolic | training-episode return (raw), mean ± sd | 33.88 ± 0.50 (15.0%) | 3 |
| [[Rule Learner Planner (dargilboa 2026)]] (64 environments) | Craftax-Classic, Symbolic, 300k / 1M | Crafter score of own training episodes, % | 3.4 / 5.9 | 3 |
| [[SCALAR (Zabounidis 2026)]] (augmented observation) | Craftax-Classic and Craftax 1B | target-achievement success, % | e.g. Collect Diamond 40.9, Enter Dungeon 89.7, Enter Gnomish Mines 0.0 | 5 |

- The 2.3% Craftax-1M value is also quoted, not rerun, in [[Multitask Preplay (Carvalho 2025)]].

## Problems and limitations

- No appreciable progress into the Gnomish Mines at 1B or 10B steps ([paper](https://arxiv.org/abs/2402.16801) §4.3, App. F.2).
- The late rise of the reward curve is attributed to the learning-rate decay schedule, not continued learning ([paper](https://arxiv.org/abs/2402.16801) §4.3, App. F.2).
- Success on `DEFEAT_ZOMBIE` falls over training, lowest for PPO-RNN: it trades overworld achievements for underground ones ([paper](https://arxiv.org/abs/2402.16801) §4.3).
- `EAT_PLANT` is ignored entirely ([paper](https://arxiv.org/abs/2402.16801) §4.3).
- No PPO-RNN run with pixels was made ([Craftax issue #34](https://github.com/MichaelTMatthews/Craftax/issues/34)).
- RNN policies cannot be loaded by `view_ppo_agent.py` (`KeyError: 'Dense_7'`) ([Baselines issue #2](https://github.com/MichaelTMatthews/Craftax_Baselines/issues/2)).
- The paper's abstract attributes the 90% Craftax-Classic result to PPO, App. F.3 to PPO-RNN; which one produced the 51-minute run is ambiguous.
- Shared wrapper issues are listed in [[PPO (Matthews 2024)]].

## Code

- Repository: https://github.com/MichaelTMatthews/Craftax_Baselines (`python ppo_rnn.py`).
- Framework: JAX, Flax (`nn.GRUCell`), optax, distrax.
- Licence: MIT.
- Citation: Matthews et al., "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning", ICML 2024, arXiv:2402.16801.
