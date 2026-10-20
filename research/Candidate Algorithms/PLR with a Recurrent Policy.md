---
title: "PLR with a Recurrent Policy"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - curriculum
  - ued
  - recurrent
origin: "Jiang, Grefenstette, Rocktäschel. Prioritized Level Replay. ICML 2021. https://arxiv.org/abs/2010.03934"
suggested-by:
  - "[[PLR (Matthews 2024)]]"
  - "[[Domain Randomisation (Matthews 2024)]]"
created: 2026-10-20
---
# PLR with a Recurrent Policy

## Idea in plain words

- Prioritized Level Replay (PLR) keeps a buffer of worlds ("levels") and trains more often on those where the agent still has much to learn.
- On Craftax, PLR was the best curriculum method, but it was run with a feed-forward policy (no memory).
- Memory is the single biggest help on Craftax: the GRU agent beats the feed-forward one.
- Combine both: PLR level selection plus a GRU policy, and test on held-out worlds.

## Why it fits Craftax

- A Craftax level is the full seed-generated world, with all 9 floors fixed at reset ([[Craftax world generation]]). Some worlds have the ladder, water or diamonds close by; others do not. Choosing worlds is a real lever.
- PLR significantly beat domain randomisation, the only curriculum method that did ([[PLR (Matthews 2024)]]).
- Memory pushes agents into the dungeon ([[PPO-RNN (Matthews 2024)]]); the UED study never combined the two.
- The UED results were measured on normally generated levels, which the authors say favours plain randomisation ([[Domain Randomisation (Matthews 2024)]]).

## How it works

Builds on the actor-critic ([[11 Policy approximation (actor-critic)]]); the level score is a prioritised sampling rule like those in [[02 Multi-armed bandits]].

1. Keep a buffer of $M$ level seeds with scores (positive value loss, an estimate of how much can still be learned there).
2. Each round: with probability $q$ sample a new random level; else sample a buffer level by rank of score.
3. Run the GRU policy on the chosen levels for several short rollouts (as in the Craftax PLR setup), reset the hidden state at episode start.
4. Update the score from the whole trajectory; insert new high-score levels into the buffer.
5. Evaluate on a held-out set of seeds never used in training.

## How to implement here

- Base: JaxUED `examples/craftax/craftax_plr.py` ([[PLR (Matthews 2024)]]), replacing its MLP actor-critic with the `ActorCriticRNN` of `ppo_rnn.py` ([[Craftax Baselines]]).
- Main work: carry the GRU hidden state through JaxUED's rollout and scoring code (about 150-250 lines changed).
- Extra hyperparameters: buffer size, replay probability, score temperature, staleness weight (JaxUED defaults), inner rollouts per level.

## Cost and risk

- Compute: like PPO-RNN plus scoring passes; the UED runs used 1B steps, so a GPU is needed for comparable numbers. Small tests (Craftax-1M scale) fit a Mac.
- Student level: yes, but medium effort; most of the work is plumbing in an existing codebase.
- Risk: curriculum-induced covariate shift: replay levels differ from normal levels ([[PLR (Matthews 2024)]]).
- Risk: recurrent hidden states make the per-level rollout bookkeeping error-prone.

## Novelty check

- Searched: web "Craftax PLR recurrent", JaxUED repository README, Semantic Scholar citations of arXiv 2402.16801.
- JaxUED still lists "Support for an RNN policy" as coming soon for Craftax.
- Since then on Craftax: rule-based achievement curricula with a GTrXL policy (Anchor-and-Harvest, [arXiv 2610.15328](https://arxiv.org/abs/2610.15328)) and LLM-written curricula (DiCode, ICML 2026, [code](https://github.com/konstantinosmitsides/dreaming-in-code)); neither is PLR with a recurrent policy.
- Not found: PLR with a recurrent policy on Craftax.

## Suggested by existing work

- JaxUED README, Craftax section, "coming soon" list: "Support for an RNN policy" ([[PLR (Matthews 2024)]]).
- Craftax paper §4.5: DR was competitive "partly due to Craftax evaluating on in distribution levels" ([[Domain Randomisation (Matthews 2024)]]).

## How to evaluate

- On 20 normal levels, as in block B of the [[Existing Implementations]] table, and on a held-out seed set.
- Compare with PPO-RNN with plain randomisation at the same budget, and with feed-forward PLR.
- Success rates of `ENTER_DUNGEON` and `ENTER_GNOMISH_MINES`.

## References

- Jiang, M., Grefenstette, E., Rocktäschel, T. "Prioritized Level Replay." ICML 2021. https://arxiv.org/abs/2010.03934
- JaxUED repository. https://github.com/DramaCow/jaxued
- Matthews, M. et al. "Craftax." ICML 2024. https://arxiv.org/abs/2402.16801
