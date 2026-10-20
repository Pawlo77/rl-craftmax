---
title: "World-Model Tokenizer Improvements"
type: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - model-based
  - world-model
  - tokenizer
origin: "Cohen et al. Simulus. 2025. https://arxiv.org/abs/2502.11537; Dedieu et al. Improving Transformer World Models for Data-Efficient RL. 2025. https://arxiv.org/abs/2502.01591"
suggested-by:
  - "[[Simulus (Cohen 2025)]]"
  - "[[Efficient MBRL (Dedieu 2025)]]"
created: 2026-10-20
updated: 2026-10-20
---
# World-Model Tokenizer Improvements

## Idea in plain words

- Token-based world models turn each observation into a sequence of discrete symbols (tokens) and predict the next tokens.
- How observations become tokens matters a lot: Simulus turns Craftax's 48 continuous inventory features into many tokens, which makes sequences long and slow.
- Suggested fixes: shorter encodings for continuous inputs (Simulus), and tokens from large pretrained image models instead of hand-made patch codes (Efficient MBRL).
- Efficient MBRL also suggests a policy that reads the world model's tokens directly, so the model never has to draw pixels.

## Why it fits Craftax

- The symbolic observation mixes a tile grid (naturally discrete) with inventory and stats (continuous scaled numbers) ([[Craftax observations and actions]]).
- Simulus uses 444 tokens per Craftax observation ([[Simulus (Cohen 2025)]]).
- The Efficient MBRL tokenizer relies on Craftax's known 7 x 7 tile size; other patch sizes hurt badly ([[Efficient MBRL (Dedieu 2025)]]).

## How it works

Builds on model-based planning ([[08a Models, planning, Dyna]]).

1. Inventory: encode small integer counts directly as tokens (Craftax counts are integers up to 99), not as quantised floats after symlog.
2. Map: one token per tile from the block, item and creature ids (already discrete in the symbolic observation).
3. Train the world model on these shorter sequences; train the controller in imagination as before.
4. Pretrained-feature variant (Efficient MBRL): replace the patch codebook with features of a large pretrained image model.

## How to implement here

- Requires a full token-based world-model agent (Simulus or a reimplementation of Efficient MBRL); the tokenizer change itself is small (about 100 lines), but the surrounding system is large.
- Extra hyperparameters: token vocabulary per field, sequence layout.
- Student level: partly. The per-field integer tokenizer is about 100 lines inside the open Simulus code ([[Simulus (Cohen 2025)]]), but learning and debugging that large world-model agent is most of the work. Rebuilding Efficient MBRL is research-grade: its code is not released ([[Efficient MBRL (Dedieu 2025)]]). The pretrained-image-model variant uses a large foundation model as the core, which this project excludes.

## Risks

- Risk: gains are inside the world model and hard to isolate; change only the tokenizer and compare with Simulus over 5 seeds.
- Risk: the repository default is now Simulus-A, untested on Craftax; reproducing the paper's numbers needs the original Simulus configuration ([[Simulus (Cohen 2025)]]).
- Runtime: Simulus reports about 94 hours per Craftax-1M run on one RTX 4090; shorter token sequences should cut this, which is part of the result.

## Novelty check

- Searched: Simulus repository README, Semantic Scholar citations of arXiv 2402.16801.
- Simulus-A (repository default since August 2026) changes the image tokenizer (FSQ) but has no Craftax results yet ([[Simulus (Cohen 2025)]]).
- A per-field integer tokenizer for symbolic Craftax: not found.

## Suggested by existing work

- Simulus §5 Limitations and Future Work: "more efficient solutions can be found for dealing with continuous inputs" ([[Simulus (Cohen 2025)]]).
- Efficient MBRL §5: generalise the tokenizer "to extract patches and tokens from large pre-trained models"; let the policy accept latent tokens ([[Efficient MBRL (Dedieu 2025)]]).

## How to evaluate

- Craftax-1M reward % of max against block D of the [[Existing Implementations]] table (Simulus 6.59%).
- Tokens per observation and training wall-clock against Simulus.

## References

- Cohen, L. et al. "Simulus: Combining Improvements in Sample-Efficient World Model Agents." 2025. https://arxiv.org/abs/2502.11537
- Dedieu, A. et al. "Improving Transformer World Models for Data-Efficient RL." 2025. https://arxiv.org/abs/2502.01591
