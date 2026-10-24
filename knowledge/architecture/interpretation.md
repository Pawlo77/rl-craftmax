---
title: "interpretation"
type: module
domain: interpretation
parent: "[[wut_murl_craftax]]"
tags: [architecture, interpretation]
created: 2026-10-24
updated: 2026-10-24
related-code:
  - "[[src/wut_murl_craftax/interpretation/__init__.py]]"
  - "[[src/wut_murl_craftax/interpretation/features.py]]"
---
# interpretation

## Responsibility

- Explain trained policies: observation features, rollout datasets, decision-tree distillation, tree policies and rules ([[2026-10-20-craftax-interpretation-and-ruleset]]).
- `features`: decode symbolic observations of both games into named, human-readable tree inputs.

## Public surface

- `FEATURES[game] -> list[str]`: column names (111 Craftax, 60 Classic): intrinsics, inventory counts, equipment, potions, flags, facing, creatures, walkable neighbours, then distance and row and column offset signs of the nearest visible target (17 Craftax, 10 Classic).
- `featurize(obs, game, xp=None) -> array`: float32 `(..., len(FEATURES[game]))` from flat observations `(..., D)`; `xp` is numpy or jax.numpy (default: the module of `obs`), so it jits inside closed-loop tree play.
- `featurize` raises `ValueError` when the observation width is not the game's (8268 Craftax, 1345 Classic).
- `CREATURES`: names of the `facing_creature` codes.
- Ordinal codes (a split such as `direction <= 2.5` means "left or right"; rules must translate thresholds into sets):

| Column | Codes |
|---|---|
| `direction` | 1 left, 2 right, 3 up, 4 down (`Action` ids) |
| `facing_creature` | 0 none, 1 melee, 2 passive, 3 ranged, 4 enemy projectile |
| `facing_block` | `BlockType` id; 0 (`INVALID`) is an unlit tile |

## Data flow

- In: flat symbolic observations (`featurize`); runs and network outputs from [[visualisation]] (`EvalRun`, `network_outputs`), final scores from MLflow.
- Out: `outputs/interpretation/<budget>/` (trees, rules, `complexity.csv`).

## Invariants

- Policies are chosen by the `phases` tag and `config_ref`, never by experiment name.
- Map decoding is `visualisation.overlay.decode_obs`; flat offsets come from `CRAFTAX_BARS`; no second decoder.
- `featurize` gives identical values under numpy and jitted jax.numpy; numpy runs in chunks of 4096 rows to bound memory.
- With no target in view, its distance is the sentinel (10 Craftax, 8 Classic) and both signs are 0; invisible tiles never count.
- An unlit tile decodes as block 0: never a target, never walkable (unknown is not safe); the game block `DARKNESS` stays walkable.
- `down_ladder` is the `LADDER_DOWN` item; whether it can be used is the `ladder_open` flag.
- Memory proxies and the current skill are not observation features; the dataset adds them.

## Known issues

- None.

## Tests

- `tests/interpretation/test_features.py`: round trips on edited reset states of both games, hand-placed targets and creatures, darkness, ladders, fountain and tree kinds, walkable blockers, Craftax creatures, width check, numpy vs jitted jax, 10,000 observations in whole chunks.
