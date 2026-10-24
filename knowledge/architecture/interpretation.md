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
---
# interpretation

## Responsibility

- Explain trained policies: observation features, rollout datasets, decision-tree distillation, tree policies and rules ([[2026-10-20-craftax-interpretation-and-ruleset]]).
- Filled by the interpretation tasks; the package is empty so far.

## Public surface

- None yet.

## Data flow

- In: runs and network outputs from [[visualisation]] (`EvalRun`, `network_outputs`), final scores from MLflow.
- Out: `outputs/interpretation/<budget>/` (trees, rules, `complexity.csv`).

## Invariants

- Policies are chosen by the `phases` tag and `config_ref`, never by experiment name.

## Known issues

- None.

## Tests

- None yet; `tests/interpretation/` comes with the first module.
