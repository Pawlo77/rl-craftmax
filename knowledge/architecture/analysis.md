---
title: "analysis"
type: module
domain: analysis
parent: "[[wut_murl_craftax]]"
tags: [architecture, analysis]
created: 2026-10-24
updated: 2026-10-24
related-code:
  - "[[src/wut_murl_craftax/analysis/__init__.py]]"
---
# analysis

## Responsibility

- Phase 4 analysis of finished runs: load the exported tables, per-method training analysis, obstacles, cases ([[2026-10-20-craftax-training-analysis]]) and the method comparison ([[2026-10-20-craftax-comparison-and-results]]).
- Filled by the training analysis and comparison tasks; the package is empty so far.

## Public surface

- None yet.

## Data flow

- In: the export tables of [[evaluation]] under `outputs/metrics/<phase>/`; eval files and checkpoints through [[visualisation]].
- Out: `outputs/analysis/<budget>/`, `outputs/results/`, MLflow experiment `analysis`.

## Invariants

- Runs belong to a phase by their `phases` tag, never by experiment name; statistics are per (method, `config_ref`).

## Known issues

- None.

## Tests

- None yet; `tests/analysis/` comes with the first module.
