---
title: "I1 - PPO-RNN 1B above the reproduction band"
type: issue
domain: evaluation
parent: "[[wut_murl_craftax]]"
status: open
severity: minor
found: 2026-11-05
created: 2026-11-05
fixed_in: ""
tags: [issue, reproduction]
related-code: ["[[src/wut_murl_craftax/evaluation/reproduction.py]]"]
---
# I1 - PPO-RNN 1B above the reproduction band

Package: [[evaluation]].

## Symptom

- Reproduction gate of `final-1b` (3 seeds each, `analysis.results`, 2026-11-05):

| Control | Ours (mean %) | Published (%) | Gap | h | Status |
|---|---|---|---|---|---|
| PPO | 11.72 | 11.9 | -0.18 | 0.63 | pass |
| PPO-RNN | 16.22 | 15.3 | +0.92 | 0.63 | fail |
| PQN-RNN | 16.10 | 16.0 | +0.10 | 0.90 | pass |

- PPO-RNN lies above the published value, not below it: there is no shortfall to explain, but the band is exceeded.

## Cause

- Not investigated. The band uses the published spread `s_pub` (0.35) over 10 seeds and our 3-seed standard error; the 1B runs ran on A100 and, for PPO-RNN, with TF32 (`configs/profiles/cuda.yaml`, [[PPO-RNN design]]).

## Repro

```bash
uv run python -m wut_murl_craftax.analysis.results --phases final-1m final-1b combo-1m --tuning tune-coarse tune-fine --out outputs/results
```

## Fix

- Report the gap beside the 1B results; the gate never blocks ([[2026-10-20-craftax-comparison-and-results]]).
