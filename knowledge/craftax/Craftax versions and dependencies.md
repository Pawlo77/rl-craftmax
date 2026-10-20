---
title: "Craftax versions and dependencies"
type: concept
domain: craftax
parent: "[[Craftax]]"
tags: [craftax, concept, dependencies, versions]
created: 2026-10-20
---
# Craftax versions and dependencies

## Current release

| Item | Value | Source |
|---|---|---|
| Latest version | 1.6.1, uploaded 2026-06-20 | [PyPI JSON](https://pypi.org/pypi/craftax/json) |
| `requires-python` | `>=3.9` | [pyproject.toml](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/pyproject.toml) |
| Python badge in README | 3.9, 3.10, 3.11, 3.12 | [README](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/README.md) |
| Runtime dependencies | `jax`, `flax`, `numpy`, `pygame`, `matplotlib`, `imageio` | pyproject.toml |
| Version pins | none: no lower or upper bound on any dependency | PyPI `requires_dist` |
| `chex` | not a dependency since 1.6.0 | PyPI per-version metadata |
| `gymnax` | not a dependency since 1.6.0 (base classes and spaces vendored in `craftax.environment_base`) | [release v1.6.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.6.0) |
| Extra `dev` | `ruff`, `pre-commit` | pyproject.toml |
| Build backend | `setuptools>=64.0` | pyproject.toml |
| Console scripts | `play_craftax`, `play_craftax_classic` | pyproject.toml |
| Licence | MIT | [LICENSE](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/LICENSE) |

- Undeclared import: `constants.py` imports `PIL` (Pillow); it arrives through `matplotlib` (`pillow>=9`) and `imageio` (`pillow>=8.3.2`).
- `pygame` is imported only by the play scripts and `matplotlib` only by a demo in `noise.py`, but both are hard dependencies ([issue #55](https://github.com/MichaelTMatthews/Craftax/issues/55), open).
- Resolved on 2026-10-20 with `uv run --isolated --python 3.12 --with craftax` on macOS arm64: craftax 1.6.1, jax 0.12.0, jaxlib 0.12.0, flax 0.12.10, numpy 2.5.4, pygame 2.6.1, matplotlib 3.11.2, imageio 2.38.1, pillow 13.0.0; all eight envs reset and stepped on CPU.

## Python support

- 1.0.0 declared `>=3.8`; 1.1.0 and later declare `>=3.9`.
- Python 3.14: `pip install craftax` fails because `pygame` has no cp314 wheel and fails to build from source; the reporter ran craftax on 3.14 only after moving `pygame` and `matplotlib` to an extra ([issue #55](https://github.com/MichaelTMatthews/Craftax/issues/55), open, filed 2026-09-07).

## Dependency history

`requires_dist` per release, from `https://pypi.org/pypi/craftax/<version>/json`:

| Versions | Runtime dependencies |
|---|---|
| 1.0.0 | jax, distrax, optax, flax, numpy, black==23.3.0, pre-commit==3.3.3, argparse, wandb, orbax-checkpoint, pygame, gymnax, chex, matplotlib, imageio |
| 1.1.0 - 1.4.3 | same without the black / pre-commit pins |
| 1.4.4 - 1.4.5 | jax, distrax, optax, flax, numpy, pygame, gymnax, chex, matplotlib, imageio; black, pre-commit moved to `dev` |
| 1.5.0 | jax, flax, numpy, pygame, gymnax, chex, matplotlib, imageio |
| 1.6.0 - 1.6.1 | jax, flax, numpy, pygame, matplotlib, imageio; `dev` = ruff, pre-commit |

- `gymnax` "is not maintained anymore and is incompatible with `jax>=0.6.0`" per the reporter of [issue #40](https://github.com/MichaelTMatthews/Craftax/issues/40); craftax up to 1.5.0 depends on it, so with `jax>=0.6.0` use craftax >= 1.6.0 (inference from that report, not tested here).
- `flax` is kept only for `flax.struct.dataclass` and its `.replace()` (maintainer in issue #40).
- Craftax_Baselines `requirements.txt` (unpinned except `orbax-checkpoint==0.5.0`): `jax[cuda12_pip]`, distrax, optax, flax, numpy, black, pre-commit, argparse, wandb, pygame, gymnax, chex, matplotlib, imageio, craftax ([requirements.txt](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/7ce36fa/requirements.txt)); its Dockerfile uses `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` with Python 3.8.

## Releases

| Version | Date (GitHub) | Changes |
|---|---|---|
| 1.0.0 | 2024-02-29 | Craftax release |
| 1.1.0 | 2024-03-05 | UED hooks for JaxUED; cached textures; README recommends v1.1.0 or earlier for the offline human dataset |
| 1.1.1 | 2024-03-06 | Fix play scripts |
| 1.1.2 | 2024-03-24 | god mode fix; symbolic obs description |
| 1.2.0 | 2024-03-26 | Breaking: package moved from `src` to `craftax`, all imports need the `craftax.` prefix |
| 1.3.0 | 2024-04-07 | Baselines moved to Craftax_Baselines |
| 1.4.0 | 2024-04-30 | 8 explicit env classes (auto-reset or not); pixel resolution fix; optimistic reset wrapper fix; `always_diamond` option |
| 1.4.1 | 2024-04-30 | Offline dataset; ladder ascend bug |
| 1.4.2 | 2024-05-29 | Achievement logging bug |
| 1.4.3 | 2024-06-11 | Obs space dtypes; `tree_map` calls; texture-cache validity check |
| 1.4.4 | 2024-10-17 | Dev dependencies split out; `jax.tree.map` reverted for older JAX |
| 1.4.5 | 2025-01-20 | Texture caching bug (cache no longer shipped in the wheel, [issue #31](https://github.com/MichaelTMatthews/Craftax/issues/31)) |
| 1.5.0 | 2025-07-07 | Plant growth bug (first planted plant could not ripen) |
| 1.6.0 | 2026-06-16 | Classic lava and final-step health reward fixes; macOS image-loading fix; `env_params` no longer static; gymnax removed; black replaced by ruff |
| 1.6.1 | 2026-06-20 | Textures load only when needed |

- Source: [GitHub releases](https://github.com/MichaelTMatthews/Craftax/releases); PyPI upload dates match within a day.
- The authors intend to change game mechanics "only in exceptional circumstances, so that old experiments are not invalidated" ([MAINTENANCE.md](https://github.com/MichaelTMatthews/Craftax/blob/c3c2e0d/MAINTENANCE.md)); 1.5.0 and 1.6.0 still changed rewards and plant growth, so record the craftax version with every result (errata in [[Craftax known issues]]).
