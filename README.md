<div align="center">
  <h1>wut_murl_craftax</h1>
  <p><strong>Textbook reinforcement learning on Craftax, measured beyond the score.</strong></p>
  <p>Course project for <em>Mathematical Underpinning of Reinforcement Learning</em> (MURL), Warsaw University of Technology.</p>
  <p>
    <a href="https://github.com/Pawlo77/wut_murl_craftax/actions/workflows/test.yaml"><img src="https://github.com/Pawlo77/wut_murl_craftax/actions/workflows/test.yaml/badge.svg" alt="Pytest"></a>
    <a href="https://github.com/Pawlo77/wut_murl_craftax/actions/workflows/prek.yaml"><img src="https://github.com/Pawlo77/wut_murl_craftax/actions/workflows/prek.yaml/badge.svg" alt="Prek Checks"></a>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12"></a>
    <a href="https://github.com/MichaelTMatthews/Craftax"><img src="https://img.shields.io/badge/env-Craftax%201.6.1-green.svg" alt="Craftax 1.6.1"></a>
    <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/JAX-0.11-orange.svg" alt="JAX 0.11"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  </p>
</div>

## About

[Craftax](https://github.com/MichaelTMatthews/Craftax) is a fast, open-ended survival and crafting game written in JAX: 67 achievements across 9 floors, sparse rewards and a partial view of the world. This project trains three published agents as controls and four textbook-level methods on it, at the two scoreboard budgets (1M and 1B environment steps), and asks how they learn, not only how much they score:

- **Learning:** stability across seeds, hyperparameter sensitivity, hardware use.
- **Playing:** which agent unlocks an achievement first, in which order, how deep it gets, how it dies, explores, fights and survives.
- **Explaining:** how complex each final policy's decision rules are (decision-tree distillation).
- **Showing:** GIFs and YouTube videos of the learning process and the most interesting moments.

## Results

Results are in the report, [`report/report.pdf`](report/report.pdf) (4 pages), and, once published, in the Hugging Face Space [`Pawlo77/wut-murl-craftax`](https://huggingface.co/spaces/Pawlo77/wut-murl-craftax) with the full MLflow store in the dataset [`Pawlo77/wut-murl-craftax-results`](https://huggingface.co/datasets/Pawlo77/wut-murl-craftax-results). This README stays result-free on purpose.

## Methods

| Method | Role | Family | Design note |
|---|---|---|---|
| PPO | control | policy gradient | [PPO design](knowledge/methods/PPO%20design.md) |
| PPO-RNN | control | policy gradient with memory | [PPO-RNN design](knowledge/methods/PPO-RNN%20design.md) |
| PQN-RNN | control | value-based, Q(λ) | [PQN-RNN design](knowledge/methods/PQN-RNN%20design.md) |
| Count-Based Progress Bonus | add-on to PPO-RNN and PQN-RNN | exploration bonus | [design](knowledge/methods/Count-Based%20Progress%20Bonus%20design.md) |
| Self-Imitation Learning | add-on to PPO-RNN | learning from own successes | [design](knowledge/methods/Self-Imitation%20Learning%20design.md) |
| SMDP Q-Learning over Hand-Defined Skills | method | hierarchical RL | [design](knowledge/methods/SMDP%20Q-Learning%20over%20Hand-Defined%20Skills%20design.md) |
| Dyna-Q over a Skill-Level Progress Model | method | model-based planning | [design](knowledge/methods/Dyna-Q%20over%20a%20Skill-Level%20Progress%20Model%20design.md) |

What each method observes and how it is rewarded: [`knowledge/methods/Methods.md`](knowledge/methods/Methods.md). All methods are scored on the environment reward only. The skill methods use hand-written skills (prior knowledge) and are reported in a separate block.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/); `ffmpeg` on `PATH` for videos.

```bash
git clone https://github.com/Pawlo77/wut_murl_craftax.git
cd wut_murl_craftax
make install          # uv sync: Craftax, JAX, Apple GPU (metaljax) or CUDA backend
make warm             # build the Craftax texture caches, print the devices in use
cp .env.example .env  # persistent XLA compilation cache for every make target
```

On Apple Silicon, environments step fastest on the CPU and network updates on the GPU; on CUDA everything runs on the GPU. Details and measurements: [`Craftax runtime setup`](knowledge/craftax/Craftax%20runtime%20setup.md).

## Quickstart

```bash
make train-ppo ARGS="--env Craftax-Classic-Symbolic-v1 --steps 1e6"   # baseline PPO with checkpoints
make vis STEP=grid ARGS="--help"                                        # one visualisation step
make help                                                               # every target
```

Experiments run as resumable MLflow runs (`make mlflow` opens the UI); the execution plan below adds `make queue PHASE=<phase>`, which can be stopped with Ctrl-C and resumed at any time.

## Project map

| Where | What |
|---|---|
| [`src/wut_murl_craftax/`](src/wut_murl_craftax) | `orchestration` (runtime, MLflow), `training` (agents), `visualisation` (videos, charts) |
| [`plans/2026-10-20-craftax-project/`](plans/2026-10-20-craftax-project/2026-10-20-craftax-overview.md) | the execution plan: phases, per-method grids and time estimates |
| [`knowledge/`](knowledge/wut_murl_craftax.md) | Obsidian vault: Craftax framework, full game rules, methods, decisions, visualisation |
| [`research/`](research/Research.md) | Craftax Baselines, every published Craftax agent, candidate algorithms, Sutton and Barto notes |
| [`report/`](report) | the 4-page report (Interspeech LaTeX template) |
| `configs/`, `outputs/` | experiment configs; generated outputs (git-ignored) |

Open the repository folder as an Obsidian vault and start at [`wut_murl_craftax`](knowledge/wut_murl_craftax.md). Development setup and checks: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgements

- [Craftax](https://github.com/MichaelTMatthews/Craftax) (MIT): the environment.
- [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines) (MIT): the PPO, PPO-RNN and exploration baselines our controls are adapted from; vendored code keeps its notice.
- [purejaxql](https://github.com/mttga/purejaxql) (Apache-2.0): the PQN-RNN reference implementation.
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*: the methods' foundations.

## Citing

Please cite this repository ([`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button gives APA or BibTeX) and Craftax, whose paper also describes the Craftax_Baselines code:

```bibtex
@misc{pozorski2026wutmurlcraftax,
  author = {Pozorski, Pawe{\l}},
  title = {wut\_murl\_craftax: Textbook reinforcement learning on Craftax},
  year = {2026},
  howpublished = {\url{https://github.com/Pawlo77/wut_murl_craftax}},
  note = {Course project, Mathematical Underpinning of Reinforcement Learning, Warsaw University of Technology}
}

@inproceedings{matthews2024craftax,
  author = {Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
  title = {Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
  booktitle = {International Conference on Machine Learning ({ICML})},
  year = {2024}
}
```

## License

MIT License. See [LICENSE](LICENSE). Third-party code keeps its own licence header.

Copyright 2026 Paweł Pozorski.
