---
title: "Craftax Hugging Face release implementation plan"
type: plan
domain: release
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, release, huggingface, publication]
---
# Craftax Hugging Face release implementation plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the last step of [[2026-10-20-craftax-project]], after the report is approved: a public dataset repo `Pawlo77/wut-murl-craftax-results` with the MLflow store, artifacts, metric exports, final-policy checkpoints and media, and a small public Space `Pawlo77/wut-murl-craftax` showing the results table, key plots, videos and GIFs, both created and uploaded with `huggingface_hub` behind an explicit confirmation gate.

**Architecture:** a new package `src/wut_murl_craftax/release/`: `scrub.py` makes a publishable copy of `mlruns.db` and the artifact folders; `bundle.py` stages `outputs/release/dataset/` and `outputs/release/space/` and writes a manifest (path, size, SHA-256); `publish.py` creates both repos as private, uploads, verifies against the manifest, and makes them public only after a second confirmation. The Space is a Gradio app of about 100 lines in `src/wut_murl_craftax/release/space/` that ships its few tables and images in its own `data/` folder and links to the dataset for everything else. Publication decisions: [[Experiment decisions]] and [[Visualisation decisions]].

**Tech Stack:** `huggingface_hub` (`HfApi.create_repo`, `upload_large_folder`, `upload_folder`, `list_repo_tree`, `update_repo_settings`, `whoami`), MLflow (SQLite), pandas, Gradio (Space runtime only).

## Preconditions

- The report is approved by the user ([[2026-10-20-craftax-report]], Task 6).
- The videos are published and their URLs are in `configs/video/published.yaml` ([[2026-10-20-craftax-visualisation-production]]).
- The final exports in `outputs/metrics/`, `outputs/results/`, `outputs/analysis/` and `outputs/interpretation/` were rebuilt after the last MLflow change.
- `make check` passes on the commit that produced the results; its hash goes into both cards.

## Dataset contents

| Path in the repo | Content | Source |
|---|---|---|
| `README.md` | dataset card | `release/cards.py` |
| `mlruns.db` | scrubbed MLflow store: tuning and final phases, analysis runs | `release/scrub.py` |
| `mlruns/<experiment id>/<run id>/artifacts/` | per run: `config.json`, `metrics.npz`, `episodes.npz`, `eval/*.npz`, `eval_tables/*.parquet` | MLflow artifact root |
| `metrics/<phase>/` | `runs`, `curves`, `eval_episodes` as parquet and CSV | [[2026-10-20-craftax-evaluation-and-metrics]] export |
| `results/figures/`, `results/tables/` | every figure (PDF, PNG) and table (CSV, LaTeX) | [[2026-10-20-craftax-comparison-and-results]] |
| `analysis/<budget>/<method>/` | hard scenarios, `cases.csv`, obstacles tables, plots | [[2026-10-20-craftax-training-analysis]] |
| `interpretation/<budget>/<method>/` | fidelity curves, complexity, 16-leaf rules, importance | [[2026-10-20-craftax-interpretation-and-ruleset]] |
| `checkpoints/<budget>/<method>/s<seed>/` | final params (`final.msgpack`), `config.json`, model card `README.md` | final runs, all 10 seeds |
| `checkpoints/<budget>/<method>/video/` | the early checkpoints used in the videos (best seed only) | `ckpt/index.json` |
| `media/gifs/`, `media/videos/` | GIFs and the three MP4s | visualisation outputs |
| `report/report.pdf` | the report | `report/` |
| `LICENSE`, `NOTICE` | licences and third-party notices | this plan |

- Size estimate for checkpoints: a Craftax-Symbolic width-512 params-only checkpoint is 38,163,145 B ([[Recording training data]]); multiply by methods x 2 budgets x 10 seeds and replace with measured sizes in Task 3.
- Videos with music go in only if every music licence in `configs/video/music.yaml` allows redistribution outside YouTube; otherwise the voice-only mix (`ffmpeg.mux_audio` with `--voice` only) is uploaded.
- Never uploaded: `.env`, tokens, host names, absolute local paths, tuning checkpoints, 16 px masters, lossless clips.

## Scrubbing

- Copy `mlruns.db` to `outputs/release/mlruns.db`; the original is never modified.
- Delete tags `mlflow.user`, `mlflow.source.name`, `mlflow.source.git.repoURL` if it holds a local path, and any tag whose value contains a host name or an absolute home path.
- Keep tags `method`, `budget`, `phase`, `seed`, `config_hash`, `profile`, `device_kind`, `mlflow.source.git.commit`.
- Rewrite every `artifact_uri` and `artifact_location` to the relative form `mlruns/<experiment id>/<run id>/artifacts`.
- Check that run, metric, param and tag counts (minus deleted tags) equal the original store's.

## Space

- SDK: Gradio (Python only; no custom HTML, CSS or JavaScript), free CPU hardware.
- `bundle.py` copies `main.csv`, `literature.csv`, `game_summary.csv`, 6 key PNGs (F1, F2, F6, F8, F17, fidelity vs leaves) and 4 GIFs into the Space's `data/` folder (a few MB), so the Space works while the dataset is still private and needs no token secret.
- `app.py` reads only `data/`; links to the dataset, the report PDF in the dataset and the videos are plain URLs.
- Tabs: Results (two tables), Plots (gallery), Videos (YouTube links from `published.yaml` and 4 GIFs), Data (how to download the dataset, open `mlruns.db` with `mlflow ui --backend-store-uri sqlite:///mlruns.db`, load a checkpoint with `flax.serialization.from_bytes`).
- `requirements.txt`: `gradio` and `pandas`, each pinned to the version tested locally.
- Space card `README.md` front matter: `title`, `emoji`, `colorFrom`, `colorTo`, `sdk: gradio`, `sdk_version`, `app_file: app.py`, `license`, `short_description`, `datasets: [Pawlo77/wut-murl-craftax-results]`.

## Cards and licence

- Dataset card front matter: `license`, `pretty_name`, `task_categories: [reinforcement-learning]`, `tags: [craftax, jax, mlflow, reinforcement-learning]`, `size_categories`, `configs` entries for the `metrics/` parquet files so the dataset viewer shows them.
- Dataset card sections: summary, contents table with measured sizes, how the data was produced (repository URL and commit), protocol and metric definitions (summary of [[2026-10-20-craftax-evaluation-and-metrics]]), how to load each part, licence and attribution, limitations, citation (from `CITATION.cff`).
- Model card per checkpoint folder: method, network, observation (`Craftax-Symbolic-v1`, craftax 1.6.1), budget, seed, final score and eval scores, loading snippet, intended use (research and teaching), limitations.
- Proposed licences: CC-BY-4.0 for data, figures, tables, GIFs and videos; Apache-2.0 (the repository licence) for checkpoints and the Space code.
- `NOTICE`: Craftax (MIT, Matthews et al.), the Craftax_Baselines portions of `training/ppo.py` (MIT), each music track's licence and attribution.

## Confirmation gate

- Gate 1: the agent shows the dry-run manifest (repo names, visibility, every top-level folder with file count and size, total size, licences) and waits.
- Gate 2: the user logs in themselves (`hf auth login`, or `HF_TOKEN` in their shell); the agent never reads, types or stores the token.
- Gate 3: the user replies "publish" in chat; only then does `publish.py --upload` run, creating both repos as private.
- Gate 4: after verification the agent shows the private URLs; the user checks the Space and the dataset in the browser.
- Gate 5: the user replies "make public" in chat; only then does `publish.py --make-public` run.
- Each gate covers one action; an approval never carries over to a later run of the script.

---

### Task 1: Scrubbed MLflow copy

**Files:**
- Create: `src/wut_murl_craftax/release/__init__.py`, `src/wut_murl_craftax/release/scrub.py`
- Modify: `pyproject.toml` (`uv add huggingface_hub` with a justification comment)
- Test: `tests/release/test_scrub.py`

- [ ] **Step 1: Write the failing tests** on a temporary store with tags `mlflow.user`, a host name tag and an absolute artifact URI: the scrubbed copy has none of them, keeps the allowed tags, has relative artifact URIs, and has the same run, metric and param counts; the source file's SHA-256 is unchanged.
- [ ] **Step 2: Implement** `scrub_store(src, dst)` with the rules under Scrubbing and `scrub_report(src, dst) -> dict` (counts before and after, deleted tag keys).
- [ ] **Step 3: Run** `uv run pytest tests/release/test_scrub.py -q`.

### Task 2: Bundle and manifest

**Files:**
- Create: `src/wut_murl_craftax/release/bundle.py`, `src/wut_murl_craftax/release/cards.py`
- Test: `tests/release/test_bundle.py`

- [ ] **Step 1: Write the failing tests** on a fixture tree: the staged dataset holds exactly the paths of the Dataset contents table; tuning checkpoints, masters and `.env` are excluded; `manifest.json` lists path, size and SHA-256 of every file; videos with music are excluded when `music.yaml` says redistribution is not allowed; the cards contain the measured sizes and the commit hash.
- [ ] **Step 2: Implement** `stage_dataset(out)`, `stage_space(out)`, `write_manifest(dir)`, `dataset_card(...)`, `model_card(run)`; the CLI `uv run python -m wut_murl_craftax.release.bundle --out outputs/release`.
- [ ] **Step 3: Run** `uv run pytest tests/release/test_bundle.py -q`.

### Task 3: Size check

- [ ] **Step 1: Run** the bundle CLI on the real outputs.
- [ ] **Step 2: Run** `du -sh outputs/release/dataset/*` and `find outputs/release/dataset -type f | wc -l`; write the sizes and counts per top-level folder under this task.
- [ ] **Step 3: Check** the current Hugging Face limits (largest file, files per folder, total repo size for a free account) on `https://huggingface.co/docs/hub/storage-limits`; write them under this task with the date read.
- [ ] **Step 4: Fix** any breach: split large folders into subfolders, or drop the video checkpoints first; never drop final checkpoints or `mlruns.db` without asking the user.

### Task 4: Space app

**Files:**
- Create: `src/wut_murl_craftax/release/space/app.py`, `src/wut_murl_craftax/release/space/requirements.txt`, `src/wut_murl_craftax/release/space/README.md`
- Test: `tests/release/test_space_app.py`

- [ ] **Step 1: Write the failing test** (`pytest.importorskip("gradio")`): `build_demo(data_dir)` on a fixture `data/` folder returns a `gr.Blocks` with the 4 tabs, the results table has the rows of `main.csv`, and building the app makes no network call.
- [ ] **Step 2: Implement** `app.py` (under 120 lines) as described under Space.
- [ ] **Step 3: Run** locally: `uv run --with gradio python outputs/release/space/app.py` (after the bundle CLI staged `outputs/release/space/`); open `http://127.0.0.1:7860`; check every tab; record the tested package versions in `requirements.txt`.

### Task 5: Publish script with gates

**Files:**
- Create: `src/wut_murl_craftax/release/publish.py`
- Test: `tests/release/test_publish.py`

- [ ] **Step 1: Write the failing tests** with a mocked `HfApi`: `--dry-run` (the default) makes no API call and prints the manifest summary; `--upload` without an interactive "publish" answer exits without calls; `--upload` creates both repos with `private=True`, uploads the dataset with `upload_large_folder` and the Space with `upload_folder`; verification fails when a remote file's size or SHA-256 differs from the manifest; `--make-public` without an interactive "make public" answer exits without calls.
- [ ] **Step 2: Implement** `dry_run`, `upload` (`create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)`; `create_repo(repo_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)`), `verify` (`list_repo_tree(recursive=True)` against the manifest) and `make_public` (`update_repo_settings(repo_id, repo_type=..., private=False)`); `whoami()` must return `Pawlo77` before any write.
- [ ] **Step 3: Run** `uv run pytest tests/release/test_publish.py -q`.

### Task 6: Release

- [ ] **Step 1: Run** `uv run python -m wut_murl_craftax.release.publish --dry-run` and show the output to the user (Gate 1).
- [ ] **Step 2: Wait** for the user to log in (Gate 2) and to reply "publish" (Gate 3).
- [ ] **Step 3: Run** `uv run python -m wut_murl_craftax.release.publish --upload`; it uploads as private and verifies; show the private URLs (Gate 4).
- [ ] **Step 4: Check** with the user: the Space builds and shows the tables, the plots and the GIFs; the dataset viewer shows the `metrics/` tables; one checkpoint downloads and loads with the snippet in its card.
- [ ] **Step 5: Wait** for "make public" (Gate 5), then run `uv run python -m wut_murl_craftax.release.publish --make-public`.

### Task 7: After the release

**Files:**
- Create: `knowledge/architecture/release.md` (template `knowledge/_templates/module.md`)
- Modify: `README.md` (links to the Space, the dataset, the report and the videos)

- [ ] **Step 1: Check** that the Space and dataset URLs cited in the report resolve while logged out.
- [ ] **Step 2: Write** the module note and the README links; append a dated section to [[Experiment decisions]] with the final licences and what was left out of the release.
- [ ] **Step 3: Run** `make knowledge` and `make check`.
