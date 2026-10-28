---
title: "Template analysis lessons implementation plan"
type: plan
domain: scripts
parent: "[[wut_murl_craftax]]"
status: draft
created: 2026-10-28
updated: 2026-10-28
tags: [plan, template]
---
# Template Analysis Lessons Implementation Plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry the lessons of this project's results and figure work into `Pawlo77/python-kit-template` v0.13.0: ship the project-independent skill `analysis-figures` (Task 1), and fix the other template problems met here that no other template plan covers (Tasks 2-9). These are template issues recorded here because the template repo is private to another owner and no GitHub issue can be opened. Already covered elsewhere, not repeated: [[2026-10-20-template-knowledge-eof-loop]], [[2026-10-20-template-obsidian-onboarding]], [[2026-10-20-template-agent-setup]], [[2026-10-20-template-claude-plugin-settings]], [[2026-10-20-template-ty-diagnostics]], [[2026-10-20-template-uv-build-pin]].

**Architecture:** Each task is one issue: reproduce on a fresh render, fix in the template with a test, then take it here with `copier update`. Render as in [[2026-10-20-template-knowledge-eof-loop]] Task 1 (`scaffold=experiment`, `vault=true`). Every problem below was checked against the scaffold commit `0fca6e8` (unchanged template files); project-specific causes (the JAX runtime, the Craftax pipeline) are left out.

**Tech Stack:** Python 3.12, pytest, prek, MLflow 3.17.1 (SQLite store), git, copier, Agent Skills.

---
### Task 1: Ship the analysis-figures skill

The template has no skill for quantitative analysis or report figures; `development`, `python-style`, `check` and rule `experiment-boundaries.mdc` cover code, style and folders only, so nothing similar exists to extend. Here, the rules were learned through review blockers: an immortal-time bias (groups formed by "ever unlocked"), selection bias (tuned configurations reported on their tuning seeds), a pooled correlation whose sign flips within groups (Simpson's paradox), censored values quoted as numbers, laptop wall-clock ranked, single-seed claims, and figures with text, cropped labels and repeated legends.

**Files:**
- Create: `template/.agents/skills/analysis-figures/` (`SKILL.md`, `references/analysis.md`, `references/figures.md`), copied from this repo's `.agents/skills/analysis-figures/`
- Modify: the experiment scaffold condition (render it only with `scaffold=experiment`, since it assumes `report/`)

- [ ] **Step 1:** Copy the three files unchanged; they name no project. Project paths (figure module, captions file, macros) go to an optional `local.md` beside `SKILL.md` (see Task 9).
- [ ] **Step 2:** `make skills-check` and `uv run python scripts/check_brevity.py` pass on a fresh experiment render; a library render has no `analysis-figures` folder.
- [ ] **Step 3:** In a fresh render, ask an agent to "compare two methods over seeds and draw the figure": the skill loads by its description. Commit `feat: add the analysis-figures skill to the experiment scaffold`.

### Task 2: commit-msg rejects git's own merge and revert subjects

- Reproduce: `python scripts/hooks/commit_msg.py --check-text "Merge branch 'main' into feat/x"` and `--check-text 'Revert "feat: x"'` both fail ("subject must match 'type: summary'"). `git merge` runs the commit-msg hook, and skill `ship` says to merge `origin/main` into the branch (never rebase), so every merge needs a hand-typed subject; here all 21 merges were renamed `chore: merge main into ...`.

**Files:**
- Modify: `template/scripts/hooks/commit_msg.py`
- Test: `template/tests/hooks/test_commit_msg.py`

- [ ] **Step 1: Write the failing test:** git's default subjects (`Merge branch '<b>'[ of <url>][ into <b>]`, `Merge remote-tracking branch ...`, `Merge pull request #<n> from ...`, `Revert "<subject>"`) pass; `Merge stuff` and `revert things` still fail.
- [ ] **Step 2:** Accept those patterns before the type check (as `[prek]` already is). Commit `fix: accept git's merge and revert commit subjects`.

### Task 3: The test fixture deletes live MLflow artifacts

- Reproduce: start a run that writes the repo-root `mlruns/` (any `MlflowRunTracker` run with the default URI), then run `make test` beside it. The autouse `cleanup_mlflow_artifacts` fixture in `tests/conftest.py` deletes every file created under `mlruns/` during each test, including the live run's checkpoints and artifacts. Tests also write into the repo store itself. Here it deleted live checkpoints; fixed in `e615e40`.

**Files:**
- Modify: `template/tests/conftest.py`, `template/src/{{module}}/orchestration/tracking.py`
- Test: a hook test that runs a tracker under pytest

- [ ] **Step 1: Write the failing test:** a test that starts a tracked run leaves the repo-root `mlruns.db` and `mlruns/` byte-identical, and a file created there by another process during the test survives.
- [ ] **Step 2:** Replace the cleanup fixture with a session fixture that points `MLFLOW_TRACKING_URI` at a temporary SQLite store, plus a session check that fails if the repo store changed. Commit `fix: isolate tests from the repository MLflow store`.

### Task 4: `.gitignore` misses the default MLflow store

- Reproduce: the default `tracking_uri` (`config.py`, `configs/experiment.yaml`) is `sqlite:///mlruns.db`, but `.gitignore` lists `mlflow.db` and `mlflow.db-*`. After one run, `git status` shows `mlruns.db` as untracked, although rule `experiment-boundaries.mdc` says it stays ignored. Fixed here in `2a19500`.

**Files:**
- Modify: `template/.gitignore`
- Test: a test that `git check-ignore` matches `mlruns.db`, `mlruns.db-journal` and `mlruns/x`

- [ ] **Step 1:** Write the test, add `mlruns*.db` and `mlruns*.db-*`. Commit `fix: ignore the default MLflow SQLite store`.

### Task 5: The SQLite store has no busy timeout

- Reproduce: hold a read transaction on `mlruns.db` for over 5 s (`sqlite3` shell, `BEGIN; SELECT count(*) FROM metrics;`) while a run logs metrics: the writer fails with `sqlite3.OperationalError: database is locked`. The store is in rollback-journal mode and SQLite's default busy timeout is 5 s. Here a long-running replay crashed this way when an agent queried the store; fixed with `sqlite:///mlruns.db?timeout=300`. `_resolve_tracking_uri` keeps the query only by accident (`Path.resolve` leaves `?timeout=300` in the file name).

**Files:**
- Modify: `template/src/{{module}}/config.py`, `template/configs/experiment.yaml`, `template/src/{{module}}/orchestration/tracking.py`
- Test: `template/tests/test_config.py`, a tracking test

- [ ] **Step 1: Write the failing test:** a writer logs while a second connection holds a read lock for 6 s; with the default URI the log call succeeds after the lock is released.
- [ ] **Step 2:** Default to `?timeout=300`; split the query off before resolving the path and append it again. Commit `fix: give the SQLite MLflow store a busy timeout`.

### Task 6: Tracking leaks environment variables into later tests

- Reproduce: in MLflow 3.17.1, `mlflow.set_tracking_uri` and `mlflow.set_experiment` (called by `MlflowRunTracker.start`) export `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_ID` into `os.environ`, and `start` sets `MLFLOW_ACTIVE_RUN_ID`, removed only by `close()`. `tests/conftest.py` restores none of them, so a test that starts a tracker in-process changes the store and experiment of every later test and of the subprocesses they start: failures depend on test order. Here the same class hit `XLA_FLAGS` (project code) and `MLFLOW_*`.

**Files:**
- Modify: `template/tests/conftest.py`, `template/src/{{module}}/orchestration/tracking.py`
- Test: a conftest test

- [ ] **Step 1: Write the failing test:** two tests in one session; the first starts and fails a tracker without `close()`, the second sees the session's `MLFLOW_*` values unchanged.
- [ ] **Step 2:** Add an autouse fixture that snapshots and restores `os.environ` around every test; in the tracker, pop `MLFLOW_ACTIVE_RUN_ID` in a `finally`. Note in `development` that a CLI called in-process must not leave process state behind. Commit `fix: restore the environment after every test`.

### Task 7: Generated knowledge notes conflict on merges and go stale on rebases

- Reproduce: two branches each add a public function, merge one into the other: `knowledge/generated/Modules.md` conflicts. Rebase instead: no hook runs, so the rebased commits carry a stale note and the next commit's `knowledge-build` rewrites it. Here 18 of 21 merge commits had to resolve a generated note and 6 commits were "regenerate the module index after the rebase".

**Files:**
- Modify: `template/.gitattributes`, `template/kit.mk`, `template/.pre-commit-config.yaml` (post-merge and post-rewrite stages)
- Test: a hook test with two branches in a temporary repo

- [ ] **Step 1: Write the failing test:** the two-branch merge completes without a conflict and the merged tree passes `make vault-check`.
- [ ] **Step 2:** Mark `knowledge/generated/**` with a merge driver that runs `make knowledge` (or `merge=ours` plus a `post-merge` and `post-rewrite` hook that rebuilds and reports the notes to stage). Commit `fix: rebuild generated notes after merges and rebases`.

### Task 8: Generated report inputs are neither tracked nor ignored

- Reproduce: the experiment scaffold ships `report/figures/.gitkeep` and a `report/.gitignore` that lists LaTeX build files only. A pipeline that writes figures, tables and a macros file (`numbers.tex`) into `report/` leaves them untracked and unignored, and nothing says which is meant; here 66 figures, the tables and `numbers.tex` sat untracked until a ruling chose to commit them so the report builds from git.

**Files:**
- Modify: `template/report/.gitignore`, `template/.cursor/rules/experiment-boundaries.mdc`, `template/CONTRIBUTING.md`
- Test: a test that a fresh render's `report/` has no untracked, unignored path after the documented report build

- [ ] **Step 1:** Decide once: generated report inputs are committed (the report builds from git) and live in `report/figures/<set>/`, `report/tables/<set>/` and `report/numbers.tex`; state it in the rule. Keep `check-added-large-files` passing for 300 dpi PNGs.
- [ ] **Step 2:** Add the test. Commit `docs: state that generated report inputs are committed`.

### Task 9: Skills point at files and skills that do not exist

- Reproduce: in a fresh render, `development`, `check`, `python-style`, `obsidian` and `upgrade` tell the agent to read `local.md` "in this folder" (`ci` and `ship` add "when it exists"), and no `local.md` is rendered. `plans/AGENTS.md` and `ship` name skill `vibe`, which is not rendered.

**Files:**
- Modify: those skills, `template/plans/AGENTS.md`
- Test: `template/tests/hooks/test_skill_frontmatter.py` (every backticked skill name and every `local.md` reference resolves, or the reference says "when it exists")

- [ ] **Step 1: Write the failing test** over a fresh render.
- [ ] **Step 2:** Render a stub `local.md` per skill that names it (a heading and "nothing project-specific yet"), or add "when it exists" everywhere; ship skill `vibe` or remove its mentions. Commit `fix: resolve dangling skill references`.

### Task 10: The test suite runs serially in CI

A grown project's suite (about 1,500 tests here, 33 min serially on a laptop) cannot finish inside the 30-minute job of `test.yaml`; the template ships no parallel runner.

**Files:**
- Modify: `template/pyproject.toml` (dev group: `pytest-xdist`; `[tool.pytest.ini_options] addopts = "-n auto --dist loadfile"`), `template/.github/workflows/test.yaml` (timeout sized for the parallel run)
- Test: a fresh render runs `make tests` and the CI command in parallel with session fixtures (the MLflow store guard, isolated tracking) correct per worker

- [ ] **Step 1:** Add xdist and the `addopts`; check every session-scoped fixture is worker-safe (each worker has its own tmp store; no fixed paths in the repo).
- [ ] **Step 2:** Fresh render, run the suite with `-n auto` and with `-n 0`; both pass. Commit `ci: run the test suite on every core`.

### Task 11: Take the fixes here

- [ ] **Step 1:** `uvx copier update --trust` to the fixed template version; drop the local copies of Tasks 3-6 and 10 where the template now matches.
- [ ] **Step 2:** `make check` passes; mark this plan `done`.
