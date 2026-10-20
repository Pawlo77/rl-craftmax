---
title: "Template ty diagnostics implementation plan"
type: plan
domain: scripts
parent: "[[wut_murl_craftax]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, template, bug]
---
# Template Ty Diagnostics Implementation Plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix an upstream issue in `Pawlo77/python-kit-template` v0.13.0: a fresh render fails `ty`, with 54 diagnostics in template-shipped files that are unchanged since the scaffold commit. The `ty` hook is advisory (`uv run ty check src || true`) and checks only `src/`, so the errors go unnoticed. This is a template issue recorded here because the template repo is private to another owner and no GitHub issue can be opened.

**Architecture:** The code runs correctly. `ty` cannot prove the types where values are unpacked or come from `Any`, so each fix is a type annotation or an explicit argument, with no change in behavior. Then make the hook blocking so the template cannot drift again.

**Tech Stack:** Python 3.12, ty 0.0.90 (locked 2026-10-20; `pyproject.toml` asks only `ty>=0.0.85`, so a newer ty may be what surfaced these), prek.

---
### Task 1: Reproduce

- Render as in [[2026-10-20-template-knowledge-eof-loop]] Task 1, then `uv run ty check src scripts tests`: "Found 54 diagnostics".
- By file: `tests/hooks/test_gates.py` 22, `tests/hooks/test_lib_sh.py` 11, `tests/hooks/test_session_start.py` 8, `src/<slug>/config.py` 5, `tests/hooks/test_merge_gate.py` 2, and 1 each in `test_vault_engine.py`, `test_skill_frontmatter.py`, `test_shell_guard.py`, `tests/conftest.py`, `scripts/upgrade.py`, `scripts/check_vault.py`.
- Examples:
  - `config.py:88` `return cls(**dict(mapping))`: "Expected `bool`, found `Any | str`" (once per `bool` field of `MLflowTrackingConfig`; the `{**mapping, "tracking_uri": ...}` merge widens the value type).
  - `upgrade.py:246` `Item("hook", repo, *match.groups(), PRE_COMMIT)`: "Expected `bool`, found `Literal[\".pre-commit-config.yaml\"]`". `_HOOK` has exactly two groups, so `PRE_COMMIT` lands in `where` at run time.
  - `check_vault.py:34` `from check_memory_files import ...`: unresolved import (resolved at run time by a `sys.path` insert).
  - `tests/conftest.py:11` a generator fixture annotated `-> None`.
  - `test_gates.py:118` `shape: dict[str, object] = CLAUDE`: a `dict[str, str]` default.

### Task 2: Fix in the template

**Files:**
- Modify: the files listed in Task 1 (under `template/`), `.pre-commit-config.yaml` (`ty` hook)

- [ ] **Step 1:** `config.py`: annotate the merged mapping `dict[str, Any]` before `cls(**...)`.
- [ ] **Step 2:** `upgrade.py`: unpack `old, new = match.groups()` and pass `where=PRE_COMMIT` by keyword.
- [ ] **Step 3:** `check_vault.py`: add `scripts` to ty's `extra-paths` in `[tool.ty]`, or keep a `# ty: ignore[unresolved-import]` with a reason.
- [ ] **Step 4:** Tests: generator fixtures `-> Iterator[None]`; widen `CLAUDE` to `dict[str, object]` (or use `Mapping[str, object]` parameters); fix the rest that `ty` lists.
- [ ] **Step 5:** `uv run ty check src scripts tests` reports 0 diagnostics; make the hook blocking (drop `|| true`, check `src scripts tests`), then `make check` passes. Commit `fix: make the rendered project pass ty`.

### Task 3: Take the fix here

- [ ] **Step 1:** `uvx copier update --trust` to the fixed template version; `uv run ty check src scripts tests` is clean except for this repo's own files (the book skills' `references/notebook-skeleton.py` is a placeholder skeleton, not template code).
- [ ] **Step 2:** `make check` passes; mark this plan `done`.
