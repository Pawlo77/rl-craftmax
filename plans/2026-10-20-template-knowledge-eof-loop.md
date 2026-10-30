---
title: "Template knowledge EOF loop implementation plan"
type: plan
domain: scripts
parent: "[[wut_murl_craftax]]"
status: in progress
created: 2026-10-20
updated: 2026-10-20
tags: [plan, template, bug]
---
# Template Knowledge EOF Loop Implementation Plan

Status: in progress (template change in Pawlo77/python-kit-template#13; open: take it here with `copier update`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Fix an upstream bug in `Pawlo77/python-kit-template` v0.13.0: a generated knowledge note with an empty table ends in a blank line, so `end-of-file-fixer` and `knowledge-build` undo each other on every run and `make check` never passes. This is a template issue recorded here because the template repo is private to another owner and no GitHub issue can be opened; fix it in the template, then here via `copier update`.

**Architecture:** `template/scripts/build_knowledge.py` builds each table as `header + "\n".join(rows) + "\n"`. With no rows (a fresh experiment's `__all__: list[str] = []`) the note ends in `"\n\n"`. Normalize once in `build_all` so every generator, and project `BUILDERS` from `load_extra`, ends in exactly one newline.

**Tech Stack:** Python 3.12, pytest, prek (`end-of-file-fixer`, `knowledge-build`), copier.

---
### Task 1: Reproduce and record the symptom

- Render: `uvx copier copy --trust --defaults --data scaffold=experiment --data vault=true --data slug=wut_murl_craftax gh:Pawlo77/python-kit-template wut_murl_craftax`, then `git init -b main && uv lock && make install && git add -A && make check`.
- `knowledge/generated/Public API.md` ends `|---|---|---|\n\n`. prek prints "Fixing knowledge/generated/Public API.md" (`end-of-file-fixer`), then "knowledge: 1 of 2 generated notes rewritten" (`knowledge-build`). Both fail on every run, so any commit touching `pyproject.toml` or `src/` fails.
- Same pattern: `build_modules` (empty package), `build_entry_points` (guarded: returns `""` with no rows), `build_dependencies` (never empty: always has the Python row).
- Workaround until fixed: `SKIP=end-of-file-fixer git commit ...`.

### Task 2: Fix in the template

**Files:**
- Modify: `template/scripts/build_knowledge.py` (`build_all`)
- Test: `template/tests/hooks/test_knowledge_engine.py`

- [x] **Step 1: Write the failing test**

```python
def test_notes_end_in_one_newline_when_a_table_is_empty(
    builder: ModuleType, project: Path
) -> None:
    _write(project, "src/demo/__init__.py", '"""Demo."""\n\n__all__: list[str] = []\n')
    notes = {path.stem: text for path, text in builder.build_all(project).items()}
    assert notes["Public API"].endswith("|---|---|---|\n")
    assert all(not text.endswith("\n\n") for text in notes.values())
```

- [x] **Step 2: Fix `build_all`**

```python
    # one trailing newline: an empty table ("\n".join([]) + "\n") would leave a
    # blank line that end-of-file-fixer strips and the next build restores
    return {
        path: finalize(path, text.rstrip("\n") + "\n")
        for path, text in notes.items()
        if text
    }
```

Verified on 2026-10-20 against this repo's rendered copy in a scratch directory: the new test fails before the change (1 failed, 7 passed) and passes after it (8 passed); `ruff format --check` and `ruff check` are clean. Not yet run: the template's own `make check`, and a fresh render through `make check`.

- [x] **Step 3:** In the template, run `make check`, render a fresh experiment as in Task 1 and confirm `make check` passes twice in a row with no rewrites. Commit `fix: end generated knowledge notes with one newline`, then release a patch version.

### Task 3: Take the fix here

- [ ] **Step 1:** `uvx copier update --trust` to the fixed template version; resolve conflicts in `scripts/build_knowledge.py` and its test.
- [ ] **Step 2:** `make knowledge && make check` passes without `SKIP=end-of-file-fixer`; mark this plan `done`.
