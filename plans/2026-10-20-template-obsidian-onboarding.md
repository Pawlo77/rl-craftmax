---
title: "Template Obsidian onboarding implementation plan"
type: plan
domain: vault
parent: "[[wut_murl_craftax]]"
status: draft
created: 2026-10-20
updated: 2026-10-25
tags: [plan, template, bug]
---
# Template Obsidian Onboarding Implementation Plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two upstream usability issues in `Pawlo77/python-kit-template` v0.13.0 (vault scaffold). The experiment folders `configs/`, `outputs/`, `notebooks/` and the MLflow store `mlruns/` are not hidden from the vault (Task 3). And `make install` neither copies `.obsidian/*.example.json` to the live files nor runs `make obsidian-plugins`. If the vault is opened in Obsidian first, the documented copy step keeps Obsidian's empty files, so the vault starts with no community plugins, ignore filters, templates folder or CSS snippet. These are template issues recorded here because the template repo is private to another owner and no GitHub issue can be opened.

**Architecture:** On first open, Obsidian writes `app.json` (`{}`), `appearance.json` (`{}`), `community-plugins.json` (`[]`) and its own `core-plugins.json`. CONTRIBUTING's "Obsidian vault setup" copies with `cp -n` ("existing files are kept"), so the defaults never land. Seed the live files during `make install` (fresh clone) and replace only files that are missing or still Obsidian's empty defaults, so real local settings survive.

**Tech Stack:** make, Python (`scripts/install_obsidian_plugins.py`), Obsidian.

---
### Task 1: Reproduce

- Render as in [[2026-10-20-template-knowledge-eof-loop]] Task 1 with `vault=true`, open the folder as a vault in Obsidian, then run the CONTRIBUTING step `find .obsidian -name '*.example.json' -exec sh -c 'cp -n "$1" "${1%.example.json}.json"' _ {} \;`.
- `app.json`, `appearance.json` and `community-plugins.json` stay empty; no plugins load.
- Workaround used here: copy the examples over with plain `cp`, run `make obsidian-plugins`, reload Obsidian.

### Task 2: Fix in the template

**Files:**
- Modify: `template/kit.mk` (`install`, new `obsidian-init` target), `template/CONTRIBUTING.md` ("Obsidian vault setup")
- Create or modify: a small seeding script (or a mode of `scripts/install_obsidian_plugins.py`)
- Test: a hook test that seeds a temp `.obsidian/` holding Obsidian's empty files

- [ ] **Step 1: Write the failing test:** for a live file that is missing, `{}` or `[]`, seeding writes the example; a live file with real content is kept.
- [ ] **Step 2:** Implement `make obsidian-init` (seed the live files, then `make obsidian-plugins`) and call it from `make install` when the vault is enabled.
- [ ] **Step 3:** Rewrite the CONTRIBUTING section: run `make install` before opening the vault, or `make obsidian-init` and reload Obsidian if it was opened first.
- [ ] **Step 4:** Fresh render, open Obsidian first, then `make install`: plugins, ignore filters, templates folder and snippet are active after a reload; `make check` passes. Commit `fix: seed the Obsidian settings on install`.

### Task 3: Hide the experiment folders in the template

The experiment scaffold renders `configs/`, `outputs/` and `notebooks/`, but neither `userIgnoreFilters` in `.obsidian/app.example.json` nor `.obsidian/snippets/hide-non-notes.css` lists them, so Obsidian shows run configs, experiment outputs and executed notebooks in the file explorer, search and graph. Fixed here on 2026-10-20 by adding the three folders to both (and to the live `app.json`). The MLflow store has the same problem: the experiment scaffold tracks runs in `mlruns/` (artifacts, and `mlartifacts/` for a tracking server), thousands of run folders that flood the file explorer and search. Fixed here on 2026-10-25 by adding `mlruns/` and `mlartifacts/` to both.

**Files:**
- Modify: `template/.obsidian/app.example.json`, `template/.obsidian/snippets/hide-non-notes.css` (only when the experiment scaffold renders these folders)

- [ ] **Step 1:** Add `"configs/"`, `"outputs/"`, `"notebooks/"`, `"mlruns/"`, `"mlartifacts/"` to `userIgnoreFilters` and a `.nav-folder:has(> .nav-folder-title[data-path="<folder>"])` rule for each to the snippet.
- [ ] **Step 2:** Add a hook test that every top-level non-note folder of the render is listed in both files, so the two lists cannot drift.
- [ ] **Step 3:** Fresh experiment render opened in Obsidian: none of the five folders shows in the file explorer, search or graph; `make check` passes. Commit `fix: hide experiment folders from the Obsidian vault`.

### Task 4: Take the fix here

- [ ] **Step 1:** `uvx copier update --trust` to the fixed template version.
- [ ] **Step 2:** `make check` passes; mark this plan `done`.
