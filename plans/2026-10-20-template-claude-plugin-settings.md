---
title: "Template Claude plugin settings implementation plan"
type: plan
domain: scripts
parent: "[[wut_murl_craftax]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, template, bug]
---
# Template Claude Plugin Settings Implementation Plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a minor upstream issue in `Pawlo77/python-kit-template` v0.13.0: `claude plugin install <plugin> --scope project` rewrites the generated `.claude/settings.json` with its keys reordered (`enabledPlugins` and `extraKnownMarketplaces` moved after `hooks`). The file then differs from `scripts/sync_configs.py` output, and the `agent tool configs match scripts/sync_configs.py` hook (`sync_configs.py --check`) fails. This is a template issue recorded here because the template repo is private to another owner and no GitHub issue can be opened.

**Architecture:** `sync_configs.py` compares the file text with `json.dumps(...)` byte for byte, so a key order change counts as stale. The plugin list comes from `[tool.agent-kit.configs]` `plugins` and `marketplaces` in `pyproject.toml`, so `make configs` alone also drops a plugin that was only installed through the CLI. Compare JSON outputs as parsed data in `--check`, and document `pyproject.toml` as the one place to add plugins.

**Tech Stack:** Python (`scripts/sync_configs.py`), Claude Code CLI, prek.

---
### Task 1: Reproduce

- Fresh render, `claude plugin install <plugin> --scope project`, then `uv run python scripts/sync_configs.py --check`: `stale: .claude/settings.json`.
- Workaround used here: restore the file, or run `make configs`. Add the plugin to `[tool.agent-kit.configs] plugins` first, or `make configs` removes it again.

### Task 2: Fix in the template

**Files:**
- Modify: `template/scripts/sync_configs.py` (`main`), the agent docs that describe plugins (`AGENTS.md` or the configs rule)
- Test: the `sync_configs` hook test

- [ ] **Step 1: Write the failing test:** a `.claude/settings.json` with the same content and different key order passes `--check`; a missing plugin still fails.
- [ ] **Step 2:** In `--check`, compare `json.loads` of `.json` outputs; keep the text comparison for `.toml`. The write mode still writes the canonical text.
- [ ] **Step 3:** Document adding a plugin: add it to `[tool.agent-kit.configs] plugins` (and `marketplaces` when it comes from a new one), then `make configs`; do not use `claude plugin install --scope project`.
- [ ] **Step 4:** `make check` passes. Commit `fix: compare agent configs as data in sync_configs --check`.

### Task 3: Take the fix here

- [ ] **Step 1:** `uvx copier update --trust` to the fixed template version; move any CLI-installed plugins into `[tool.agent-kit.configs] plugins`, then `make configs`.
- [ ] **Step 2:** `make check` passes; mark this plan `done`.
