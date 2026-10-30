---
title: "Template agent setup implementation plan"
type: plan
domain: scripts
parent: "[[wut_murl_craftax]]"
status: done
created: 2026-10-20
updated: 2026-10-20
tags: [plan, template, bug]
---
# Template Agent Setup Implementation Plan

Status: done

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Fix an upstream issue in `Pawlo77/python-kit-template` v0.13.0: when an agent creates a repo from the template, or clones a repo made from it, it never runs a proper setup. It does not run `make install` (env, prek hooks), copy the Obsidian settings, run `make obsidian-plugins` or run the tests. The agent should detect a repo that was never set up and finish the setup before other work. This is a template issue recorded here because the template repo is private to another owner and no GitHub issue can be opened.

**Architecture:** Setup is written down only for people: `README.md` (`make install`) and `CONTRIBUTING.md` ("Development setup", "Obsidian vault setup"). Neither `AGENTS.md`, the skills nor `scripts/hooks/session_start.sh` mention it. The hook reports branch, changes, commits and the vault check, but not whether the repo was ever set up. Add one idempotent `make setup`, have the session-start hook detect missing setup and tell the agent to run it, and list it first in `AGENTS.md`.

**Tech Stack:** make, bash (`scripts/hooks/session_start.sh`), Python, prek, Obsidian.

---
### Task 1: Reproduce

- Render as in [[2026-10-20-template-knowledge-eof-loop]] Task 1 (or `git clone` a repo made from the template) and start an agent session (Claude Code, Codex or Cursor) with a task.
- The session-start context says `exit criterion: make check` but nothing about setup. The agent works without `.venv` and git hooks, `.obsidian/*.json` stay missing or empty, no community plugins are downloaded, and nothing runs the tests until `make check` at the end.
- Workaround used here: run `make install`, copy `.obsidian/*.example.json` over the live files, run `make obsidian-plugins` and `make check` by hand.

### Task 2: One setup target

**Files:**
- Modify: `template/kit.mk`, `template/README.md`, `template/CONTRIBUTING.md`

- [x] **Step 1:** Add `make setup`: `install`, then (vault scaffold) seed the Obsidian settings and `obsidian-plugins` (see [[2026-10-20-template-obsidian-onboarding]] Task 2), then `make tests`. Every step skips work already done, so rerunning is cheap.
- [x] **Step 2:** Point README's quick start and CONTRIBUTING's setup sections at `make setup`.

### Task 3: Detect a repo that was never set up

**Files:**
- Modify: `template/scripts/hooks/session_start.sh`
- Test: `template/tests/hooks/test_session_start.py`

- [x] **Step 1: Write the failing test:** with no `.venv`, no installed prek hook in `.git/hooks`, or (vault scaffold) a missing or empty `.obsidian/app.json` or `community-plugins.json`, or plugins missing from `.obsidian/plugins/`, the session-start context includes `- setup: incomplete (<what is missing>); run make setup before other work`. A repo that was set up shows `- setup: ok`.
- [x] **Step 2:** Implement the checks in `state()`; keep them filesystem-only (no `uv run`) so the hook stays fast.

### Task 4: Tell the agents

**Files:**
- Modify: `template/AGENTS.md`, the `development` skill

- [x] **Step 1:** Make `AGENTS.md` start with: on a new render or fresh clone, or when session start reports `setup: incomplete`, run `make setup` before anything else and report its result.
- [x] **Step 2:** Make the `development` skill's first step run `make setup` when setup is incomplete.
- [x] **Step 3:** Fresh render and fresh clone, each started in a new agent session: the agent runs `make setup` unprompted, and env, git hooks, Obsidian settings, plugins and tests are all in place; `make check` passes. Commit `feat: detect and run project setup in new agent sessions`.

### Task 5: Take the fix here

- [x] **Step 1:** `uvx copier update --trust` to the fixed template version.
- [x] **Step 2:** A fresh clone of this repo in a new agent session sets itself up; `make check` passes; mark this plan `done`.
