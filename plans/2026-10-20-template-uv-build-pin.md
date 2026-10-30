---
title: "Template uv-build pin implementation plan"
type: plan
domain: build
parent: "[[wut_murl_craftax]]"
status: done
created: 2026-10-20
updated: 2026-10-20
tags: [plan, template, bug]
---
# Template uv-build Pin Implementation Plan

Status: done

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Fix an upstream issue in `Pawlo77/python-kit-template` v0.13.0: the rendered `pyproject.toml` pins `uv_build>=0.11.0,<0.12.0`, behind the uv in use (0.12.23), so every `uv sync` and `uv run` that builds the project warns. Harmless, but it is noise on every command and the pin goes stale with each uv minor release. This is a template issue recorded here because the template repo is private to another owner and no GitHub issue can be opened.

**Architecture:** uv warns when `[build-system].requires` excludes the running uv version. Bump the range in the template's `pyproject.toml` and keep it current, for example with Renovate/Dependabot or the template's `scripts/upgrade.py`.

**Tech Stack:** uv 0.12.23, uv_build, copier.

---
### Task 1: Reproduce

- With uv 0.12.23 installed, render as in [[2026-10-20-template-knowledge-eof-loop]] Task 1, then `uv lock && make install`.
- Output on every sync: `warning: build_system.requires = ["uv-build>=0.11.0,<0.12.0"] does not contain the current uv version 0.12.23`.
- No workaround applied; the build works.

### Task 2: Fix in the template

**Files:**
- Modify: `template/pyproject.toml.jinja` (or wherever `[build-system]` is rendered)

- [x] **Step 1:** Set `requires = ["uv_build>=0.12.0,<0.13.0"]`.
- [x] **Step 2:** Have the dependency updater (or `scripts/upgrade.py`) bump the `uv_build` range with each uv minor release.
- [x] **Step 3:** Fresh render, `uv lock && make install` prints no `build_system.requires` warning; `make check` passes. Commit `build: track the current uv_build minor version`.

### Task 3: Take the fix here

- [x] **Step 1:** `uvx copier update --trust` to the fixed template version (or bump the range in `pyproject.toml` by hand), `uv lock`.
- [x] **Step 2:** `make install` is warning-free and `make check` passes; mark this plan `done`.
