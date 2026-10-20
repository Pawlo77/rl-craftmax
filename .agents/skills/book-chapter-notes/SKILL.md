---
name: book-chapter-notes
description: 'Write detailed Obsidian study notes for one chapter (or sub-chapter group) of the Sutton & Barto "Reinforcement Learning: An Introduction" 2nd-edition draft PDF in this research/reinforcement-learning-an-introduction/ folder, and maintain the main book note. Use whenever asked to take notes on, summarize, or analyze a chapter or section of this book, or to write or refresh the main book index note.'
---

# Book chapter notes

One note per chapter (or per sub-chapter group for long chapters), written from the PDF, then
reviewed twice before it counts as done. The main note ties the chapters together.

The book lives in `research/reinforcement-learning-an-introduction/`; chapter notes link only to notes inside that folder. Read pages with
the helper in this skill (`scripts/pages.py`).

## Files

```
research/reinforcement-learning-an-introduction/
  sources/SuttonBartoIPRLBook2ndEd.pdf          the source of truth
  chapters/<NN>[a|b] <Title>.md                 chapter notes (style reference: chapters/01 Introduction.md)
  Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft).md   main note
  sources/SuttonBartoIPRLBook2ndEd/…md          Marker snapshot: navigation aid ONLY, never a source
```

The Markdown snapshot has OCR and math errors; use it to find sections, never to copy a number,
equation or sentence. Everything quoted comes from the PDF.

## Chapter map and split plan

Printed pages; the PDF index is printed page + 13. Contents are on PDF indexes 2-6.

| Note file | Covers | Printed pages |
|-----------|--------|---------------|
| 01 Introduction | Ch 1, sections 1.1-1.6 (skip 1.7 history, 1.8 remarks) | 1-16 |
| 02 Multi-armed bandits | Ch 2 | 31-52 |
| 03a MDP setting and returns | 3.1-3.5 | 53-66 |
| 03b MDPs, value functions, optimality | 3.6-3.10 | 67-88 |
| 04 Dynamic programming | Ch 4 | 89-112 |
| 05a Monte Carlo prediction and control | 5.1-5.4 | 113-126 |
| 05b Off-policy Monte Carlo | 5.5-5.9 | 127-142 |
| 06 Temporal-difference learning | Ch 6 | 143-166 |
| 07a n-step TD and TD(λ) | 7.1-7.4 | 167-182 |
| 07b Eligibility-trace control and off-policy | 7.5-7.10 | 183-194 |
| 08a Models, planning, Dyna | 8.1-8.5 | 195-212 |
| 08b Trajectory sampling, search, MCTS | 8.6-8.9 | 213-222 |
| 09a Function approximation and linear methods | 9.1-9.3 | 225-240 |
| 09b Control with approximation, bootstrapping | 9.4-9.6 | 241-254 |
| 11 Policy approximation (actor-critic) | Ch 11 | 257-264 |
| 14.1-14.6 TD-Gammon, Samuel's checkers, Acrobot, elevator dispatching, dynamic channel allocation, job-shop scheduling | one short note each | 273-302 |
| 15 The unified view | 15.1 only | 303-305 |

Not noted separately: Ch 10, 12, 13 (two-page placeholders) and 15.2-15.5. Give each a one-line row
in the main note. Chapter 1's section 1.7 (history) is skipped under the no-history rule.
Section boundaries inside a chapter come from the contents pages; confirm on the PDF.

Chapter and section numbers of this draft differ from the final book; use the draft's.

## Workflow for one note

1. **Locate** the chapter pages. Read the whole body, skim the reference/bibliographic remarks only
   for facts the body lacks.
   - Text: `uv run --no-project --with pypdfium2 --with pillow python -I <skill-dir>/scripts/pages.py text research/reinforcement-learning-an-introduction/sources/SuttonBartoIPRLBook2ndEd.pdf <first> <last>`
   - Renders: same with `render … --out <scratchpad>/pages` (then view the PNGs).
2. **Take equations, numbers, algorithm boxes and table values from renders only**; the text layer
   scrambles math. Unreadable → say so in the note; never fill in from memory. Quote the
   book's notation (`V(s)`, `S_t`, `\pi`) exactly as the chapter does.
3. **Plan before writing**: list every fact you will state and assign it exactly one home section
   (the fact → section map). A fact needed elsewhere is referred to by name, not restated.
4. **Write** from `references/chapter-note-skeleton.md`, following the rules below. Chapter 1's note
   is the style and depth reference.
5. **Final self-review** (mandatory, last thing you do): re-read the note top to bottom as a human
   reader who has not seen the chapter, run every check in `references/final-review.md`, fix what
   fails, and re-read once more. State in your report which checks needed fixes.
6. **Independent fact-check**: spawn a read-only reviewer agent (`note-fact-checker` if the
   environment has it, else a general-purpose agent) with the note path, PDF path and printed page
   range, using the prompt in `references/final-review.md`. Fix confirmed mismatches; delete or
   label unsupported claims ("derived here: …"). If more than about five edits were needed,
   run the self-review once more.

Parallel runs: one subagent per note; give each this SKILL.md path, the note's row in the table,
the printed page range and the output path. Notes may link only to notes that exist or are in the
same batch.

## Requirements for every chapter note

**Structure**
- Sections in this order, from the skeleton: TL;DR, Problem, Examples in the chapter (when it has
  worked or running examples), Conceptual framework, Approach, Math, Dataset example, Where it
  matters in practice, Assumptions & limitations. Omit a section only if it has nothing to say;
  never invent filler.
- Textbook chapters have no Validation or Data sections.
- Sub-chapter notes (a/b) are self-contained: define in b the terms b needs, linking to the a
  note for the full definition.

**Content**
- **Examples overview**: when the chapter uses examples (running or worked ones), give a high-level
  table: example, setting/decision, what it demonstrates. Do not reproduce the example's details.
- **Where it matters in practice** is a neutral section (no marketing or attribution framing, no
  industry advice): signs a problem fits, what you need to use it (data, reward, state, model,
  compute), when not to trust it. Only content the chapter supports, not restated from earlier sections.
- **Dataset example**: one lead line naming the grain, an "illustrative rows" table of 3-6 rows
  with the columns the method consumes in the book's variable names (include a terminal or failure
  row when the method needs it), then 1-3 bullets (column → symbol, what is derived, what is
  absent). Rows the book prints are copied from renders; invented rows are labelled illustrative
  and must be internally valid (check boards, returns, probabilities by hand).
- **Math**: the book's notation plus a notation table; each equation gets a bold descriptive label
  (no number) and a plain reading; for every algorithm state its update rule and what it
  estimates; at most one worked `[!example]` (label your own arithmetic "formalized here"); at most
  one Mermaid diagram, after the notation table, only when words hide the structure.

**Writing rules** (adapted from the repo's paper-notes skill)
- Every sentence carries a mechanism, assumption, formula, number or decision implication. Cut
  narration ("the chapter presents"), framing and process material.
- One idea per bullet, bold lead term where useful; split anything over about two sentences. The
  TL;DR is 1-3 sentences, with at most one number.
- No history (who invented what, when), no citations of other works.
- No source pointers: no page, section, figure, table, exercise, equation or theorem numbers.
  State the content and refer back by name.
- Spell out every abbreviation at its first body mention, TL;DR included: "temporal-difference
  (TD)". Headings, frontmatter and link text are exempt.
- Define every term at first use, in the note or by link to the chapter note that defines it.
  The reader should not need the book open.
- Fix obvious typos of the draft silently; mention a draft typo in the note only if it changes
  meaning. Draft quirks (placeholder references, inconsistent numbering, missing figure content)
  go in the closing report, not the note.
- Notation stays consistent across notes: reuse the symbols already fixed in earlier chapter notes
  (state `S_t`, action `A_t`, reward `R_t`, return `G_t`, value `v_\pi`, `q_\pi`, step size `\alpha`,
  discount `\gamma`, trace decay `\lambda`) and say so when the chapter deviates.

**Obsidian formatting**
- Frontmatter: `title`, `parent: "[[<main note title>]]"`, `type: book-chapter`, `chapter`,
  optional `section_range`, `authors`, `venue`, `pdf: "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf"` (the public URL, not the local copy; `sources/` is only for reading pages),
  `tags`, `created`. No `related-code`, no repo links.
- Every chapter note MUST carry `parent: "[[Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft)]]"`
  (the main note) and the public `pdf` URL above; the main note must list and link every chapter note.
  After a batch the parent checks all notes for both fields (`grep -L '^parent:' chapters/*.md`).
- Math with `$…$` and `$$…$$` on their own lines; no `\tag`, `\label`, `\eqref`.
- Mermaid `flowchart LR`, quoted labels with spaces, no LaTeX in labels.
- Callouts: `[!abstract]` TL;DR, `[!example]` worked example, `[!warning]` necessary caveat.
- File name = `<NN>[a|b] <Title>.md`; no `:/#^[]|` in names.

## The main note

Write it last, after every chapter note exists. Name:
`Sutton & Barto - Reinforcement learning, an introduction (2nd ed draft).md`, frontmatter
`type: book`, `tags: [book, reinforcement-learning, moc]`.

1. **One scope paragraph**: what the book covers and its three-part shape (tabular methods,
   approximate methods, frontiers); note that it is a draft with placeholder chapters.
2. **Chapter table**: columns chapter, topic, printed pages, a short gist (one sentence), and the
   link to the chapter note. Rows for the skipped placeholder chapters carry no link and say so.
3. **Crucial parts**: a section linking every key idea and algorithm to the note that explains it
   (Markov decision process and Bellman equations, policy and value iteration, Monte Carlo
   estimation and importance sampling, TD(0), Sarsa, Q-learning, n-step and TD(λ), eligibility
   traces, Dyna and planning, function approximation and gradient methods, actor-critic). One
   line each: what it is, why it matters, link. Wikilinks to headings where useful.
4. **How the parts fit**: the dependency story between parts (what each builds on), in a few bullets;
   no repeats of the table.
5. The same final self-review applies (no repeated information, defined terms, logical order);
   the main note must not duplicate chapter content, only point to it.

## Report when finished

Notes written, with line counts; checks that needed fixes in the self-review; fact-check findings
and what was fixed; anything unverifiable (unreadable math, draft inconsistencies, missing figure
content); skipped chapters; open questions.
