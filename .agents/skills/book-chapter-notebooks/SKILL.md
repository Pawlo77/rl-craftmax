---
name: book-chapter-notebooks
description: 'Create a runnable Jupytext notebook in research/reinforcement-learning-an-introduction/notebooks/ that implements and demonstrates everything algorithmic in one chapter (or sub-chapter group) of the Sutton & Barto "Reinforcement Learning: An Introduction" 2nd-edition draft. Use whenever asked to implement, code up, simulate, or build a notebook for a chapter of this book.'
---

# Book chapter notebooks

One notebook per chapter note, same name and numbering: `research/reinforcement-learning-an-introduction/chapters/<NN>[a|b] <Title>.md` →
`research/reinforcement-learning-an-introduction/notebooks/<NN>[a|b] <Title>.ipynb`, executed, with outputs. Only the `.ipynb` is kept in
`research/reinforcement-learning-an-introduction/notebooks/`; author the Jupytext percent `.py` in a scratch directory and delete nothing else. The notebook implements what the chapter describes in a
small, readable, reproducible form and checks it against known truth or the book's stated results.

The book lives in `research/reinforcement-learning-an-introduction/`; notebooks rely only on the helpers inside this skill and on the
chapter notes.

## Inputs

- The chapter note `research/reinforcement-learning-an-introduction/chapters/<name>.md` (the notebook's section order follows its Approach and
  Math sections, and its notation). If it does not exist, write it first with `book-chapter-notes`.
- The PDF `research/reinforcement-learning-an-introduction/sources/SuttonBartoIPRLBook2ndEd.pdf`, read with
  `.agents/skills/book-chapter-notes/scripts/pages.py` (printed page numbers). Algorithm boxes,
  hyperparameters, example setups and result descriptions come from page renders, not from memory.
- The format skeleton `references/notebook-skeleton.py` and the checklist `references/notebook-review.md`.

## What to implement

- **Every algorithm the chapter states** (update rules, pseudocode boxes), in its notation, as a
  small function or class with a docstring saying what it estimates.
- **The chapter's running examples** at toy scale (the example gets its own section, built from
  the setup the book describes: states, rewards, parameters). Where the book describes a result
  (a figure's qualitative shape, a quoted number), reproduce it at reduced scale and say in the
  markdown whether the notebook agrees. Never tune parameters until the picture matches.
- **Known truth**: wherever a method estimates something computable exactly (a value function by
  linear solve or dynamic programming, a bandit's true best action), compute the truth and compare.
- **A baseline or contrast** when the chapter argues for a method against an alternative
  (naive vs improved, one method family vs another), so the notebook shows why the method exists.
- A chapter that is only conceptual still gets the notebook of its worked example (Chapter 1 =
  the tic-tac-toe learner).
- Sub-chapter notebooks (a/b) are self-contained: copy any small helper b needs from a.

## Format (identical in every notebook; see the skeleton)

1. **Jupytext header** (`formats: py:percent`, kernel `python3`).
2. **Title cell** (markdown): `# <Chapter title>: implementation`, one sentence of scope, a link to
   the chapter note (`../chapters/<name>.md`) and a contents list of the sections below.
3. **Setup cell** (code, always the first code cell): **all imports**, matplotlib defaults, global
   constants (`SEED`, sizes, step sizes), nothing else. No imports anywhere else.
   Allowed libraries: standard library, `numpy`, `matplotlib`, `scipy`, `pandas`. Nothing else.
4. **Sections**, one per method or example, each with the same pattern:
   - markdown *What and why*: the idea in plain words, the update rule or equation in the book's
     notation (`$…$`), what it estimates, what problem it solves;
   - code *Implementation*: small functions, type hints, docstrings, vectorized, no global state
     except the setup constants, each cell at most ~40 lines;
   - code *Experiment*: a seeded run (`rng = np.random.default_rng(SEED_<SECTION>)`; one named seed per
     section; never share a generator across sections), with a plot when it clarifies, and
     `print("Checkpoint: …")` lines that state what was verified;
   - markdown *Reading the result*: what the output shows, with numbers that match the executed
     output exactly.
5. **Final cell**: a summary of checkpoints (`assert` statements against known truth or
   tolerances, then a short table of method, result, truth) so a silent regression fails the run.
6. Plots: labelled axes, title, legend, `plt.show()`; at most 2 figures per cell; figures render at
   a size readable in a notebook (about 7x4 inches).

## Writing rules for the markdown (same as the chapter notes)

- Teach the method, not the literature: no history, no citations, no page/section/figure/equation
  numbers; refer back by name.
- Spell out every abbreviation at its first mention in the notebook; define every term at first use
  or link to the chapter note that defines it.
- Plain sentences, one idea per bullet, no walls of text; number only what the output confirms.
- Say where the toy differs from the book (scale, number of runs, simplified opponent) and why.
- Notation matches the chapter notes (`S_t`, `A_t`, `R_t`, `G_t`, `v_\pi`, `q_\pi`, `\alpha`, `\gamma`, `\lambda`);
  code names follow it (`s`, `a`, `r`, `V`, `Q`, `alpha`, `gamma`, `lam`).

## Code rules

- Comments only where the code cannot say it (a non-obvious choice, a deviation from the book, a unit). No
  comments that restate a name, a cell's purpose or the structure (for example "# Setup: imports").
- Performance first, then readability, compact code (the repo's python-style spirit): vectorize
  with numpy, avoid needless classes, no dead code, no print debugging left behind.
- Deterministic: all randomness from seeded `numpy.random.Generator`s created in the cell that
  uses them; two runs give identical outputs.
- Budget: the whole notebook runs top to bottom in a fresh kernel in under about 60 seconds; scale
  runs down (fewer episodes/runs) and say so, rather than waiting.
- No network, no files read or written, no `%pip`, no notebook magics.

## Workflow

1. Read the chapter note and the chapter's algorithm boxes and example descriptions in the PDF.
2. Plan sections: a table of section, what is implemented, truth or baseline, checkpoint.
3. Write the notebook from the skeleton as a percent `.py` in a scratch directory (not in `research/reinforcement-learning-an-introduction/notebooks/`).
4. **Execute**:
   ```bash
   uv run --no-project --python 3.12 --with jupytext --with ipykernel --with jupyter-client \
     --with nbformat --with numpy --with matplotlib --with scipy --with pandas \
     python -I .agents/skills/book-chapter-notebooks/scripts/run_notebook.py \
     "<scratch>/<name>.py" --out "research/reinforcement-learning-an-introduction/notebooks/<name>.ipynb" --print
   ```
   It runs a fresh kernel, writes the executed `.ipynb` to `--out`, stops at the first error, and prints
   each cell's text output so the markdown can be checked against it.
5. Write the *Reading the result* text from the real output (rewrite prose to fit the output, never
   the reverse); run again; confirm identical numbers.
6. **Final review** (mandatory, last step; `references/notebook-review.md`): imports only in the
   setup cell, format identical to the skeleton, prose and code consistent, all terms explained,
   no repeated explanations, numbers in markdown equal the output, runtime within budget, seeds
   fixed, assertions pass. Fix and re-run; report which checks needed edits.
7. **Independent check**: spawn a read-only reviewer against the chapter's PDF pages and the chapter
   note: are the algorithms implemented as in the book's boxes, are hyperparameters those of the
   book's examples, do claims about the book's results match it. Fix confirmed mismatches.

Parallel runs: one subagent per notebook, given this SKILL.md path, the chapter note, the page range
and the output path.

## Report when finished

Notebook (`.ipynb`) path, runtime, sections and checkpoints, where the toy deviates from the book,
results that disagree with the book (stated plainly), anything unverifiable, review findings fixed.

## Lessons from the first independent reviews (apply before you write)

Every first draft so far failed the same checks; avoid them up front.

- **Checks must test something independent.** Comparing a quantity with itself (the same recursion on two paths, a "truth" computed by the same code, a coding identity) proves nothing: label such checks "consistency of two code paths" or replace them with an independent route (a linear solve, a closed form, simulation, an exhaustive computation).
- **Tolerances are derived, not guessed.** Derive each from the sample size or the stopping rule and say how in the markdown; a tolerance that would also pass a wrong implementation is a bug. For many simultaneous comparisons use a Bonferroni-style limit or a share-within-limit rule, never `all(|z| < 3)`.
- **Claims vs evidence.** Quote a crossing point, a "settles", "stays" or "monotone" statement only when a printed number supports it (print counts at intervals); single-run crossings are "rough". Every explanation you did not test is labelled "explanation, not tested". Every comparison that differs by less than about two standard errors is not a finding; add standard errors (paired where methods share tasks).
- **Same random draws.** Methods compared on the same task must see the same draws (shared seed or paired generators), or say they do not.
- **Seeds.** One named seed constant per section; derive several generators from it with `np.random.SeedSequence(SEED).spawn(k)`; never `SEED + 1`-style offsets across sections.
- **Say where you differ from the book**: every rule the book leaves open, every tie-break, every in-place vs two-array choice, every parameter you invented, every scaled-down count, each stated once in the section where it applies.
- **Book results.** State plainly where the notebook agrees or disagrees with the book's figures or numbers, and say figure comparisons are by eye.
- **Not implemented** topics of the chapter (figure-only examples, stubs, unproven claims) get one sentence in the title cell saying why.
- **Format traps**: all constants (counts, tolerances, snapshot indices, run counts) in the setup cell; cells at most about 40 lines; refer back by section name, not number; define "standard error", "p-value", and every symbol at first use; type hints on functions; no restating comments; no `np.float64` in printed output.
