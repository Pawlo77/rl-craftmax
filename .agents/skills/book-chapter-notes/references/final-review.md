# Final review and fact-check

Two reviews follow every note: your own consistency pass (A) and an independent fact-check (B).
Run A as the last act of writing, before B. Do not skip A because the note "feels done".

## A. Self-review (read the whole note top to bottom first)

Read as a human who has never opened the chapter. Fail any item → fix → re-read the whole note once more.

1. **No repeated information.**
   - List each fact once with its section; any fact appearing in two sections is cut from the
     weaker one (or replaced by a name-level reference).
   - Typical offenders: exploration/convergence/limit statements repeated in Approach, practice
     and limitations; a definition restated in Math; "Where it matters" echoing Problem;
     limitations repeating earlier caveats.
2. **Logical structure.**
   - Every concept is introduced before it is used; Problem → Examples → Framework → Approach →
     Math → Dataset → Practice → Limitations flows without forward dependencies.
   - Each section does one job; subsections group related methods; nothing sits in a section
     whose heading it does not match.
3. **Readability for a human.**
   - Bullets are one idea each (about two sentences at most); no wall of nested bullets deeper
     than two levels; tables are narrow enough to read.
   - Every bullet is understandable without the book open; no dangling "this", "the above",
     "as before".
   - Sentences are plain: no stacked clauses, no jargon that the note has not explained.
4. **All terms explained.**
   - Every technical term has a definition at first use in this note, or a link to the chapter
     note that defines it.
   - Abbreviations are spelled out at their first body mention (TL;DR counts), e.g.
     "temporal-difference (TD)", "Markov decision process (MDP)".
   - Every symbol in Math appears in the notation table; every table symbol is used.
5. **Content discipline.**
   - No page, section, figure, equation or exercise numbers; no history; no citations; no
     narration of the book ("the chapter shows").
   - The examples overview is high level and consistent with what the chapter covers; the
     Dataset example rows are valid (hand-check boards, returns, probabilities) and labelled
     illustrative unless printed in the book.
   - Own derivations and own arithmetic are labelled ("formalized here").
6. **Consistency across notes.**
   - Symbols match earlier chapter notes (`S_t`, `A_t`, `R_t`, `G_t`, `v_\pi`, `q_\pi`, `\alpha`,
     `\gamma`, `\lambda`); deviations are stated.
   - Links point to notes that exist or are in the batch; the `parent` link names the main note.
7. **Format.**
   - Frontmatter valid and complete; headings in skeleton order; math delimiters correct
     (no `\tag`, `\label`, `\eqref`); Mermaid labels quoted; callouts used as specified.

Report: which checks needed edits, and what you changed. If nothing needed editing say so, but
only after genuinely re-reading.

## B. Independent fact-check

Spawn a read-only reviewer (`note-fact-checker` if present, otherwise a general-purpose agent
told not to edit). Prompt template:

> Check one Obsidian note against its source PDF. Note: `<note path>`. PDF:
> `research/Reinforcement Learning An Introduction/sources/SuttonBartoIPRLBook2ndEd.pdf`. The note covers printed pages `<first>-<last>`
> (PDF index = printed + 13; render with `.claude/skills/book-chapter-notes/scripts/pages.py` or
> any renderer). Verify every claim, number, algorithm, update rule and equation against rendered
> pages (the text layer scrambles math). Rows in the Dataset example are labelled illustrative:
> check only that they are internally valid and that the note says so. Flag: claims the chapter
> does not support or contradicts, wrong notation, wrong numbers, and important chapter content
> the note omits. The note intentionally has no page/section/figure numbers, no history, and a
> neutral "Where it matters in practice" section. Report a short table: note line, issue, PDF
> evidence. Do not edit anything.

Apply fixes for confirmed mismatches; for "unclear" or "derived" items either add the missing
support, label them ("derived here"), or delete them. Then, if more than about five edits were
made, run A once more.
