# Notebook final review

Run after the notebook executes cleanly and its *Reading the result* text is written. Open the
scratch `.py` and the executed `.ipynb`; read top to bottom as a reader who has not seen the chapter. Any
failed item: fix, re-run the notebook, re-read.

1. **Format identical to the skeleton.**
   - Jupytext header; title cell with scope sentence, link to the chapter note, contents list.
   - Setup is the first code cell and holds every import and constant; `grep -n "^import\|^from"`
     finds nothing below it.
   - Each section: what/why markdown, implementation, seeded experiment with checkpoint prints,
     reading-the-result markdown; final checkpoints cell with asserts and a table.
2. **Correct against the book.**
   - Each algorithm matches the book's box or equation (initialization, update, termination).
   - Example parameters (sizes, step sizes, exploration rates, runs) match the book's description,
     or the deviation is stated in the markdown.
   - Claims about the book's results match the PDF; if the notebook disagrees with the book, the
     markdown says so plainly.
3. **Prose and output agree.** Every number in markdown equals the printed output of the last run
   (check by search); no stale values; nothing claimed that no cell verifies.
4. **Readability.**
   - Markdown is plain: one idea per bullet, short paragraphs, no jargon left undefined, every
     abbreviation spelled out at first mention, no history or citations or numbering references.
   - No explanation repeated in two sections; each concept explained once where it is introduced.
   - Code: no comment restates a name or the structure; descriptive names matching the notation, docstrings state what is estimated, no cell
     over about 40 lines, no commented-out code, no dead helpers.
5. **Reproducible and fast.** One fresh-kernel run under about 60 seconds; per-section seeded
   generators; two runs print identical numbers; no network, files, magics or installs.
6. **Plots.** Titles, labelled axes, legends, readable size; each plot is referred to in the
   following markdown.
7. **Checkpoints.** Assertions encode the claims (truth within tolerance, ordering of methods);
   they fail if the implementation is broken.

Report which checks needed edits and what changed. Say "nothing needed editing" only after a real
second read.
