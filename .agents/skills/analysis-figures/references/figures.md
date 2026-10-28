# Figures reference

## Pipeline

- One theme helper builds rc parameters from a seaborn style and context (one report
  theme, optionally one video theme) and binds them to each figure with
  `matplotlib.rc_context`; a test checks that no global rcParams change.
- Save at a fixed dpi and an exact physical size (for example 80 mm column, 170 mm
  double). Fit the layout to the canvas (measure `fig.get_tightbbox`, shrink the layout
  rect, re-measure) instead of `bbox_inches="tight"`, which changes the size.
- Layout: `outputs/<results>/figures/<set>/<stem>.png`, stems without the set name; the
  report copy mirrors it and removes stale stems. Tables: `tables/<set>/`.
- Each figure stage writes `captions.csv` (stem, id, description) beside its PNGs; a make
  target turns all of them into one notes table (figure link | description), grouped by
  figure family. The report adds its own text later.
- Panels meant to sit together share size and axis limits so LaTeX can tile them.

## What may appear on the image

| Allowed | Not allowed |
|---|---|
| axis and tick labels, units | notes, explanations, p-values as annotations |
| short panel label (several panels) | titles on single-panel figures, wrapped titles |
| data values that are the plot | benchmark or budget name, "pending", "indicative" |
| a compact non-category legend outside the data | the category legend (use the legend figure) |

The standalone legend figure shows every category's colour, line style and marker, in
the column and the double width.

## Audit test (every figure kind, every theme variant)

- No text cut by the canvas edge.
- No legend over an axes; no text-text overlap, tick labels included.
- No layout `UserWarning`; no title wider than its axes.
- No category legend outside the legend figure.
- One data axes per file (colour bars excepted).
- No lone y tick label that names a quantity.
- Only allowed text kinds.

Make the audit fail on purpose once (force a legend inside, disable the fit step) to prove
it is not vacuous; keep its skip list empty.

## Look at every figure

Render all figures from real data into a scratch folder, view a contact sheet of every
PNG, and check that each reads at the column width.
