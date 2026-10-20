---
title: "Highlight review workflow"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, highlights, review]
created: 2026-10-20
updated: 2026-10-20
---
# Highlight review workflow

Automatic ranking from [[Highlight detection]], then a person picks clips on one HTML page and downloads `review.csv`; the cut step reads only that file ([[Assembling the 3 and 12 minute cuts]]).

```mermaid
flowchart LR
  A[candidates.csv] --> B[bin/review.sh: strips, thumbs, previews]
  B --> C[review.html + contact_sheet.png]
  C -->|keep / drop, caption, speed| D[review.csv]
  D --> E[bin/cut.sh]
```

## Steps

1. Build the review material for the top 60 candidates:

   ```bash
   bin/review.sh review/candidates.csv
   ```

   Per candidate, from the 16 px master (CSV rows split on `\x1f` by a one-line Python `csv` reader, because a tab `IFS` collapses empty fields):

   ```bash
   mid=$(( (start + peak) / 2 ))
   ffmpeg -i "$M" -vf "select='eq(n,$start)+eq(n,$mid)+eq(n,$peak)+eq(n,$end)',scale=iw*2:ih*2:flags=neighbor,tile=4x1:padding=4:color=white" -frames:v 1 strips/$id.png
   ffmpeg -i "$M" -vf "select='eq(n,$peak)',scale=iw*2:ih*2:flags=neighbor" -frames:v 1 thumbs/$id.png
   ffmpeg -i "$M" -vf "trim=start_frame=$start:end_frame=$((end + 1)),setpts=PTS-STARTPTS,scale=iw*2:ih*2:flags=neighbor,format=yuv420p" \
          -c:v libx264 -crf 20 -preset veryfast -movflags +faststart previews/$id.mp4
   ```

2. `review_page.py` (Pillow) writes:
   - `contact_sheet.png`: peak thumbnails 6 per row with id, score, type, caption, rollout and time range under each; for a quick look or to paste in chat.
   - `review.html`: one row per candidate with the 4-frame strip, the event list, a preview `<video>` (poster = peak frame, loads on play), an editable caption, a speed select (0.5, 1, 2, 4, 8) and keep / drop radios; header shows kept count and kept seconds (seconds / speed); "Keep top 10" and "Download review.csv" buttons; choices persist in `localStorage` per page; narrow screens get one block per candidate.

3. Open the page over HTTP so the previews play:

   ```bash
   cd review && python -m http.server 8765 --bind 127.0.0.1   # then http://127.0.0.1:8765/review.html
   ```

4. Mark keep / drop, fix captions, set speed, press "Download review.csv". The file has the same columns as `candidates.csv`; `keep` is `keep`, `drop` or empty; `bin/cut.sh` and `bin/make_short.sh` take rows with `keep == keep` in file order (score order).

5. Without a browser: edit the `keep`, `caption` and `speed` columns of `candidates.csv` in any spreadsheet and save as CSV.

## Results

| Measurement | Result |
|---|---|
| Strips + thumbnails + preview MP4s for 60 candidates | 33.2 s; 7.0 MB, 2.1 MB and 29 MB |
| `contact_sheet.png` | 2160 x 4700 px, 2.2 MB |
| `review.html` | 82 KB, no external scripts |
| Browser test (Claude Browser pane, served on 127.0.0.1) | "Keep top 10", then drop c002, speed 2 on c001: header showed 9 kept, 159.0 s; the CSV built by "Download review.csv" (Blob read in the page) had 61 lines, 9 `keep`, 1 `drop`, `speed` 2 on c001; a preview loaded as 352 x 416, 19.1 s |
| Reviewed files used for the test cuts (marks applied by a script, same columns) | `review_short.csv` (12 kept, 198.8 s at the chosen speeds), `review_long.csv` (all 60 kept, 656.0 s) |

## Pitfalls

- The Claude Browser pane shows a `file://` page only as a static snapshot (no clicks, no video); serve it over HTTP.
- `trim=start_frame/end_frame` is frame-exact but decodes from frame 0; fine for masters of a few thousand frames, for 100,000-step episodes input seeking (`-ss` before `-i`) is the faster option, not tested here.
- The CSV is the contract between steps: never rename columns; add new ones at the end.

## Not tested

- Several reviewers merging their choices (one CSV per person, majority vote).
- Hosting the page as a shared artifact for remote review.
