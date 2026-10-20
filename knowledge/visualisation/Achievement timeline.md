---
title: "Achievement timeline"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, matplotlib, achievements, heatmap]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/visualisation/timeline.py]]"
  - "[[src/wut_murl_craftax/training/records.py]]"
  - "[[tests/visualisation/test_timeline.py]]"
---
# Achievement timeline

An animated heatmap: one row per achievement (22 in Classic, 67 in Craftax; list in [[Craftax achievements]]), columns are PPO updates, colour is the success rate (% of training episodes), rows sorted by first unlock with a ring at the first unlock. The cursor sweeps left to right; a counter shows how many achievements have been unlocked at least once. Data: `metrics.npz` and `episodes.npz` from [[Recording training data]].

## Steps

1. Render (default 1920 x 1080, 12 s sweep + 2 s hold at 30 fps; the last frame is also saved as PNG):

   ```bash
   uv run python -m wut_murl_craftax.visualisation.timeline outputs/runs/classic [--logx] [--size 1920x1080]
   ```

2. First unlock = first update in which any training episode earned the achievement (`records.first_unlocks` on the unpacked episode bits); `first_unlock_order` sorts rows by it, never-unlocked rows last.
3. Colour: one-hue ramp from the surface colour to light blue with `PowerNorm(0.5, 0, 100)`, so 1 to 5% rates are visible.
4. Per frame the columns right of the cursor are masked and passed rings appear; `--logx` sweeps on a geometric schedule over a log axis.
5. Encode the lossless output for sharing: `ffmpeg -i timeline.mkv -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart timeline.mp4`.

## Results

Prototype runs:

| Run | Rows | Frames, size | Render | x264 | MP4 size |
|---|---|---|---|---|---|
| Classic, 2M steps, 122 updates, linear x | 22 | 180, 960 x 540 | 5.9 s | 0.26 s | 58,033 B |
| Same, log x | 22 | 180, 960 x 540 | 6.1 s | 0.21 s | 52,213 B |
| Craftax-Symbolic, 213k steps, 13 updates | 67 | 120, 960 x 1080 | 9.2 s | 0.25 s | 38,304 B |

Repo code:

| Run | Rows | Command | Wall | Output |
|---|---|---|---|---|
| Classic, 0.5M steps, 30 updates | 22 | `timeline ... --seconds 6 --fps 15 --size 960x540` | 6.1 s | 120 frames, 108,551 B (lossless) |
| Craftax-Symbolic, 8,192 steps, 8 updates | 67 | `timeline ... --seconds 2 --fps 10 --size 960x1080` | 4.8 s | |

- Classic at 2M steps: 18 of 22 achievements unlocked at least once; never: collect iron, collect diamond, make iron pickaxe, make iron sword.
- First unlocks in that run were spread over 0 to 0.5M steps except eat plant at 1.3M; on a linear axis they crowd the left 25%, the log axis spreads them.
- Craftax at 213k steps: 13 of 67 unlocked, including enter dungeon.

## Pitfalls

- A linear colour scale hides rare achievements (1 to 5%) in the near-black end; the square-root norm fixed it.
- The first-unlock ring marks the right edge of the update's column; one lucky episode creates it, so a ring can precede visible colour.
- At 67 rows labels need about 1080 px of height (font = height / 120).
- Never-unlocked rows stay (empty); dropping them hides how much is left.
- Training-episode rates count only episodes that finished in that update; see [[Recording training data]].
- `first_unlocks` returns int32; the sort key casts to int64 before using `iinfo(int64).max` for "never" (an int32 overflow was caught by the test).

## Not tested

- A long Craftax run (the 67-row renders used 13 and 8 updates).
- Grouping the 67 rows by floor or tier ([[Game]] lists them).
- Eval-seed success rates per checkpoint instead of training episodes.
- A cursor synced to a gameplay clip (the cursors of [[Animated training curves]] would apply).
