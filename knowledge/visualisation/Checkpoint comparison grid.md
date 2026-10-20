---
title: "Checkpoint comparison grid"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, ffmpeg, checkpoints, video]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/visualisation/grid.py]]"
  - "[[src/wut_murl_craftax/visualisation/eval_rollouts.py]]"
  - "[[tests/visualisation/test_grid.py]]"
---
# Checkpoint comparison grid

The same eval seed played by four checkpoints in a 2x2 grid, all starting at step 0, one game step per video frame, each panel labelled with its training steps and fixed-seed eval score. Inputs come from [[Recording training data]]; frames are replayed as in [[Rendering Craftax frames]].

## Steps

1. Pick checkpoints spread along the curve, not in steps; the 2M-step test used updates 0, 9, 33, 122 (0.0M, 0.1M, 0.5M, 2.0M steps).
2. Build the grid (default 540 px panels = 1080 x 1080):

   ```bash
   uv run python -m wut_murl_craftax.visualisation.grid outputs/runs/classic --updates 0,9,33,122 --seed 0
   ```

3. What `build_grid` does:
   - replays each checkpoint's stored episode (`episode_frames`, 16 px tiles) into a lossless panel through `ffmpeg.FrameWriter`;
   - draws the labels with Pillow (`panel_label`: colour key shared with the curve markers, `X.XM steps`, `eval score N% of max (16 seeds)`) and an `EPISODE OVER, step N` banner (`text.banner`), because this ffmpeg has no `drawtext`;
   - runs one ffmpeg graph per panel and an `xstack`:

   ```text
   [k:v]scale=P:P:force_original_aspect_ratio=decrease:flags=neighbor,pad=P:P:(ow-iw)/2:(oh-ih)/2:color=#1a1a19,
        tpad=stop_mode=clone:stop_duration=D[sk];
   [sk][label_k]overlay=0:0[lk];[lk][ended_k]overlay=0:0:enable='gte(t,(len_k-1)/fps)'[pk];
   [p0][p1][p2][p3]xstack=inputs=4:layout=0_0|P_0|0_P|P_P:fill=#1a1a19[v]   -> -frames:v N, libx264 crf 18
   ```

4. Scale-and-pad keeps Craftax frames (11 x 13 tiles incl. the inventory) undistorted; Classic frames are square.
5. Stack it with the synced curves for a Short ([[Animated training curves]]); final encodes per [[Encoding GIFs and YouTube videos]].

## Results

Prototype, Craftax-Classic 2M-step run, seed 0, episodes of 192, 204, 273 and 274 steps, 15 fps, 270 px panels (540 x 540):

| Stage | Time | Output |
|---|---|---|
| 4 lossless panels from stored frames + 8 PNGs | 0.34 s | |
| Same from replayed actions (4 compiles + render) | 31.42 s | identical frames |
| ffmpeg composite + libx264 crf 18 | 0.38 s | 18.27 s, 274 frames, 2,789,565 B |

Repo code, 0.5M-step run (checkpoints 0, 4, 11, 30), 16 eval seeds, 270 px panels:

| Command | Wall | Output |
|---|---|---|
| `grid ... --updates 0,4,11,30 --panel 270` | 14.5 s (replay, labels, encode) | `grid.mp4`, 540 x 540, 208 frames, 13.87 s, 1,670,735 B |

- Test (`test_grid.py`): 96 px panels of the 4-update test run give a 192 x 192 clip with as many frames as the longest episode.

## Pitfalls

- This ffmpeg build has no libfreetype or libass: `drawtext`, `subtitles` and `ass` do not exist; text arrives as PNG overlays.
- A PNG input to `overlay` is one frame; without `shortest=1` the default `eof_action=repeat` keeps it for the whole clip (the shared `ffmpeg.overlay_filter` sets `shortest=1`, so it is not used here).
- Without `tpad=stop_mode=clone` a panel that died early goes black or ends the `xstack` early.
- Use `flags=neighbor` for pixel art; bilinear scaling blurs the tiles.
- Nights and sleeping darken the frame; a sleeping agent fills its panel with a dark view (seen at 0.1M and 0.2M steps in the repo smoke run).
- With a shared seed the first frames of all panels are identical; the difference shows after the first divergent action.
- The panel label is the fixed-seed eval score; the curve shows training episodes; name both.

## Not tested

- Layouts other than 2x2 (1x3, 3x3).
- A step counter per panel.
- More than one seed per checkpoint.
- Choosing the seed from eval returns; see [[Highlight detection]].
- The default 540 px panels (1080 x 1080) were not rendered; tests and smoke runs used 64 to 270 px.
