---
title: "Animated training curves"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, matplotlib, ffmpeg, curves, gif]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/visualisation/curves.py]]"
  - "[[src/wut_murl_craftax/visualisation/charts.py]]"
  - "[[src/wut_murl_craftax/visualisation/montage.py]]"
  - "[[tests/visualisation/test_curves.py]]"
  - "[[tests/visualisation/test_montage.py]]"
---
# Animated training curves

A panel that grows with training: score as % of the maximum reward (main curve, the scoreboard metric), achievements unlocked per episode (second panel) and success rate per achievement (bars). matplotlib renders straight into an ffmpeg pipe; the panel sits under the [[Checkpoint comparison grid]] in the Short and beside the checkpoint montage in the long cut. Data: `metrics.npz` from [[Recording training data]].

## Steps

1. Series, all EMA-smoothed (alpha 0.05): score = `return / max_reward * 100` (22 Classic, 226 Craftax; [[Craftax scoring and evaluation]]); achievements unlocked = sum of success rates / 100; bars = the `--bars` (default 22) achievements with the highest final rate.
2. Sync the cursor (an update index per frame) to the gameplay:
   - grid sync (`grid_cursor`): reveal over the first 70% of the clip, then hold; each checkpoint marker appears in its grid label colour when the cursor passes it.
   - sequence sync (`seq_cursor`): the cursor jumps to the checkpoint on screen; only its marker shows.
3. Render (default 1080 x 840, under a 1080 x 1080 grid = 1080 x 1920 Short):

   ```bash
   N=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 outputs/visualisation/grid.mp4)
   uv run python -m wut_murl_craftax.visualisation.curves outputs/runs/classic --updates 0,9,33,122 --frames $N
   ```

   - Static artists are drawn once; per frame only data changes, then `charts.frame(fig)` hands RGB to `ffmpeg.FrameWriter` (no PNG files).

4. Short (9:16) MP4 and GIF: stack grid over curves with the frame count pinned to the grid's, GIF through the shared palette filter:

   ```bash
   uv run python -m wut_murl_craftax.visualisation.montage --short outputs/visualisation/grid.mp4 outputs/visualisation/curves.mkv --out outputs/visualisation/short.mp4
   # vstack=inputs=2:shortest=1 -> -frames:v N, x264 crf 20 slow; GIF: fps=10,scale=360:-1:flags=lanczos + ffmpeg.write_gif(colors=128)
   ```

5. Long cut (16:9): checkpoints one after another at 1080 x 1080 beside an 840 x 1080 sequence-synced panel:

   ```bash
   uv run python -m wut_murl_craftax.visualisation.montage outputs/runs/classic --updates 0,9,33,122 --seconds 6
   ```

6. Final encode settings: [[Encoding GIFs and YouTube videos]]; placement in the cuts: [[Assembling the 3 and 12 minute cuts]].

## Results

Prototype (2 panels: score and bars), 2M-step Classic run, checkpoints 0, 9, 33, 122, 15 fps:

| Output | Render | Encode | Size | ffprobe |
|---|---|---|---|---|
| Curve panel, grid sync, 540 x 420, 274 frames (FFV1) | 9.3 s (29.6 fps) | in pipe | 12,038,314 B | |
| Short MP4, 540 x 960 | | 0.77 s (x264 slow, crf 20) | 1,974,658 B | h264, yuv420p, 15/1, 274 frames, 18.27 s |
| Curve panel, sequence sync, 560 x 720, 240 frames | 17.6 s (13.6 fps) | in pipe | | |
| Long-cut montage, 1280 x 720, 4 x 4 s | | 1.3 s | 1,202,757 B | |

GIF from the prototype's 18.27 s Short (bayer dither 4, `stats_mode=diff`):

| fps | Width | Colours | Time | Size |
|---|---|---|---|---|
| 8 | 270 | 64 | 0.44 s | 1,544,575 B |
| 10 | 360 | 128 | 0.75 s | 2,976,019 B |
| 15 | 360 | 128 | 1.06 s | 4,361,935 B |
| 10 | 540 | 256 | 1.38 s | 7,378,994 B |

Repo code (3 panels), 0.5M-step run, checkpoints 0, 4, 11, 30, 15 fps:

| Command | Wall | Output | ffprobe |
|---|---|---|---|
| `curves ... --frames 208 --size 540x620` | 9.8 s | `curves.mkv` (lossless x264rgb), 156,068 B | |
| `montage --short grid.mp4 curves.mkv` | 3.6 s | `short.mp4`, 1,270,289 B | h264, 540 x 1160, 15/1, 208 frames, 13.87 s |
| same, GIF (shared filter: no dither, `stats_mode=full`, 128 colours) | | `short.gif`, 2,416,235 B | gif, 360 x 773, 10/1, 139 frames, 13.90 s |
| `montage ... --updates 0,4,11,30 --seconds 3` (default size) | 23.2 s | `montage.mp4`, 1,493,575 B | 1920 x 1080, 180 frames, 12.0 s |

- Render speed varies with machine load: the prototype grid-sync panel took 9.3 to 12.3 s in different runs.

## Pitfalls

- `vstack` without `shortest=1` and `-frames:v` gave 275 frames from two 274-frame inputs; the concat-and-`hstack` montage gave 14 frames instead of 12 in its test until `-frames:v` was pinned.
- The training-episode score (curve) and the fixed-seed eval score (grid label) differ; titles name both.
- Per-update success rates are noisy (tens of episodes per update); without the EMA the bars flicker.
- matplotlib clips markers at the axes edge; the last checkpoint marker needs `clip_on=False` and a 3% x margin.
- Font sizes scale with the panel width (`width / 60`); render at final resolution rather than upscaling.
- The checkpoint colours are the dark-surface categorical slots 1 to 4 (`#3987e5`, `#d95926`, `#199e70`, `#c98500`), validated for colour-vision deficiency on `#1a1a19`; identity also carries in the label text.

## Not tested

- The default 1080 x 840 panel and the 1080 x 1920 Short (tests and smoke runs were low-res).
- Two-pass GIF (`palettegen` to a file) and `gifski`.
- A log-scaled x axis for the score curve (only on the [[Achievement timeline]]).
- Curves from several seeds with a confidence band.
