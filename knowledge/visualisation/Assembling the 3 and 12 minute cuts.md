---
title: "Assembling the 3 and 12 minute cuts"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, ffmpeg, youtube, editing]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/visualisation/cuts.py]]"
  - "[[src/wut_murl_craftax/visualisation/ffmpeg.py]]"
  - "[[src/wut_murl_craftax/visualisation/text.py]]"
  - "[[tests/visualisation/test_cuts.py]]"
---
# Assembling the 3 and 12 minute cuts

From `review.csv` ([[Highlight review workflow]]) to upload-ready MP4s with `python -m wut_murl_craftax.visualisation.cuts`: one lossless 60 fps clip per kept row (trim, speed, freeze-frame, upscale, captions with the footage label), Pillow title cards, then crossfades (3-minute cut, 16:9 and Short) or plain concatenation with chapter cards (12-minute cut), encoded as in [[Encoding GIFs and YouTube videos]]. Craftax-Classic clips open each cut as a short warm-up, Craftax follows. Grids for [[Checkpoint comparison grid]] are at the end.

## Steps

1. Final-size windows of the kept rows ([[Rendering Craftax frames]]), then clips; `--layout land` (1920 x 1080, default), `vert` (1080 x 1920 Short) or `land4k` (3840 x 2160):

   ```bash
   R=outputs/visualisation
   uv run python -m wut_murl_craftax.visualisation.render --kept $R/review/review.csv --layout land --out-dir $R/hires_land
   uv run python -m wut_murl_craftax.visualisation.cuts clips $R/review/review.csv --layout land --hires-dir $R/hires_land
   uv run python -m wut_murl_craftax.visualisation.cuts clips $R/review/review.csv --layout vert   # from the 16 px masters
   ```

   Per kept row (`cut_clip`; `a = peak - start + 1`, freeze `fz` = 1 s for achievement, floor, breakthrough, boss, death, else 0; `tf = a / (10 x speed)`):

   ```text
   [0]trim=start_frame=S:end_frame=E+1,setpts=PTS-STARTPTS,split[x][y];
   [x]trim=end_frame=a,setpts=(PTS-STARTPTS)/speed,tpad=stop_mode=clone:stop_duration=fz[a];
   [y]trim=start_frame=a,setpts=(PTS-STARTPTS)/speed[b];[a][b]concat=n=2:v=1:a=0,
   fps=60,scale=iw*K:ih*K:flags=neighbor,pad=W:H:X:Y:color=0x121418[g];
   [g][1]overlay=0:0:shortest=1[c];[c][2]overlay=0:0:shortest=1:enable='between(t,tf,tf+fz)',<BT709>[v]
   ```

   Inputs 1 and 2 are the caption panel (caption, game name, footage label, steps, speed) and the freeze banner from `captions`; output `-c:v libx264 -qp 0 -preset veryfast` with BT.709 tags. Layout table `LAYOUTS` (16 px master scale K and offset): `land` Craftax x5 at (520, 20), Classic x7 at (456, 36); `vert` Craftax x6 at (12, 336), Classic x7 at (36, 456); `land4k` Craftax x10 at (1040, 40), Classic x14 at (912, 72). With `--hires-dir` the window already has the final size (K = 1).

2. 3-minute cuts with 0.5 s crossfades (`short_cut`: intro card, at most 2 Classic clips, then Craftax clips while the total stays at or under 180 s, outro):

   ```bash
   uv run python -m wut_murl_craftax.visualisation.cuts short $R/review/review.csv --layout land
   uv run python -m wut_murl_craftax.visualisation.cuts short $R/review/review.csv --layout vert --voice VOICE.wav --music MUSIC.wav
   ```

   `ffmpeg.xfade` chains `xfade=transition=fade:duration=0.5:offset=...` with transition k at sum(d_0 .. d_k-1) - k x 0.5, every input `settb=AVTB,fps=60`.

3. 12-minute cut (`long_cut`: 4 s intro, "Warm-up: Craftax-Classic" with at most 10 Classic clips, Craftax chapters Going deeper (floor, boss), Unlocks, Fights, Breakthroughs (breakthrough, close call), Deaths, each with a 3 s card, outro; concat demuxer with absolute paths, 0.5 s fade-out):

   ```bash
   uv run python -m wut_murl_craftax.visualisation.cuts long $R/review/review.csv --voice VOICE.wav --music MUSIC.wav
   ```

4. Cards: `card(layout, out, seconds, title, subtitle)` holds a `text.title_card` PNG at 60 fps with 0.4 s fades from and to black.

5. Audio: `--voice` / `--music` mux after the video encode (`ffmpeg.mux_audio`: ducking + two-pass loudnorm, [[Encoding GIFs and YouTube videos]]); without them the track is silent. `cuts mux VIDEO OUT --voice ... --music ...` re-mixes an existing cut.

6. Grids (`grid`, `cuts grid OUT masters... --cols 2 --scale 2 --seconds 60`): each master `tpad=stop_mode=clone:stop=-1,trim=duration=...,fps=60` and integer upscale, a `text.label_bar` (looped at 60 fps) `vstack`ed on top, `ffmpeg.stack_filter(..., "x", cols, cell)` (`xstack`), padded to 1920 x 1080, `fps=60`.

## Results

First scripts (23 test rollouts, 60 candidates):

| Output | Content | Duration | Size | Time |
|---|---|---|---|---|
| `clips_short_land/` | 12 kept clips, lossless 1080p60 | 207.8 s in total | 137 MB | 61.7 s |
| `clips_short_vert/` | same, 1080 x 1920 | | 133 MB | 47.0 s |
| `clips_long_land/` | 60 kept clips | 690.7 s in total (656.0 s of footage + 35 one-second freezes) | 577 MB | 153.1 s |
| One clip, lossless 1080p60 | 191 frames + 1 s freeze | 20.1 s | 2.9 MB | 7.6 s |
| Same row at speed 0.5 / 4 | slow motion / time-lapse | 39.3 s / 5.7 s | | |
| `short_3min_land.mp4` | intro + 10 clips + outro, 0.5 s crossfades | 171.4 s | 93.8 MB | 74.0 s |
| `short_3min_vert.mp4` | same, vertical (Short) | 171.4 s | 91.7 MB | 76.1 s |
| `hires_short_land/` | the 12 kept windows at 80 / 112 px per tile, 2,084 frames | | 577 MB | 128.3 s (3 compiles) |
| `short_3min_land_hq.mp4` | same cut from those windows, sharp HUD | 171.4 s | 140.3 MB (6.5 Mb/s) | 60.5 s |
| `long_12min_land.mp4` | 67 pieces: 6 cards + 60 clips + outro, concat | 712.7 s | 431.8 MB | 230.1 s |
| `grid_2x2.mp4` | 4 Craftax masters x2, 60 s (one episode ends at 28.4 s and holds) | 60.0 s | 42.6 MB | 17.7 s |
| `grid_4x2_classic.mp4` | 8 Classic masters x2 | 30.0 s | 30.7 MB | 12.0 s |
| `side_by_side.mp4` | Craftax x5 next to Classic x7 (`hstack`) | 30.0 s | 36.2 MB | 16.5 s |

Package in the repo (smoke test: 7 new rollouts, 30 candidates, 29 kept):

| Output | Content | Duration | Size | Time |
|---|---|---|---|---|
| `clips_land/` from `hires_land/` | 29 clips | 300.5 s in total | 459 MB | 58 s |
| `clips_vert/` from masters | 29 clips | 300.5 s in total | 237 MB | 54 s |
| `out/short_3min_land.mp4` | 2 Classic + Craftax clips while within 180 s, crossfades | 172.9 s | 136.7 MB (6.3 Mb/s) | 65 s |
| `out/short_3min_vert.mp4` | same, vertical, tone voice + music, -14.0 LUFS | 173.0 s | 102.2 MB (4.7 Mb/s) | 70 s |
| `out/long_12min.mp4` | all 29 kept clips with chapters (only 300 s of footage in the smoke set) | 325.5 s | 287.9 MB | 127 s |
| `out/grid_2x2.mp4` | 4 Craftax masters x2, 30 s | 30.0 s | 16.7 MB | 8 s |
| Package tests (`test_cuts.py`) | freeze, layouts, warm-up order, short (both layouts, one with audio), long chapters, GIF clamp, grid, audio mix | | | 7 tests |

Throughput per minute of 1080p60 output, from the first long cut (load average 11 to 17 during these runs):

| Stage | Seconds per output minute |
|---|---|
| 16 px master render (600 steps at 1x, after compile) | about 1 to 2 |
| Cut to lossless clips | 13.3 |
| Final encode (concat, CRF 12 slow) | 19.4 |
| Final encode with crossfades (short cut) | 25.9 |
| Re-render of kept windows at final size (2,084 frames in 128.3 s incl. 3 compiles) | 36.9 |
| Cut from re-rendered windows (12 clips, 34.1 s for 207.8 s) | 9.8 |

## Pitfalls

- The concat demuxer resolves relative paths against the list file's folder and reports no duration (`N/A`); `ffmpeg.concat` writes absolute paths and sums `ffprobe` durations per clip.
- `xfade` needs equal time bases; `settb=AVTB,fps=60` on every input.
- A looped PNG input runs at its own rate (the first grid came out at 300/1 with irregular keyframes); `-loop 1 -framerate 60` and a final `fps=60` fix it.
- Filtering a flag list by value drops repeated values: removing the audio options from the encode arguments also removed the `2` of `-bf 2`; `youtube_args(audio=False)` builds video-only arguments instead.
- `fps=60` after a speed change resamples: 4x of 191 frames gave 5.7 s instead of 5.775 s.
- A Short must stay at or under 180 s including cards and crossfades; `short_cut` stops adding clips before the limit.
- The land caption box must end left of the Classic frame at x = 456: the box is 396 px wide plus 16 px padding.

## Not tested

- Real voice-over and music (generated tones only).
- Chapter markers in the YouTube description generated from the card timestamps.
- Transitions other than `fade` (`xfade` offers many).
- A 3 x 3 grid at 1080p (cells would need a scale between x1 and x2, which is non-integer).
- Side by side of two different-size masters through the package (the first scripts' `hstack` run above); `ffmpeg.stack_filter(..., "h")` builds the filter.
