---
title: "Assembling the 3 and 12 minute cuts"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, ffmpeg, youtube, editing]
created: 2026-10-20
updated: 2026-10-20
---
# Assembling the 3 and 12 minute cuts

From `review.csv` ([[Highlight review workflow]]) to upload-ready MP4s with bash + ffmpeg: one lossless 1080p60 clip per kept row (trim, speed, freeze-frame, upscale, captions), Pillow title cards, then crossfades (short) or plain concatenation with chapter cards (long), encoded as in [[Encoding GIFs and YouTube videos]]. Grids and side-by-side layouts for [[Checkpoint comparison grid]] are at the end.

## Steps

1. Shared helpers (`bin/common.sh`): `ff() { ffmpeg -nostdin -hide_banner -loglevel error -y "$@"; }`; `tsv()` prints the CSV with fields joined by `\x1f`, read with `while IFS=$'\x1f' read -r id keep score type caption rollout game policy start end peak seconds speed events`.

2. Clips (`bin/cut.sh review/review_short.csv land clips_short_land`; `vert` for 1080 x 1920). Per kept row, with `a = peak - start + 1`, `K X Y` the integer scale and offset of the layout, `fz` = 1 s for achievement / floor / breakthrough / boss / death and 0 otherwise, `tf = a / (10 * speed)`:

   ```bash
   ffmpeg -nostdin -y -i masters/$rollout.mkv -loop 1 -i caps/${id}_land_panel.png -loop 1 -i caps/${id}_land_banner.png -filter_complex "
   [0]trim=start_frame=$start:end_frame=$((end + 1)),setpts=PTS-STARTPTS,split[x][y];
   [x]trim=end_frame=$a,setpts=(PTS-STARTPTS)/$speed,tpad=stop_mode=clone:stop_duration=$fz[a];
   [y]trim=start_frame=$a,setpts=(PTS-STARTPTS)/$speed[b];
   [a][b]concat=n=2:v=1:a=0,fps=60,scale=iw*$K:ih*$K:flags=neighbor,pad=1920:1080:$X:$Y:color=0x121418[g];
   [g][1]overlay=0:0:shortest=1[c];[c][2]overlay=0:0:shortest=1:enable='between(t,$tf,$tf+$fz)',
   scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv[v]" \
     -map "[v]" -r 60 -c:v libx264 -qp 0 -preset veryfast -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv clips/$id.mp4
   ```

   Best quality: re-render only the kept windows at their final size and cut those with scale 1 (`render.py --kept` loops all windows in one process; tile size 80 px for Craftax and 112 px for Classic in `land`, 96 / 112 in `vert`), so the HUD is drawn sharp instead of upscaled from 16 px:

   ```bash
   bin/render_kept.sh review/review_short.csv land hires_short_land      # render.py --kept review.csv --layout land --out-dir ...
   HIRES=hires_short_land bin/cut.sh review/review_short.csv land clips_short_land_hq
   ```

   When the peak is the last frame the split is skipped (`trim ... ,setpts=(PTS-STARTPTS)/$speed,tpad=...`). Caption PNGs come from `captions.py clips review.csv land caps/`.

3. Cards (`bin/card.sh land cards/intro.mp4 3 "Craftax agent highlights" "test cut"`):

   ```bash
   ffmpeg -loop 1 -framerate 60 -t 3 -i card.png -vf "fade=t=in:st=0:d=0.4,fade=t=out:st=2.6:d=0.4,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=..." -c:v libx264 -qp 0 -preset veryfast card.mp4
   ```

4. Short cut with 0.5 s crossfades (`bin/make_short.sh review/review_short.csv land clips_short_land out/short_3min_land.mp4 10`): intro, the first 10 kept clips, outro. `bin/assemble.sh` builds the `xfade` chain, offset of transition k = sum of the first k durations - k x 0.5:

   ```bash
   graph="[0:v]settb=AVTB,fps=60[s0];[1:v]settb=AVTB,fps=60[s1];...;
          [s0][s1]xfade=transition=fade:duration=0.5:offset=2.5[x1];[x1][s2]xfade=transition=fade:duration=0.5:offset=12.5[x2];..."
   ffmpeg -i intro.mp4 -i c001.mp4 ... -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
     -filter_complex "$graph" -map "[x11]" -map 12:a -shortest $ENC out/short_3min_land.mp4
   ```

5. Long cut with chapters (`bin/make_long.sh review/review_long.csv clips_long_land out/long_12min_land.mp4`): intro card (4 s), then per chapter a 3 s card and its kept clips in score order (Going deeper = floor + boss, Unlocks = achievement, Fights, Breakthroughs = breakthrough + close call, Deaths), outro; joined with the concat demuxer and a 0.5 s fade-out:

   ```bash
   for c in "$@"; do echo "file '$(cd "$(dirname "$c")" && pwd)/$(basename "$c")'" >> list.txt; done
   ffmpeg -f concat -safe 0 -i list.txt -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
     -vf "fade=t=out:st=$((total - 0.5)):d=0.5" -map 0:v -map 1:a -shortest $ENC out/long_12min_land.mp4
   ```

6. Grids (`bin/grid.sh out.mp4 <cols> <scale> <seconds> masters...`): per input `tpad=stop_mode=clone:stop=-1,trim=duration=$SEC,fps=60,scale=iw*$K:ih*$K:flags=neighbor`, a Pillow label bar `vstack`ed on top, then `xstack=inputs=$n:layout=0_0|${cw}_0|0_${ch}|${cw}_${ch}:fill=0x121418`, padded to 1920 x 1080. Shorter episodes hold their last frame until the longest ends.

7. Two games side by side (`bin/side_by_side.sh craftax.mkv classic.mkv 30 out.mp4`): Craftax x5 (880 x 1040), Classic x7 padded to 1008 x 1040, `hstack=inputs=2`, padded to 1920 x 1080.

## Results

| Output | Content | Duration | Size | Time |
|---|---|---|---|---|
| `clips_short_land/` | 12 kept clips, lossless 1080p60 | 207.8 s in total | 137 MB | 61.7 s |
| `clips_short_vert/` | same, 1080 x 1920 | | 133 MB | 47.0 s |
| `clips_long_land/` | 60 kept clips | 690.7 s in total (656.0 s of footage + 35 one-second freezes) | 577 MB | 153.1 s |
| One clip, lossless 1080p60 | 191 frames + 1 s freeze | 20.1 s | 2.9 MB | 7.6 s |
| Same row at speed 0.5 / 4 | slow motion / time-lapse | 39.3 s / 5.7 s | | |
| `short_3min_land.mp4` | intro + 10 clips + outro, 0.5 s crossfades | 171.4 s | 93.8 MB | 74.0 s |
| `short_3min_vert.mp4` | same, vertical (Shorts) | 171.4 s | 91.7 MB | 76.1 s |
| `hires_short_land/` | the 12 kept windows re-rendered at 80 / 112 px per tile, 2,084 frames | | 577 MB | 128.3 s (3 compiles) |
| `clips_short_land_hq/` | 12 clips cut from those windows | | 256 MB | 34.1 s |
| `short_3min_land_hq.mp4` | same cut as `short_3min_land.mp4`, sharp HUD | 171.4 s | 140.3 MB (6.5 Mb/s) | 60.5 s |
| `long_12min_land.mp4` | 67 pieces: 6 cards + 60 clips + outro, concat | 712.7 s | 431.8 MB | 230.1 s |
| `grid_2x2.mp4` | 4 Craftax masters x2, 60 s (one episode ends at 28.4 s and holds) | 60.0 s | 42.6 MB | 17.7 s |
| `grid_4x2_classic.mp4` | 8 Classic masters x2 | 30.0 s | 30.7 MB | 12.0 s |
| `side_by_side.mp4` | Craftax x5 next to Classic x7 | 30.0 s | 36.2 MB | 16.5 s |

Throughput per minute of 1080p60 output, from the long cut (load average 11 to 17 during these runs):

| Stage | Seconds per output minute |
|---|---|
| 16 px master render (600 steps at 1x, after compile) | about 1 to 2 |
| Cut to lossless clips | 13.3 |
| Final encode (concat, CRF 12 slow) | 19.4 |
| Final encode with crossfades (short cut) | 25.9 |
| Re-render of kept windows at final size (`render_kept.sh`, 2,084 frames in 128.3 s incl. 3 compiles) | 36.9 |
| Cut from re-rendered windows (12 clips, 34.1 s for 207.8 s) | 9.8 |

## Pitfalls

- zsh (the macOS default shell) does not word-split `$VAR`; run the scripts with `bash`.
- `set -euo pipefail` plus a `while` loop ending in `[ cond ] && echo` makes the pipeline fail on the last non-matching row; end the line with `|| true`.
- `IFS=$'\t'` collapses empty fields (`keep` is empty in `candidates.csv`); use a non-whitespace separator such as `\x1f`.
- The concat demuxer resolves relative paths against the list file's folder, and reports no duration (`N/A`); write absolute paths and sum `ffprobe` durations per clip.
- `xfade` needs equal time bases; `settb=AVTB,fps=60` on every input.
- macOS `mktemp` needs the `X`s at the end of the template.
- `fps=60` after a speed change resamples: 4x of 191 frames gave 5.7 s instead of 5.775 s.
- A 3-minute Short must stay at or under 180 s including cards; 10 clips + 2 cards + 0.5 s crossfades gave 171.4 s.

## Not tested

- Music or voice-over tracks (`amix`, loudness normalisation).
- Chapter markers in the YouTube description generated from the card timestamps.
- Transitions other than `fade` (`xfade` offers many).
- A 3 x 3 grid at 1080p (cells would need a scale between x1 and x2, which is non-integer).
