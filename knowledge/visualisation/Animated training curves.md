---
title: "Animated training curves"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, matplotlib, ffmpeg, curves, gif]
created: 2026-10-20
updated: 2026-10-20
---
# Animated training curves

A two-part panel that grows with training: score (episode return as % of the maximum reward) over environment steps, and per-achievement success rates as bars. Rendered with matplotlib straight into an ffmpeg pipe, then stacked with gameplay. Data: `metrics.npz` from [[Recording training data]].

## Steps

1. Smooth the per-update series with an EMA (alpha 0.05); score = `return / max_reward * 100` with max 22 (Classic) or 226 (Craftax), see [[Craftax scoring and evaluation]].
2. Map every output frame to a cursor (an update index):
   - grid sync (all checkpoints on screen, [[Checkpoint comparison grid]]): reveal the curve over the first 70% of the clip, hold the rest; each checkpoint's marker appears when the cursor passes it, in the colour of its grid label.
   - sequence sync (one checkpoint at a time, long cut): the cursor jumps to the checkpoint whose gameplay is on screen; only that checkpoint's marker is shown.

   ```python
   if mode == "grid":
       n_rev = int(n_frames * 0.7)
       cur = np.r_[np.linspace(0, len(x) - 1, n_rev), np.full(n_frames - n_rev, len(x) - 1)].astype(int)
   else:  # frames per segment = gameplay frames of that checkpoint
       cur = np.concatenate([np.full(n, max(0, u - 1)) for u, n in zip(updates, seg_frames)])
   ```

3. Draw static artists once, then per frame update data and pull the RGB buffer into an ffmpeg raw pipe (no PNG files):

   ```python
   import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
   fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#1a1a19")
   gs = fig.add_gridspec(2, 1, height_ratios=[2, 3], hspace=0.6, left=0.30, right=0.95, top=0.93, bottom=0.09)
   ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
   ax.plot(x, score, color="#2c2c2a", lw=1.5)                       # full curve as a faint track
   (line,) = ax.plot([], [], color="#f0efec", lw=2)
   dots = [ax.plot([x[max(0, u - 1)]], [score[max(0, u - 1)]], "o", ms=8, color=c, mec="#1a1a19", mew=2,
                   visible=False, clip_on=False)[0] for u, c in zip(updates, ["#3987e5", "#d95926", "#199e70", "#c98500"])]
   vline = ax.axvline(0, color="#898781", lw=1, ls=":")
   bars = bx.barh(np.arange(len(names)), np.zeros(len(names)), height=0.7, color="#3987e5")  # sorted by final rate
   bx.set_xlim(0, 100); bx.invert_yaxis()
   pipe = subprocess.Popen(["/opt/homebrew/bin/ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", "15", "-i", "-", "-c:v", "ffv1", "curves.mkv"], stdin=subprocess.PIPE)
   for f, i in enumerate(cur):
       line.set_data(x[: i + 1], score[: i + 1]); vline.set_xdata([x[i], x[i]])
       for k, d in enumerate(dots):
           d.set_visible(max(0, updates[k] - 1) <= i if mode == "grid" else k == seg[f])
       for b, v in zip(bars, np.nan_to_num(rates[i, order])):
           b.set_width(v)
       fig.canvas.draw()
       pipe.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
   pipe.stdin.close(); pipe.wait()
   ```

4. Short (9:16): stack the 540 x 540 grid over a 540 x 420 curve panel and pin the frame count to the grid's:

   ```bash
   ffmpeg -y -i grid.mp4 -i curves_grid.mkv \
     -filter_complex "[0:v][1:v]vstack=inputs=2:shortest=1,format=yuv420p[v]" \
     -map "[v]" -frames:v 274 -c:v libx264 -crf 20 -preset slow -movflags +faststart composite.mp4
   ffprobe -v error -count_frames -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration,size -of compact composite.mp4
   ```

5. GIF up to 20 s from the composite, one-pass palette:

   ```bash
   ffmpeg -y -i composite.mp4 -filter_complex \
     "fps=10,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
     -loop 0 composite.gif
   ```

6. Long cut (16:9): play the checkpoints one after another at 720 x 720 and put a 560 x 720 sequence-synced curve panel beside them:

   ```bash
   # per checkpoint k: 4 s = 60 frames; label scaled to half width; then concat and hstack
   ffmpeg -y -i panel_0.mkv -i label_0.png ... -i panel_3.mkv -i label_3.png -i curves_seq.mkv -filter_complex "\
   [0:v]scale=720:720:flags=neighbor,tpad=stop_mode=clone:stop_duration=4,trim=end_frame=60,setpts=PTS-STARTPTS[s0];[1:v]scale=360:-1[l0];[s0][l0]overlay=0:0[g0];\
   ... same for k = 1..3 ...
   [g0][g1][g2][g3]concat=n=4:v=1:a=0[game];[game][8:v]hstack=inputs=2,format=yuv420p[v]" \
     -map "[v]" -r 15 -c:v libx264 -crf 18 -movflags +faststart seq.mp4
   ```

7. Final encode settings per platform: [[Encoding GIFs and YouTube videos]]; placement in the cuts: [[Assembling the 3 and 12 minute cuts]].

## Results

2M-step Classic run (122 updates), checkpoints 0, 9, 33, 122; 15 fps.

| Output | Render | Encode | Size | ffprobe |
|---|---|---|---|---|
| Curve panel, grid sync, 540 x 420, 274 frames (FFV1) | 9.3 s (29.6 fps) | in pipe | 12,038,314 B | |
| Short composite MP4, 540 x 960 | | 0.77 s (x264 slow, crf 20) | 1,974,658 B | h264, yuv420p, 15/1, 274 frames, 18.27 s |
| Curve panel, sequence sync, 560 x 720, 240 frames | 17.6 s (13.6 fps) | in pipe | | |
| Long-cut montage MP4, 1280 x 720, 4 x 4 s | | 1.3 s | 1,202,757 B | |

GIF from the 18.27 s composite (`palettegen` + `paletteuse`, bayer dither 4):

| fps | Width | Colours | Time | Size |
|---|---|---|---|---|
| 8 | 270 | 64 | 0.44 s | 1,544,575 B |
| 10 | 360 | 128 | 0.75 s | 2,976,019 B |
| 15 | 360 | 128 | 1.06 s | 4,361,935 B |
| 10 | 540 | 256 | 1.38 s | 7,378,994 B |

- ffprobe of the 10 fps, 360-wide GIF: gif, 360 x 640, 10/1, 183 frames, 18.30 s.
- Render speed varied with machine load: the same grid-sync panel took 11.9 to 12.3 s in earlier runs.

## Pitfalls

- `vstack` without `shortest=1` and `-frames:v` gave 275 frames from two 274-frame inputs; pin the count.
- The training-episode score (curve) and the fixed-seed eval score (grid label) differ; title both.
- Per-update success rates are noisy (tens of episodes per update); without the EMA the bars flicker.
- matplotlib defaults clip markers at the axes edge; the last checkpoint's marker needs `clip_on=False` and a 3% x margin.
- Achievement names at 7 pt are legible at 540 px only on a large screen; render the panel at final resolution rather than upscaling.
- The four checkpoint colours are the dark-surface categorical slots 1 to 4 (`#3987e5`, `#d95926`, `#199e70`, `#c98500`), validated for colour-vision deficiency on `#1a1a19`; identity also carries in the label text.

## Not tested

- Two-pass GIF (`palettegen` to a PNG, then `paletteuse`) and `gifski`.
- Log-scaled x axis for the score curve (tested only on the [[Achievement timeline]]).
- Curves from several seeds with a confidence band.
- Driving the cursor by wall-clock training time instead of environment steps.
