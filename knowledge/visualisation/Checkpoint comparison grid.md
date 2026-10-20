---
title: "Checkpoint comparison grid"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, ffmpeg, checkpoints, video]
created: 2026-10-20
updated: 2026-10-20
---
# Checkpoint comparison grid

The same eval seed played by four checkpoints in a 2x2 grid, all starting at step 0, one game step per video frame, each panel labelled with its training stage and eval score. Inputs come from [[Recording training data]]; frames from [[Rendering Craftax frames]].

## Steps

1. Pick checkpoints spread on the curve, not in steps; with log-spaced checkpoints of a 2M-step run the test used updates 0, 9, 33, 122 (0.0M, 0.1M, 0.5M, 2.0M steps).
2. Get one episode of frames per checkpoint for the shared seed: the stored `eval/<update>_frames_<seed>.npy`, or replay of the stored actions (bit-identical, slower).
3. Write each panel losslessly through a raw pipe at native size (144 x 144 for Classic at 16 px blocks):

   ```python
   cmd = ["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
          "-s", f"{w}x{h}", "-r", "15", "-i", "-", "-c:v", "ffv1", f"panel_{k}.mkv"]
   p = subprocess.Popen(cmd, stdin=subprocess.PIPE); p.stdin.write(frames.tobytes()); p.stdin.close(); p.wait()
   ```

4. Render the labels as RGBA PNGs with Pillow, because this ffmpeg build has neither `drawtext` nor `subtitles`/`ass`:

   ```python
   from PIL import Image, ImageDraw, ImageFont
   FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
   def label_png(path, text, sub, w, color, size=15):
       f = ImageFont.truetype(FONT, size)
       im = Image.new("RGBA", (w, int(size * 2.9)), (26, 26, 25, 200)); d = ImageDraw.Draw(im)
       d.rectangle([0, 0, 5, im.height], fill=color)                 # colour key shared with the curve markers
       d.text((12, 3), text, font=f, fill="#f0efec")
       d.text((12, 3 + int(size * 1.3)), sub, font=ImageFont.truetype(FONT, int(size * 0.85)), fill="#c3c2b7")
       im.save(path)
   # label_png("label_0.png", "0.0M steps", "eval score 11% of max", 270, "#3987e5")
   # plus ended_<k>.png: a 270 x 45 translucent black strip "episode over at step N"
   ```

5. Composite in one ffmpeg call: nearest-neighbour upscale, freeze the last frame of panels whose episode ended (`tpad=stop_mode=clone`), label overlay, "episode over" banner from the panel's end time, `xstack`:

   ```bash
   P=270; D=18.267   # panel px; clip seconds = longest episode / fps, capped at 20 s
   ffmpeg -y -i panel_0.mkv -i panel_1.mkv -i panel_2.mkv -i panel_3.mkv \
     -i label_0.png -i ended_0.png -i label_1.png -i ended_1.png -i label_2.png -i ended_2.png -i label_3.png -i ended_3.png \
     -filter_complex "\
   [0:v]scale=$P:$P:flags=neighbor,tpad=stop_mode=clone:stop_duration=$D[s0];[s0][4:v]overlay=0:0[l0];[l0][5:v]overlay=0:(H-h)/2:enable='gte(t,12.733)'[p0];\
   [1:v]scale=$P:$P:flags=neighbor,tpad=stop_mode=clone:stop_duration=$D[s1];[s1][6:v]overlay=0:0[l1];[l1][7:v]overlay=0:(H-h)/2:enable='gte(t,13.533)'[p1];\
   [2:v]scale=$P:$P:flags=neighbor,tpad=stop_mode=clone:stop_duration=$D[s2];[s2][8:v]overlay=0:0[l2];[l2][9:v]overlay=0:(H-h)/2:enable='gte(t,18.133)'[p2];\
   [3:v]scale=$P:$P:flags=neighbor,tpad=stop_mode=clone:stop_duration=$D[s3];[s3][10:v]overlay=0:0[l3];[l3][11:v]overlay=0:(H-h)/2:enable='gte(t,18.200)'[p3];\
   [p0][p1][p2][p3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=#1a1a19[v]" \
     -map "[v]" -t $D -r 15 -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart grid.mp4
   ```

   The `enable` times are `(episode_length - 1) / fps` per panel.
6. Stack it with the synced curve panel for a Short ([[Animated training curves]]) and encode the final cut per [[Encoding GIFs and YouTube videos]].

## Results

`Craftax-Classic-Symbolic-v1`, seed 0, episodes of 192, 204, 273 and 274 steps, 15 fps, 4 panels of 270 px (540 x 540 output):

| Stage | Time | Output |
|---|---|---|
| Write 4 FFV1 panels from stored frames, make 8 PNGs | 0.34 s | 4 lossless `.mkv` |
| Same from replayed actions (4 jit compiles + render) | 31.42 s | identical frames |
| ffmpeg composite + libx264 crf 18 | 0.38 s | `grid.mp4`, 18.27 s, 274 frames, 2,789,565 B |

## Pitfalls

- `/opt/homebrew/bin/ffmpeg` 9.0.2 is built without libfreetype and libass: `drawtext`, `subtitles` and `ass` filters do not exist; text has to arrive as images.
- A PNG input to `overlay` is a single frame; the default `eof_action=repeat` keeps it on screen for the whole clip.
- Without `tpad=stop_mode=clone` a panel that died early goes black or ends the `xstack` output early.
- Use `flags=neighbor` for pixel art; bilinear scaling blurs the 16 px tiles.
- Classic nights darken the frame and add noise; a grid recorded across nightfall is hard to read at 270 px.
- With a shared seed the first frames of all panels are identical; the difference shows only after the first divergent action.
- The panel label shows the fixed-seed eval score; the curve shows training episodes; name both.

## Not tested

- A step counter per panel (needs per-frame PNGs or burning text into frames in Python).
- Other `xstack` layouts (1x3, 3x3); only 2x2 was rendered.
- More than one seed per checkpoint (grid of seeds x checkpoints).
- Choosing the seed automatically from eval returns, for example the seed where the late checkpoint beats the early one most; see [[Highlight detection]].
- 1080 x 1080 panels for the final export (only 540 x 540 rendered).
