---
title: "Encoding GIFs and YouTube videos"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, ffmpeg, youtube, gif]
created: 2026-10-20
updated: 2026-10-20
---
# Encoding GIFs and YouTube videos

Final encodes with the ffmpeg CLI (Homebrew ffmpeg 9.0.2): GIFs up to 20 s straight from the 16 px master, YouTube MP4s from the 1080p60 lossless clips of [[Assembling the 3 and 12 minute cuts]]. Text is drawn by Pillow into PNGs and composited with `overlay`, because this ffmpeg build has no text filters.

## YouTube upload targets

From YouTube Help, [Recommended upload encoding settings](https://support.google.com/youtube/answer/1722171), [Shorts up to 3 minutes](https://support.google.com/youtube/answer/15424877) and [Upload videos longer than 15 minutes](https://support.google.com/youtube/answer/71673), read 2026-10-20:

| Item | YouTube recommendation | Used here |
|---|---|---|
| Container | MP4, no edit lists, moov atom at the front | `-movflags +faststart -use_editlist 0` |
| Video codec | H.264, progressive, High profile, 2 consecutive B-frames, closed GOP of half the frame rate, CABAC, variable bitrate, 4:2:0 | `libx264 -profile:v high -bf 2 -g 30 -keyint_min 30 -sc_threshold 0 -flags +cgop -coder cabac -pix_fmt yuv420p`, CRF (VBR) |
| Frame rate | the recorded rate; 24, 25, 30, 48, 50, 60 common, others accepted | 60 fps output (10 game steps/s = every frame shown 6 times at 1x) |
| 1080p SDR bitrate | 8 Mbps (24 to 30 fps), 12 Mbps (48 to 60 fps) | CRF 12 lands at 2.5 to 4.8 Mbps on this content, see the quality table |
| Colour | BT.709 primaries, transfer and matrix; full range is converted to limited | RGB -> YUV with `scale=out_color_matrix=bt709:out_range=tv`, tags via `setparams` and `-colorspace/-color_primaries/-color_trc bt709 -color_range tv` |
| Audio | AAC-LC or Opus, stereo, 48 kHz (stereo 384 kbps recommended); the page says nothing about video without audio | silent AAC-LC stereo 48 kHz from `anullsrc` (encodes to about 2.3 kbps) |
| Aspect | 16:9 standard; vertical and square supported, the player adapts | 1920 x 1080 and 1080 x 1920 |
| Shorts | a square or vertical upload of at most 3 minutes is categorised as a Short; 16:9 keeps it long-form | vertical 171.4 s test cut |
| Length | 15 minutes per upload by default; longer after verifying the account | 12 min cut fits |
| 4K | viewing 4K needs a VP9-capable browser or device | one 2160p test clip |

## Steps

1. Final encode settings (`bin/assemble.sh`, `$ENC`):

   ```bash
   ENC="-c:v libx264 -preset slow -crf 12 -profile:v high -bf 2 -g 30 -keyint_min 30 -sc_threshold 0 -flags +cgop -coder cabac
    -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv
    -c:a aac -b:a 128k -ar 48000 -ac 2 -movflags +faststart -use_editlist 0"
   ffmpeg -i clips... -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 ... -shortest $ENC out.mp4
   ```

2. Verify every upload with `bin/verify_youtube.sh out.mp4`:

   ```bash
   ffprobe -v error -show_entries stream=codec_name,profile,level,width,height,pix_fmt,field_order,r_frame_rate,has_b_frames,color_space,color_primaries,color_transfer,color_range,sample_rate,channels:format=duration,bit_rate -of compact out.mp4
   ffprobe -v error -select_streams v -skip_frame nokey -show_entries frame=pts_time -of csv=p=0 out.mp4   # keyframe spacing
   ffprobe -v error -select_streams v -show_frames -read_intervals %+2 -show_entries frame=pict_type -of csv=p=0 out.mp4  # IBBP...
   ```

   plus a 10-line Python reader of the top-level MP4 boxes (`moov` before `mdat`) and a search for `elst` inside `moov`.

3. Captions and banners: `captions.py` (Pillow, `/System/Library/Fonts/Supplemental/Arial Bold.ttf` and `Arial.ttf`) writes full-frame RGBA PNGs: a caption panel left of the game (landscape) or above it (vertical), a freeze-frame banner ("UNLOCKED", "NEW FLOOR", "GAME OVER", ...) over the game, title cards, grid labels. ffmpeg composites them:

   ```bash
   ffmpeg -i clip_source -loop 1 -i panel.png -loop 1 -i banner.png -filter_complex \
     "[0]...[g];[g][1]overlay=0:0:shortest=1[c];[c][2]overlay=0:0:shortest=1:enable='between(t,12.0,13.0)'" ...
   ```

4. GIF (`bin/gif.sh <review.csv> <id> <out.gif>`, defaults: x3, no dither, `stats_mode=full`, 256 colours, clamp to 20 s of output time):

   ```bash
   ffmpeg -i masters/$rollout.mkv -i caps/${id}_gif.png -filter_complex \
     "[0]trim=start_frame=$start:end_frame=$((end + 1)),setpts=(PTS-STARTPTS)/$speed,fps=$((10 * speed)),scale=iw*3:ih*3:flags=neighbor[g];
      [1]scale=iw*3/3:ih*3/3:flags=bicubic[c];[g][c]vstack,split[a][b];
      [a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 out.gif
   ```

## Results

Final-encode options on one 20.1 s lossless 1080p60 clip (Craftax floor change, freeze and banner), VMAF and PSNR against that clip:

| Encoder settings | Size | Bitrate | Encode time | VMAF | PSNR (dB) |
|---|---|---|---|---|---|
| libx264 slow CRF 12 | 6.35 MB | 2,528 kb/s | 5.3 s | 97.32 | 61.59 |
| libx264 slow CRF 16 | 5.26 MB | 2,094 kb/s | 5.2 s | 97.18 | 57.20 |
| libx264 slow CRF 20 | 4.21 MB | 1,675 kb/s | 4.9 s | 96.78 | 53.57 |
| libx264 medium CRF 23 | 3.70 MB | 1,471 kb/s | 2.1 s | 96.31 | 50.65 |
| libx264 slow `-b:v 12M -maxrate 18M -bufsize 24M` | 9.29 MB | 3,698 kb/s | 5.7 s | 97.41 | 71.38 |
| h264_videotoolbox `-b:v 12M` | 5.78 MB | 2,298 kb/s | 5.4 s | 96.94 | 56.49 |

| Check on the produced cuts | Result |
|---|---|
| `short_3min_land.mp4` | 171.4 s, 1920 x 1080, 60 fps, H.264 High level 5.0, yuv420p, progressive, has_b_frames 2, BT.709 tagged, AAC-LC 48 kHz stereo, keyframes every 30 frames, `IBBPBBP...`, moov before mdat, no `elst`; 93.8 MB |
| `short_3min_vert.mp4` | same, 1080 x 1920, 91.7 MB |
| `long_12min_land.mp4` | 712.7 s (11:53), same checks pass, 431.8 MB, 4.8 Mb/s |
| Without `-use_editlist 0` | `elst` present in `moov` (B-frame delay); a `-c copy` remux with the flag removes it |
| Colour round trip (clip decoded back to RGB vs the 16 px master upscaled) | mean absolute difference 0.4, 99th percentile 3 levels; decoding with the wrong (BT.601) matrix: 99th percentile 14 |
| 2160p test (Craftax x10 = 1760 x 2080, padded to 3840 x 2160, CRF 12 slow, 19.1 s) | 6.72 MB, 10.0 s to encode; the same clip at 1080p: 5.50 MB, 4.5 s |

GIF options (20 s clips, size / encode time / PSNR of the game area against the un-paletted frames):

| Settings | Classic fight, 432 x 466, 195 frames | Craftax floor change, 528 x 658, 191 frames |
|---|---|---|
| x3, `dither=none`, `stats_mode=diff`, 256 | 3.04 MB, 0.4 s, 36.15 dB | 2.04 MB, 0.4 s, 43.93 dB |
| x3, `dither=none`, `stats_mode=full`, 256 (default) | 2.89 MB, 0.5 s, 38.42 dB | 2.00 MB, 0.5 s, 47.27 dB |
| x3, `dither=bayer:bayer_scale=3` | 5.44 MB, 0.5 s, 33.33 dB | 2.92 MB, 0.4 s, 34.12 dB |
| x3, `dither=sierra2_4a` (ffmpeg default) | 6.02 MB, 1.2 s, 34.02 dB | 2.05 MB, 0.6 s, 41.40 dB |
| x3, `dither=floyd_steinberg` | 5.22 MB, 1.2 s, 34.32 dB | 2.05 MB, 0.7 s, 41.61 dB |
| x3, `dither=none`, 64 colours | 2.05 MB, 0.4 s, 28.20 dB | 1.85 MB, 0.4 s, 28.98 dB |
| x2, `dither=none`, `stats_mode=diff` | 1.73 MB, 0.3 s, 36.14 dB | 1.16 MB, 0.3 s, 43.67 dB |
| x4, `dither=none`, `stats_mode=diff` | 4.51 MB, 0.6 s, 36.14 dB | 3.02 MB, 0.6 s, 43.78 dB |

- A 2x-speed GIF (191 master frames at 20 fps, 9.5 s): 2.14 MB.

## Pitfalls

- Homebrew's ffmpeg 9.0.2 bottle has no `drawtext` (no libfreetype) and no `subtitles`/`ass` (no libass): `ffmpeg -filters | grep -E "drawtext|subtitles"` prints nothing; draw text in Pillow and `overlay` it.
- swscale converts RGB to YUV with the BT.601 matrix unless `out_color_matrix=bt709` is set; tagging alone does not convert.
- With a filtergraph the `-color_primaries/-color_trc` output options were not written (ffprobe showed `unknown`) until `setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv` was added to the graph.
- Dithering pixel art adds noise and size: `dither=none` with `stats_mode=full` was both the smallest and the closest GIF in both tests.
- GIF delays are whole centiseconds: 10 fps (1x) and 20 fps (2x) are exact; 40 fps (4x) rounds to 2 or 3 cs.
- A bitrate target does not add quality here: `-b:v 12M` ended at 3.7 Mb/s because the frames are mostly repeats.
- `-vsync` is gone in ffmpeg 9; use `-fps_mode passthrough`.
- In shell scripts `ffmpeg` reads stdin; inside `while read` loops add `-nostdin` or it eats the loop input.

## Not tested

- Uploading to YouTube (processing, which renditions a 2160p upload gets, whether a silent track is needed).
- Animated WebP/AVIF instead of GIF: this ffmpeg build has no libwebp encoder.
- An ffmpeg build with libfreetype/libass for `drawtext`/`subtitles`.
- HEVC, AV1 (`libsvtav1` is present) or ProRes masters for upload.
- Whether even upscale factors (x10 at 2160p) reduce colour bleeding at tile edges: 4:2:0 chroma covers 2 x 2 pixels, so odd factors (x5, x7) split chroma samples across tiles.
