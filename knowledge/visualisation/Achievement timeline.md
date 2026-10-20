---
title: "Achievement timeline"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, matplotlib, achievements, heatmap]
created: 2026-10-20
updated: 2026-10-20
---
# Achievement timeline

An animated heatmap: one row per achievement (22 in Classic, 67 in Craftax; list in [[Craftax achievements]]), columns are PPO updates, colour is the success rate (% of training episodes), rows sorted by when the achievement was first unlocked, with a ring at the first unlock. The cursor sweeps left to right, a counter shows how many achievements have been unlocked at least once. Data: `metrics.npz` and `episodes.npz` from [[Recording training data]].

## Steps

1. First unlock per achievement = first update in which any training episode earned it, from the bit-packed episode log:

   ```python
   z = np.load("episodes.npz")
   ach = np.unpackbits(z["ach"], axis=-1, count=len(names)).astype(bool)        # (episodes, achievements)
   first = np.array([z["update"][np.nonzero(ach[:, i])[0][0]] if ach[:, i].any() else -1 for i in range(len(names))])
   order = np.lexsort((np.arange(len(first)), np.where(first < 0, 1 << 30, first)))   # never-unlocked rows last
   R = np.stack([ema(m.rates[:, i], 0.1) for i in range(len(names))])[order]        # (achievements, updates), %
   ```

2. Draw once with `pcolormesh` on a sequential one-hue ramp from the surface colour to light blue and a square-root norm, so 1 to 5% rates are visible:

   ```python
   from matplotlib.colors import LinearSegmentedColormap, PowerNorm
   cmap = LinearSegmentedColormap.from_list("seq", ["#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])
   edges = np.r_[x[0] - (x[1] - x[0]), x]                     # x = env steps (M) per update; log axis: np.r_[x[0] / 2, x]
   mesh = ax.pcolormesh(edges, np.arange(nA + 1), np.ma.masked_all(R.shape), cmap=cmap,
                        norm=PowerNorm(0.5, 0, 100), shading="flat")
   dots = ax.scatter(fx, np.arange(nA) + 0.5, s=18, facecolors="none", edgecolors="#f0efec", linewidths=1.2)
   ```

3. Per frame mask the columns right of the cursor, show rings already passed, update title and counter, write RGB to the ffmpeg pipe (same pipe as [[Animated training curves]]):

   ```python
   for i in cur:                                               # linspace over updates, or geomspace for --logx
       vis = R.copy(); vis[:, i + 1:] = np.nan
       mesh.set_array(np.ma.masked_invalid(vis).ravel())
       seen = (first[order] >= 0) & (first[order] <= i + 1)
       dots.set_offsets(np.c_[np.where(seen, fx[order], np.nan), np.arange(nA) + 0.5])
       count.set_text(f"unlocked at least once: {seen.sum()} / {nA}")
       fig.canvas.draw(); pipe.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
   ```

4. Hold the last frame 2 s and save it as a PNG thumbnail (`fig.savefig`).
5. Encode the FFV1 master: `ffmpeg -y -i timeline.mkv -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart timeline.mp4`.

## Results

| Run | Rows | Frames, size | Render | x264 | MP4 size |
|---|---|---|---|---|---|
| Classic, 2M steps, 122 updates, linear x | 22 | 180, 960 x 540 | 5.9 s | 0.26 s | 58,033 B |
| Same, log x | 22 | 180, 960 x 540 | 6.1 s | 0.21 s | 52,213 B |
| Craftax-Symbolic, 213k steps, 13 updates | 67 | 120, 960 x 1080 | 9.2 s | 0.25 s | 38,304 B |

- FFV1 masters: 7,806,434 B (Classic, linear) and 5,582,072 B (Craftax).
- Classic at 2M steps: 18 of 22 achievements unlocked at least once; never: collect iron, collect diamond, make iron pickaxe, make iron sword.
- First unlocks in the Classic run were spread over 0 to 0.5M steps except eat plant at 1.3M; on a linear axis they crowd the left 25% of the plot, the log axis spreads them.
- Craftax at 213k steps: 13 of 67 unlocked, including enter dungeon.

## Pitfalls

- A linear colour scale hides rare achievements (1 to 5%) in the near-black end; the square-root norm fixed it.
- The first-unlock ring marks the right edge of the update's column; a single lucky episode creates it, so a ring can precede any visible colour.
- At 67 rows a 5 pt label needs at least 1080 px height; a 16:9 frame fits 67 rows only with tiny labels.
- Achievements that were never unlocked keep their rows (all empty); dropping them hides the size of what is left.
- Training-episode rates count only episodes that finished in that update; see [[Recording training data]].

## Not tested

- A long Craftax run (the 67-row test used 13 updates).
- Grouping the 67 rows by floor or tier with separators ([[Game]] lists them).
- Eval-seed success rates per checkpoint instead of training episodes.
- Syncing the timeline cursor to a gameplay clip (the cursor logic of [[Animated training curves]] applies unchanged).
