---
title: "Agent internals overlay"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, matplotlib, ffmpeg, policy, value]
created: 2026-10-20
updated: 2026-10-20
---
# Agent internals overlay

The game frame on the left, a synced panel on the right: the observation the policy receives, the action distribution with the chosen action highlighted, and the value estimate against the discounted return actually obtained, one row of eval data per frame. 1280 x 720 (16:9) for the long cut. Data: `eval/<update>.npz` and frames from [[Recording training data]].

## Steps

1. Load the per-step arrays of one eval episode: `probs` (T x 17), `value`, `reward`, `act`, `obs` (T x 1345 for Classic), `alive`; frame `t` shows the state before action `t`.
2. Compute what the value head is trying to predict, the discounted return from each step, and the policy entropy:

   ```python
   rtg = np.zeros_like(reward)
   for t in range(n_alive - 1, -1, -1):
       rtg[t] = reward[t] + 0.99 * (rtg[t + 1] if t + 1 < n_alive else 0.0)
   ent = -(probs * np.log(probs + 1e-8)).sum(1)
   ```

3. Decode the Classic symbolic observation (layout in [[Craftax observations and actions]]): a 7 x 9 map of 17 one-hot block channels plus 4 mob channels, then 12 inventory, 4 vitals, 4 direction, light, sleeping:

   ```python
   nb = len(BlockType)                                            # 17, craftax.craftax_classic.constants
   mp = obs[:, :7 * 9 * (nb + 4)].reshape(-1, 7, 9, nb + 4)
   blocks, mobs = mp[..., :nb].argmax(-1), mp[..., nb:]           # mobs: zombie, cow, skeleton, arrow
   inv = obs[:, 7 * 9 * (nb + 4): 7 * 9 * (nb + 4) + 12]
   vitals = obs[:, 7 * 9 * (nb + 4) + 12: 7 * 9 * (nb + 4) + 16]  # health, food, drink, energy (0..1)
   ```

4. Lay out a 560 x 720 matplotlib figure: map image (`imshow` with one fixed colour per block type, a white X at row 3, column 4 for the agent, scatter per mob type) and inventory/vitals bars on top; 17 action bars in the middle; value line over the discounted-return line with reward markers and a cursor at the bottom.
5. Per step update only data, never re-create artists, and pipe RGB to ffmpeg as in [[Animated training curves]]:

   ```python
   for t in range(s0, s1):
       im.set_data(blocks[t])
       for k, sc in enumerate(mob_sc):
           yy, xx = np.nonzero(mobs[t, :, :, k] > 0)
           sc.set_offsets(np.c_[xx, yy] if len(xx) else np.empty((0, 2)))
       for b, v in zip(inv_bars, np.r_[inv[t], vitals[t]]):
           b.set_width(min(1.0, float(v)))
       for k, b in enumerate(pi_bars):
           b.set_width(float(probs[t, k])); b.set_color("#d95926" if k == act[t] else "#3987e5")
       pi_title.set_text(f"policy  pi(a|s)   chosen: {acts[act[t]]}   entropy {ent[t]:.2f} nats")
       v_line.set_data(ts[s0: t + 1], value[s0: t + 1]); cursor.set_xdata([t, t])
       v_title.set_text(f"value V(s) = {value[t]:.2f}   reward r = {reward[t]:+.1f}")
       fig.canvas.draw(); pipe.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
   ```

6. Write the same steps of game frames to `game.mkv` (FFV1 raw pipe), then join:

   ```bash
   ffmpeg -y -i game.mkv -i internals.mkv -filter_complex \
     "[0:v]scale=720:720:flags=neighbor,pad=720:720:0:(oh-ih)/2:color=#1a1a19[g];[g][1:v]hstack=inputs=2,format=yuv420p[v]" \
     -map "[v]" -c:v libx264 -crf 18 -movflags +faststart overlay.mp4
   ```

## Results

Classic checkpoint at 2.0M steps (update 122), eval seed 0, steps 0 to 150, 10 fps:

| Stage | Time | Output |
|---|---|---|
| Internals panel, 560 x 720, 150 frames | 7.4 s (20.3 fps) | FFV1 |
| hstack + libx264 crf 18 | 0.4 s | `overlay.mp4`, 977,167 B |

- ffprobe: h264, 1280 x 720, 10/1, 150 frames, 15.0 s.
- Under heavy machine load an earlier 120-frame render ran at 6.5 fps.
- The decoded observation matched the rendered frame tile for tile (water, sand and the crafting table in the same cells).
- At step 120 of the test clip the agent faces water with 0.75 probability on `do` (drink); the value estimate dropped after the achievement rewards near step 70, as the remaining achievable return shrank.

## Pitfalls

- The symbolic map has no player channel; mark the agent at the centre cell (row 3, column 4) by hand.
- Bars for 17 actions fit at 7 pt in 720 px height; Craftax has 43 actions and needs a taller or two-column bar chart.
- Plot the value against the discounted return actually obtained, not the raw reward; otherwise the value line has nothing to be compared with.
- The value line jumps when the episode-end bootstrap is near; keep the y range from the visible window (`s0:s1`) or late steps look flat.
- PPO has no intrinsic reward; an exploration agent (RND, ICM, E3B in Craftax_Baselines) must log `reward_i` per eval step to fill a third trace.
- Keep frames and panel at the same fps and frame count; `hstack` ends at the shorter input.

## Not tested

- Decoding the Craftax (full) symbolic observation (8,268 values: 9 x 11 map with block, item, mob and light channels, plus inventory and status).
- An intrinsic-reward trace (no exploration agent was trained).
- Showing the pixel observation the policy sees for `Craftax-Pixels-v1` beside the human-sized render.
- Attention or saliency maps over the observation.
- A 9:16 layout of the overlay for Shorts.
