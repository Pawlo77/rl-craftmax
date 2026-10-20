---
title: "Agent internals overlay"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, matplotlib, ffmpeg, policy, value]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/visualisation/overlay.py]]"
  - "[[src/wut_murl_craftax/visualisation/eval_rollouts.py]]"
  - "[[tests/visualisation/test_overlay.py]]"
---
# Agent internals overlay

The game frame on the left, a synced panel on the right: the observation the policy receives, the action distribution with the chosen action highlighted, and the value estimate against the discounted return actually obtained, one eval step per frame. Default 1920 x 1080 (1080 px game + 840 x 1080 panel). Data: `eval/<update>.npz` from [[Recording training data]]; works for Craftax and Craftax-Classic.

## Steps

1. Render a checkpoint's eval episode:

   ```bash
   uv run python -m wut_murl_craftax.visualisation.overlay outputs/runs/classic --update 122 --seed 0 --steps 0:300 --fps 10
   ```

2. What the panel shows, per step `t` (the frame shows the state before action `t`):
   - observation, from `decode_obs` ([[Craftax observations and actions]]): Classic 7 x 9 map with 17 block one-hots and 4 mob channels, then 12 inventory and 4 vitals; Craftax 9 x 11 x 83 map (37 blocks, 40 creature channels collapsed to 5 classes) and 13 inventory/vitals values; blocks are coloured with the mean colour of their game texture, the agent marked with an X at the centre;
   - `pi(a|s)` for the 17 or 43 actions, chosen action in orange, entropy in the title;
   - value `V(s)` over the discounted return obtained from each step (`discounted_return`, gamma 0.99), cursor at `t`.
3. The panel is a `_Panel` figure updated in place and streamed frame by frame into `ffmpeg.FrameWriter`; the game frames come from `episode_frames` (replay); one ffmpeg call scales the game with `flags=neighbor`, pads it to the panel height and `hstack`s both.

## Results

Prototype, Classic checkpoint at 2.0M steps, seed 0, steps 0 to 150, 10 fps, 560 x 720 panel (1280 x 720):

| Stage | Time | Output |
|---|---|---|
| Panel, 150 frames | 7.4 s (20.3 fps) | lossless |
| hstack + libx264 crf 18 | 0.4 s | 977,167 B; h264, 1280 x 720, 10/1, 150 frames, 15.0 s |

Repo code, 540 px game + 420 x 540 panel:

| Run | Steps | Wall | Output |
|---|---|---|---|
| Classic, 0.5M steps, update 30 | 0 to 100 | 12.4 s | 960 x 540, 100 frames, 554,371 B |
| Craftax-Symbolic, 8,192 steps, update 8 | 0 to 60 | 15.8 s | 960 x 540 |

- The decoded Classic observation matched the rendered frame tile for tile (water, sand, crafting table); in Craftax the tree and grass layout matched by eye.
- In the prototype clip at step 120 the agent faced water with 0.75 probability on `do` (drink); the value estimate dropped after the achievement rewards near step 70.

## Pitfalls

- The symbolic map has no player channel; the agent is drawn at the centre cell.
- 43 Craftax action labels are unreadable at 540 px height; use the default 1080 px or fewer labels.
- Plot the value against the discounted return obtained, not the raw reward.
- Keep the y range from the visible window (`s0:s1`), or late steps look flat.
- Streaming matters: 300 panels at 840 x 1080 held in memory would take 0.8 GB.
- PPO has no intrinsic reward; an exploration agent (RND, ICM, E3B in Craftax_Baselines) must log `reward_i` per eval step for a third trace.

## Not tested

- An intrinsic-reward trace (no exploration agent was trained).
- The default 1920 x 1080 size (smoke runs used 960 x 540).
- The pixel observation of `Craftax-Pixels-v1` beside the human-sized render.
- Attention or saliency maps.
- A 9:16 layout for Shorts.
