---
title: "Rendering Craftax frames"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, rendering, jax, ffmpeg]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/visualisation/rollout.py]]"
  - "[[src/wut_murl_craftax/visualisation/render.py]]"
  - "[[src/wut_murl_craftax/visualisation/ffmpeg.py]]"
  - "[[tests/visualisation/test_rollout.py]]"
  - "[[tests/visualisation/test_render.py]]"
---
# Rendering Craftax frames

An episode is recorded as seed + actions + per-step features (`Rollout`), replayed inside `jax.lax.scan` with the pixel renderer, and piped into a lossless RGB master with one env step per frame. Measurements: Apple M4 Pro (14 cores, 48 GB, CPU JAX), `craftax` 1.6.1, ffmpeg 9.0.2; first runs with `jax` 0.12.0, the package in the repo env (`jax` 0.11.2) reproduces them bit for bit; other jobs ran on the machine (load average 3 to 32), so timings are indicative.

## Steps

1. Play episodes; one file `<game>_<heur|rand|net|god>_<seed>.npz` each in `outputs/visualisation/runs/`:

   ```bash
   uv run python -m wut_murl_craftax.visualisation.rollout --game classic --seeds 1 2 3
   uv run python -m wut_murl_craftax.visualisation.rollout --game craftax --policy random --seeds 1 2
   uv run python -m wut_murl_craftax.visualisation.rollout --game craftax --god --seeds 1 --max-steps 1500
   ```

2. A trained agent plugs in as a `Policy` (any `(state, rng) -> int` with a `name`); `NetworkPolicy` wraps a network, the label shown on screen comes from `Rollout.label` (`random policy`, `scripted agent`, `trained agent`, `god-mode dive`, or any text such as a checkpoint name):

   ```python
   from wut_murl_craftax.visualisation.rollout import NetworkPolicy, game_for, play
   game = game_for("craftax")
   policy = NetworkPolicy(game, apply=network.apply, params=params)   # argmax over the logits
   play(game, policy, seed=3, label="checkpoint 2M steps").save(Path("outputs/visualisation/runs/craftax_net_3.npz"))
   ```

3. Keys: `keys(seed)` returns `(PRNGKey(seed), fold_in(PRNGKey(seed), 1))`, the training recorder's scheme; step `t` uses `fold_in(env_key, t)`; the file stores `key_scheme`, and files without it are the first `split` runs.

4. Masters (16 px per tile, lossless `libx264rgb -qp 0`, one compile per game and tile size for all files):

   ```bash
   uv run python -m wut_murl_craftax.visualisation.render outputs/visualisation/runs/*.npz
   ```

   `replay(rollout, bps=16, start, end, chunk=256)` streams `ReplayChunk(start, features, frames)`; steps before `start` replay without rendering; every render compares the replayed features with the recorded ones (`RenderStats.features_match`).

5. Best-quality frames for the final cut: only the kept windows of a reviewed CSV, at 16 px x the layout factor (`FINAL_BPS`: `land` Craftax 80 px = 880 x 1040, Classic 112 px = 1008 x 1008; `vert` 96 / 112; `land4k` 160 / 224):

   ```bash
   uv run python -m wut_murl_craftax.visualisation.render --kept outputs/visualisation/review/review.csv --layout land --out-dir outputs/visualisation/hires_land
   ```

   `Game.renderer(bps)` builds textures for sizes the package does not cache (it ships 7/10, 16, 64) and adds them to the dict the renderer reads.

6. Upscale for video only with integer factors and `flags=neighbor`, then pad (layouts in [[Assembling the 3 and 12 minute cuts]]).

## Results

Renderer speed, 256 random-policy states, frames per second ("per-frame" = jitted renderer called once per state from Python, including slicing the stacked state; "scan" = env step + render inside one `lax.scan`), measured at load average 3 to 5:

| Game | Tile px | Frame (H x W) | Per-frame jit | vmap(256) | scan step+render | MB per 1000 raw frames |
|---|---|---|---|---|---|---|
| Craftax | 10 (agent) | 130 x 110 | 141 | 3,769 | 2,527 | 43 |
| Craftax | 16 | 208 x 176 | 337 | 1,601 | 1,123 | 110 |
| Craftax | 64 (play GUI) | 832 x 704 | 78 | 164 | 81 | 1,757 |
| Classic | 7 (agent) | 63 x 63 | 125 | 34,069 | 8,502 | 12 |
| Classic | 16 | 144 x 144 | 207 | 6,942 | 3,466 | 62 |
| Classic | 64 (play GUI) | 576 x 576 | 107 | 627 | 252 | 995 |

| Measurement | Result |
|---|---|
| Texture cache build (first pixel use) | Craftax 38.2 s, Classic 9.6 s; later loads 1.1 s and 0.4 s; cache files 1.75 MB and 1.26 MB inside the installed package |
| 64 px render vs 16 px upscaled x4 nearest | map area identical; 27.6% (Craftax) and 31.4% (Classic) of pixels differ, all in the inventory/HUD rows (icons drawn at 0.8 x tile, digits at 0.4 x tile) |
| All 23 test rollouts to 16 px masters, one process | 14,986 frames in 90.7 s including 3 compiles (165 fps); 350 to 650 fps per master after compile |
| Repo smoke test, 7 rollouts to masters | 7,939 frames in 30.7 s including compiles |
| Craftax 80 px master (880 x 1040), 2048 frames | 66.1 s (31 fps), 925 MB |
| 29 kept windows at 80 / 112 px (repo `render --kept`) | 2,865 frames in 153.3 s including 3 compiles |
| Lossless 16 px masters (`libx264rgb -qp 0`) | Classic 12.2 MB, Craftax 15.5 MB per 1000 frames |
| Rollout file (seed + actions + features + achievements) | 8.5 KB for a 2,047-step Craftax episode |
| Full `EnvState` per step, if states were stored instead | Craftax 273,276 bytes, Classic 20,952 bytes |
| Replayed features == recorded features | 23 of 23 rollouts, both key schemes |
| Range render == slice of the full master | bit-identical (test `test_range_render_equals_master_slice`) |
| Package port vs first scripts | Classic and Craftax heuristic seed 1: same 390 and 769 actions and features; masters of `classic_heur_7` (923 frames) and `craftax_god_1` (1,501 frames) bit-identical |
| Frames from a live per-frame `jit(render)` vs the scan replay | Craftax 18 of 301 frames differ, Classic 7 of 195; max difference 1 intensity level, 1.9e-6 and 2.2e-6 of pixels |

Lossless master codecs on one 2,048-frame Craftax master (176 x 208, 10 fps), all decode bit-exact to the source RGB:

| Codec | Size | Encode time |
|---|---|---|
| `-c:v ffv1 -level 3 -g 1` | 84.6 MB | (inside render) |
| `-c:v ffv1 -level 3 -coder 1 -context 1 -g 1` | 58.5 MB | 5.5 s |
| `-c:v png` (in MKV) | 50.8 MB | 2.4 s |
| `-c:v libx264rgb -qp 0 -preset medium` (used) | 41.0 MB | 5.7 s |
| `-c:v libx264rgb -qp 0 -preset veryslow` | 40.1 MB | 10.6 s |

Test rollouts (23 episodes in 3 min; Python loop 85 to 614 steps/s including the first-step compile):

| Set | Episodes | Length (steps) | Achievements per episode | Notes |
|---|---|---|---|---|
| Classic, heuristic | 8 | 194 to 922 | 15 to 18 | all die |
| Craftax, heuristic | 8 | 283 to 2,047 | 15 to 20 | 4 reach the Dungeon (floor 1) |
| Classic, random | 3 | 78 to 256 | 1 to 4 | |
| Craftax, random | 3 | 217 to 367 | 1 to 3 | |
| Craftax, heuristic, god mode | 1 | 1,500 (cap) | 39 | descends every 60 steps, reaches the Graveyard (floor 8) |

## Heuristic policy (`HeuristicPolicy`)

- BFS over walkable tiles (`GRASS, SAND, PATH, FIRE_GRASS, ICE_GRASS, GRAVEL`) to the nearest tile next to a target, face it, `DO`.
- Priority: attack an adjacent hostile; `SLEEP` at energy < 3; water at drink < 5; cow at food < 5; wood; `PLACE_TABLE`; wood pickaxe and sword; sapling; stone; `PLACE_STONE`; stone tools; `PLACE_FURNACE`; on Craftax after step 600 walk to `down_ladders[level]` and `DESCEND`; coal; iron; diamond; else a random move.
- 3% random moves and a random move after 12 steps without moving.
- God mode (`--god`, `EnvParams.god_mode`): `DESCEND` works anywhere and walls are passable, the player still takes damage; footage of every floor for the game intro.

## Pitfalls

- `Achievement` enum order is not value order in Craftax (`MAKE_DIAMOND_SWORD = 25` comes after `COLLECT_RUBY = 59`); `Game.achievement_names` is indexed by `.value`.
- The auto-reset env replaces the terminal state with a fresh world, so the death frame is lost; `Game` uses `auto_reset=False`.
- `make_craftax_pixel_renderer(bps)` raises `KeyError` for a tile size outside 7/10, 16, 64 unless its textures are added first (`Game.renderer` does).
- Replay must use the recorder's key scheme: with threefry `split(k)[1] == fold_in(k, 1)`, so `split` and `fold_in` share the step keys and differ only in the reset key, which changes the world.
- A live per-frame `jit(render)` and a `scan`/`vmap` render of the same state can differ by 1 intensity level in a few pixels (XLA fusion); masters and windows both come from `replay`, so they match each other bit for bit.
- Night noise is drawn per screen pixel from `state.state_rng`: deterministic, but at 80 px it is fine static; the 80 px master is 22.6 times the 16 px one (925 MB vs 41.0 MB) for 25 times the pixels.
- Integer upscales only; a non-integer `neighbor` scale (for example 832 to 1080) gives uneven pixel widths.
- The first chunk of every new (game, tile size, god mode) combination pays the compile (10 to 45 s depending on load); render many rollouts in one process.

## Not tested

- The pygame play GUI (`play_craftax`) as a recorder: it renders the 64 px frame and repeats pixels, so it adds nothing over the 64 px renderer above.
- Rendering on a GPU or the cluster nodes.
- Full-map renders (`OBS_DIM` set to the map size, see [[Craftax API and wrappers]]).
