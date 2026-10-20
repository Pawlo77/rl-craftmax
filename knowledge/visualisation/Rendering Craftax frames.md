---
title: "Rendering Craftax frames"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, rendering, jax, ffmpeg]
created: 2026-10-20
updated: 2026-10-20
---
# Rendering Craftax frames

Record an episode as seed + actions + per-step features, replay it inside `jax.lax.scan` with the pixel renderer, and pipe the frames into a lossless RGB master video with one env step per frame. Every number below comes from runs on an Apple M4 Pro (14 cores, 48 GB, CPU JAX) with `craftax` 1.6.1, `jax` 0.12.0, `numpy` 2.5.4, Pillow 13.0.0, ffmpeg 9.0.2; other jobs were running on the machine (load average 3 to 32), so timings are indicative.

## Steps

1. Environment (outside the repo, it never touches `pyproject.toml`):

   ```bash
   uv venv --python 3.12 venv
   uv pip install --python venv/bin/python craftax imageio pillow numpy
   ```

2. One interface for both games (`vis/game.py`). The NoAutoReset env keeps the death frame; any tile size works once its textures are added to the cached dict the renderer reads:

   ```python
   from craftax.craftax_env import make_craftax_env_from_name
   NAMES = {"craftax": "Craftax-Symbolic-v1", "classic": "Craftax-Classic-Symbolic-v1"}

   class Game:
       def __init__(self, kind, god_mode=False):
           self.kind, self.env = kind, make_craftax_env_from_name(NAMES[kind], auto_reset=False)
           self.params = self.env.default_params.replace(god_mode=True) if god_mode else self.env.default_params
           if kind == "craftax":
               from craftax.craftax import constants as C
               from craftax.craftax.renderer import make_craftax_pixel_renderer as mk
           else:
               from craftax.craftax_classic import constants as C
               from craftax.craftax_classic.renderer import make_craftax_pixel_renderer as mk
           self.C, self._mk, self.obs_dim = C, mk, C.OBS_DIM
           self.achievements = [a.name for a in sorted(C.Achievement, key=lambda a: a.value)]
           self.reset, self.step, self.features = jax.jit(self.env.reset), jax.jit(self.env.step), jax.jit(self._features)

       @functools.cache
       def renderer(self, bps):  # bps = screen pixels per tile
           tex = self.C.load_all_textures()          # lru-cached dict {7|10, 16, 64: textures}
           if bps not in tex:
               tex[bps] = self.C.load_all_textures_given_size(bps)   # e.g. 80 -> Craftax 880x1040
           f = self._mk(bps) if self.kind == "classic" else self._mk(bps, do_night_noise=True)
           return lambda s: f(s).astype(jnp.uint8)

   def keys(seed, scheme="fold_in"):  # same convention as the training recorder
       base = jax.random.PRNGKey(seed)
       return (base, jax.random.fold_in(base, 1)) if scheme == "fold_in" else tuple(jax.random.split(base))
   ```

   `_features(state)` returns a float vector (health, food, drink, energy, floor, sleeping, melee / ranged / projectiles on screen, hostiles within 2 tiles, boss progress, row, col) plus the achievement vector; it is jittable so it also runs inside the replay scan. Mob arrays: Craftax `melee_mobs`, `ranged_mobs`, `mob_projectiles` indexed `[player_level]`; Classic `zombies`, `skeletons`, `arrows`.

3. Rollout (`rollout.py <craftax|classic> <policy> <seed> <out.npz> [max_steps] [--god]`). A policy is any callable `policy(state, rng) -> int`; a trained network drops in as `jnp.argmax(network.apply(params, env.get_obs(state)))` (template `NetworkPolicy` ran with random weights):

   ```python
   k_reset, k_step = keys(seed)
   _, s = g.reset(k_reset, g.params)

   @jax.jit
   def step(t, s, a):
       _, s, r, d, _ = g.step(jax.random.fold_in(k_step, t), s, a, g.params)
       f, ach = g.features(s)
       return s, r, d, f, ach

   for t in range(max_steps):
       a = policy(s, rng)
       s, r, d, f, ach = step(t, s, a)
       ...  # append a, r, f, ach
       if bool(d): break
   np.savez_compressed(out, key_scheme="fold_in", kind=kind, policy=pol, seed=seed, god=god, actions=..., rewards=...,
                       features=..., feature_names=..., achievements=..., achievement_names=...)
   ```

4. Master render (`render.py <rollout.npz>... --out-dir masters [--bps 16] [--fps 10] [--start a --end b]`): replay in chunks of 256 steps, steps before `--start` without rendering, frames piped to ffmpeg; one compile per (game, tile size) serves every rollout:

   ```python
   @jax.jit
   def run(k, s, ts, acts):
       def body(s, ta):
           _, s, *_ = g.step(jax.random.fold_in(k, ta[0]), s, ta[1], g.params)
           return s, (render(s), g.features(s)[0])
       return jax.lax.scan(body, s, (ts, acts))

   cmd = f"ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s {W}x{H} -r 10 -i - -c:v libx264rgb -qp 0 -preset medium"
   ff = subprocess.Popen([*cmd.split(), out], stdin=subprocess.PIPE)
   ff.stdin.write(first_frame.tobytes())        # frame 0 = reset state, so frame index == step index
   for i in range(start, end, 256):             # last chunk padded to 256 so the compiled shape is reused
       s, (frames, f) = run(k_step, s, jnp.arange(i, i + 256), padded_actions)
       ff.stdin.write(np.asarray(frames[:n]).tobytes())
   ```

   The replayed features are compared with the recorded ones after every render.

5. Upscale for video with ffmpeg only, integer factors and `flags=neighbor`, then pad: Craftax `scale=iw*5:ih*5:flags=neighbor,pad=1920:1080:520:20` (880x1040), Classic `scale=iw*7:ih*7:flags=neighbor,pad=1920:1080:456:36` (1008x1008); vertical 1080x1920: Craftax x6 at (12, 336), Classic x7 at (36, 456).

6. Best-quality frames for the final cut: render only the kept windows at the target size, `render.py run.npz --out clip.mkv --bps 80 --start 880 --end 1080` (Craftax 880x1040 = 16 px x5; Classic `--bps 112` = 1008x1008 = 16 px x7, `--bps 120` = 1080x1080), or every kept window of a reviewed CSV with `render.py --kept review.csv --layout land --out-dir hires/` (used in [[Assembling the 3 and 12 minute cuts]]). Same map pixels as 16 px upscaled, sharper HUD icons and digits.

7. Checks that were run:

   ```bash
   # range render == the same frames of the full master
   ffmpeg -nostdin -i masters/craftax_heur_6.mkv -vf "select=between(n\,880\,1080)" -fps_mode passthrough -f rawvideo -pix_fmt rgb24 - | md5
   ffmpeg -nostdin -i range_b16.mkv -f rawvideo -pix_fmt rgb24 - | md5
   # lossless master codecs: decode both to rgb24 and compare md5 (bench_master_codecs.sh)
   ```

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
| Craftax 80 px master (880 x 1040), 2048 frames | 66.1 s (31 fps), 925 MB |
| Classic 120 px master (1080 x 1080), 923 frames | 61.6 s (15 fps, load average about 30), 296 MB (FFV1) |
| Lossless 16 px masters (libx264rgb -qp 0) | Classic 12.2 MB, Craftax 15.5 MB per 1000 frames |
| Rollout file (seed + actions + features + achievements) | 8.5 KB for a 2,047-step Craftax episode |
| Full `EnvState` per step, if states were stored instead | Craftax 273,276 bytes, Classic 20,952 bytes |
| Replayed features == recorded features | 23 of 23 rollouts, both key schemes |
| Range render (frames 880 to 1080) == slice of full master | bit-identical |
| Frames from a live per-frame `jit(render)` vs the scan replay | Craftax 18 of 301 frames differ, Classic 7 of 195; max difference 1 intensity level, 1.9e-6 and 2.2e-6 of pixels |

Lossless master codecs on one 2,048-frame Craftax master (176 x 208, 10 fps), all decode bit-exact to the source RGB:

| Codec | Size | Encode time |
|---|---|---|
| `-c:v ffv1 -level 3 -g 1` | 84.6 MB | (inside render) |
| `-c:v ffv1 -level 3 -coder 1 -context 1 -g 1` | 58.5 MB | 5.5 s |
| `-c:v png` (in MKV) | 50.8 MB | 2.4 s |
| `-c:v libx264rgb -qp 0 -preset medium` | 41.0 MB | 5.7 s |
| `-c:v libx264rgb -qp 0 -preset veryslow` | 40.1 MB | 10.6 s |

Test rollouts (`make_rollouts.sh`, 3 min wall clock for all 23; Python loop 85 to 614 steps/s including the first-step compile):

| Set | Episodes | Length (steps) | Achievements per episode | Notes |
|---|---|---|---|---|
| Classic, heuristic | 8 | 194 to 922 | 15 to 18 | all die |
| Craftax, heuristic | 8 | 283 to 2,047 | 15 to 20 | 4 reach the Dungeon (floor 1) |
| Classic, random | 3 | 78 to 256 | 1 to 4 | |
| Craftax, random | 3 | 217 to 367 | 1 to 3 | |
| Craftax, heuristic, god mode | 1 | 1,500 (cap) | 39 | descends every 60 steps, reaches the Graveyard (floor 8) |

## Heuristic test policy

- BFS over walkable tiles (`GRASS, SAND, PATH, FIRE_GRASS, ICE_GRASS, GRAVEL`) to the nearest tile next to a target, face it, `DO`.
- Priority: attack an adjacent hostile; `SLEEP` at energy < 3; water at drink < 5; cow at food < 5; wood; `PLACE_TABLE`; wood pickaxe and sword; sapling; stone; `PLACE_STONE`; stone tools; `PLACE_FURNACE`; on Craftax after step 600 walk to `down_ladders[level]` and `DESCEND`; coal; iron; diamond; else a random move.
- 3% random moves and a random move after 12 steps without moving.
- God mode (`EnvParams.god_mode`): `DESCEND` works anywhere and walls are passable, the player still takes damage; used only to get footage of every floor.

## Pitfalls

- `Achievement` enum order is not value order in Craftax (`MAKE_DIAMOND_SWORD = 25` comes after `COLLECT_RUBY = 59`); index names by `.value`.
- The auto-reset env replaces the terminal state with a fresh world, so the death frame is lost; use `auto_reset=False`.
- `make_craftax_pixel_renderer(bps)` raises `KeyError` for a tile size outside 7/10, 16, 64 unless its textures are added to `load_all_textures()` first.
- Replay must use the recorder's key derivation; the test rollouts used `split(PRNGKey(seed))`, the training recorder in [[Recording training data]] uses `fold_in(base, 1)`; the file stores `key_scheme`.
- A live per-frame `jit(render)` and a `scan`/`vmap` render of the same state can differ by 1 intensity level in a few pixels (XLA fusion); render masters and clips with the same path when bit-exactness matters.
- Night noise is drawn per screen pixel from `state.state_rng`: deterministic, but at 80 px it is fine static; the 80 px master is 22.6 times the 16 px one (925 MB vs 41.0 MB) for 25 times the pixels.
- Integer upscales only; a non-integer `neighbor` scale (for example 832 to 1080) gives uneven pixel widths.
- The first chunk of every new (game, tile size, god mode) combination pays the compile (10 to 45 s depending on load); render many rollouts in one process.

## Not tested

- The pygame play GUI (`play_craftax`) as a recorder: it renders the 64 px frame and repeats pixels, so it adds nothing over the 64 px renderer above.
- Rendering on a GPU or the other cluster nodes.
- Full-map renders (`OBS_DIM` set to the map size, see [[Craftax API and wrappers]]).
