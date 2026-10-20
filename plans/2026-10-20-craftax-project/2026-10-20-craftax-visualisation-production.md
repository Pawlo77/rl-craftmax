---
title: "Craftax visualisation production implementation plan"
type: plan
domain: visualisation
parent: "[[2026-10-20-craftax-project]]"
status: draft
created: 2026-10-20
updated: 2026-10-20
tags: [plan, visualisation, video, youtube, gif]
---
# Craftax visualisation production implementation plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** with the existing pipeline in `src/wut_murl_craftax/visualisation/` ([[Visualisation]]), produce the GIFs (at most 20 s each), the 3-minute video as a 16:9 upload and as a vertical YouTube Short (at most 180 s), and the 12-minute 16:9 video, all 1080p with voice-over and ducked music, showing the trained agents of [[2026-10-20-craftax-project]] in interesting situations: achievement unlocks, multi-enemy fights, and a long-stuck problem being solved.

**Architecture:** `visualisation/footage.py` turns `configs/video/footage.yaml` into rollouts (trained agents, early checkpoints, random policy, scripted agent, god-mode dive; Craftax main, Craftax-Classic warm-up); the existing render, highlight and review steps produce one reviewer CSV ([[Highlight review workflow]]); a new `visualisation/storyboard.py` reads `configs/video/short.yaml` and `configs/video/long.yaml` (scene list with durations, sources and narration) and assembles clips, cards, stills of the result figures and training graphics with `visualisation.cuts`, then muxes voice and music with `ffmpeg.mux_audio`. Decisions already taken: [[Visualisation decisions]].

**Tech Stack:** JAX + craftax 1.6.1 renderers, Pillow, matplotlib, ffmpeg 9 CLI (no `drawtext`), PyYAML, existing modules `rollout`, `render`, `highlights`, `review`, `cuts`, `curves`, `grid`, `timeline`, `overlay`, `montage`, `ffmpeg`, `text`.

## Footage

| Kind | Which | Label on screen | Use |
|---|---|---|---|
| Trained agent, final | final checkpoint of the best seed and the median seed per method, 1B and 1M, eval seeds `10000` to `10063` | method, budget, "final" | highlights, contenders chapter |
| Early checkpoints | 4 log-spaced checkpoints of the best 1B seed per method ([[Checkpoint comparison grid]]) | env steps, fixed-seed eval score | grid, before-learning scene |
| Random policy | `RandomPolicy`, Craftax and Classic, 8 seeds each | "random policy" | before-learning, warm-up |
| Scripted agent | `HeuristicPolicy`, Craftax and Classic, 8 seeds each ([[Rendering Craftax frames]]) | "scripted agent" | warm-up, contenders contrast |
| God-mode dive | `HeuristicPolicy` with `god_mode=True`, 2 seeds | "god mode tour" | intro only, never as agent footage |
| Cases | episodes flagged `video = yes` in `outputs/analysis/<budget>/cases.csv` ([[2026-10-20-craftax-training-analysis]]) | method, case kind | stuck-then-solved, deepest floor, deaths |
| Classic trained | final checkpoint of the PPO debugging run on Craftax-Classic | "Craftax-Classic, PPO" | warm-up only |

## Storyboard: 3-minute video (16:9 and Short)

Durations are on-screen seconds after crossfades; total 175 s, so the Short stays under 180 s. The Short uses the same scenes in the `vert` layout.

| # | Scene | Seconds | Content | Source |
|---|---|---|---|---|
| 1 | Cold open | 8 | the best trained agent's top kept highlight, freeze-frame banner | top trained row of `review.csv` |
| 2 | Title | 3 | title card | `cuts.card` |
| 3 | What is Craftax | 15 | god-mode dive through floors 0 to 8, floor names as captions | god-mode master |
| 4 | Warm-up on Classic | 12 | random policy vs trained agent on Craftax-Classic, side by side | `cuts.grid` |
| 5 | Before learning | 12 | Craftax random policy and the earliest checkpoint dying | `cuts.grid` |
| 6 | Learning curve | 15 | headline curve, % of max reward per method, achievements panel | `curves` (multi-run) |
| 7 | Same world, more training | 15 | one fixed seed, 4 checkpoints of the best method, panels labelled with the fixed-seed eval score | `grid`, `montage short` (vertical) |
| 8 | Unlocks | 25 | 3 to 4 achievement unlocks of trained agents | kept rows, type achievement |
| 9 | Many enemies at once | 15 | 1 to 2 multi-enemy fights | kept rows, type fight |
| 10 | Stuck, then solved | 20 | a long stall at speed 8, then the unlock at speed 1 | stall-breaker case from `cases.csv`: stall start to unlock - 30 at speed 8, then unlock - 30 to unlock + 20 at speed 1 |
| 11 | Who unlocks first | 12 | first-unlock race chart (F6 of [[2026-10-20-craftax-comparison-and-results]]) | still |
| 12 | Playstyles | 10 | playstyle radars (F17) | still |
| 13 | Scores | 8 | headline numbers vs controls and the scoreboard | still of T1 |
| 14 | Outro | 5 | links to the Space, the report and the 12-minute video; music credit | card |
| | **Total** | **175** | | |

## Storyboard: 12-minute video (16:9)

Total 720 s; each chapter starts with a 3 s chapter card counted inside its first scene; chapters become YouTube chapters.

| Chapter | Scene | Seconds | Content | Source |
|---|---|---|---|---|
| 0 Intro | Cold open | 15 | best moment of the whole project | kept row |
| 0 Intro | Title | 5 | title card | card |
| 1 What is Craftax | God-mode tour | 40 | floors 0 to 8, creatures, resources | god-mode master |
| 1 What is Craftax | Achievements | 10 | 67 achievements, tiers 1 / 3 / 5 / 8, max 226 ([[Craftax scoring and evaluation]]) | still |
| 1 What is Craftax | Benchmarks | 10 | Craftax-1M and Craftax-1B, scoreboard metric | still |
| 2 Warm-up on Classic | Random | 10 | random policy on Classic | master |
| 2 Warm-up on Classic | Scripted | 10 | scripted agent on Classic | master |
| 2 Warm-up on Classic | Trained | 15 | trained agent on Classic | kept row |
| 2 Warm-up on Classic | Classic curve | 10 | Classic training curve, % of 22 | `curves` |
| 3 The contenders | One per method | 120 | per method and control: 5 s card (idea in one line) + 15 s clip of its final agent; 6 entries; with 5 entries the spare 20 s go to chapter 6 | cards, kept rows |
| 4 How they learn | Curves | 30 | headline curves, all methods | `curves` (multi-run) |
| 4 How they learn | Achievement timeline | 30 | best method's 67-row heatmap ([[Achievement timeline]]) | `timeline` |
| 4 How they learn | Checkpoint grid | 30 | 4 checkpoints, one seed | `grid` |
| 4 How they learn | Inside the agent | 30 | observation, action probabilities and value at a breakthrough ([[Agent internals overlay]]) | `overlay` |
| 5 Results | Score table | 20 | T1 and T2 headline rows | still |
| 5 Results | First-unlock race | 25 | F6 | still |
| 5 Results | Order of achievements | 20 | F7 for two contrasting methods | still |
| 5 Results | Progress funnel | 20 | F8 | still |
| 5 Results | Playstyles | 25 | F17 | still |
| 6 Highlights | Unlocks | 50 | 6 to 8 unlocks, rare ones first | kept rows, type achievement |
| 6 Highlights | Fights | 40 | 3 to 4 multi-enemy fights | kept rows, type fight |
| 6 Highlights | Stuck, then solved | 40 | 2 stall breakers, time-lapse then 1x | stall-breaker cases, two segments each |
| 6 Highlights | Deepest floor | 30 | the deepest descent of the project | kept rows, type floor |
| 7 How they fail | Deaths | 20 | one clip per top death cause | kept rows, type death |
| 7 How they fail | Stalls never escaped | 15 | time-lapse of an unescaped stall | case clip |
| 8 What the agent learned | Decision rules | 30 | 16-leaf tree rules in words, fidelity vs leaves ([[2026-10-20-craftax-interpretation-and-ruleset]]) | still |
| 9 Outro | Credits | 20 | team, Craftax credit, music licence, links | card |
| | **Total** | **720** | | |

## Audio

- Narration lives in the storyboard YAML (`narration:` per scene); budget 2.3 words per second of scene, so about 400 words for the 3-minute video and 1,650 for the 12-minute one.
- Every number in the narration comes from `knowledge/results/Final results.md` (executed output).
- Recording: one WAV per scene, 48 kHz, 24 bit, mono, quiet room, the same microphone for all takes, saved as `outputs/visualisation/audio/voice/<video>/<scene>.wav`.
- The storyboard pads each take with silence to its scene length and fails if a take is more than 0.3 s longer than its scene.
- Mix: `ffmpeg.mux_audio` puts the voice over music ducked with `sidechaincompress` and runs two-pass `loudnorm` to integrated -14 LUFS ([[Encoding GIFs and YouTube videos]]).
- Music: one track per video, from the YouTube Audio Library or a CC0 or CC-BY source, recorded in `configs/video/music.yaml` before use.

## YouTube upload checklist

- [ ] `ffmpeg.check_youtube` passes on each file: H.264 High, yuv420p, BT.709, 60 fps, closed GOP 30, 2 B-frames, AAC-LC 48 kHz stereo, moov before mdat, no `elst`.
- [ ] 3-minute 16:9 is 1920 x 1080; the Short is 1080 x 1920 and at most 180 s including cards; the 12-minute video is 1920 x 1080 and under the 15-minute default limit.
- [ ] Title at most 100 characters; description at most 5,000 characters.
- [ ] Description: one-paragraph summary, chapters, links to the Space, the dataset and the report, Craftax credit (Matthews et al. 2024, MIT), music attribution text exactly as its licence requires.
- [ ] Chapters (12-minute video): first at 0:00, at least 3, each at least 10 s, generated from the storyboard.
- [ ] Captions: the storyboard's `captions.srt` uploaded as English subtitles.
- [ ] Thumbnail for each 16:9 video: 1280 x 720 PNG under 2 MB (custom thumbnails need a verified channel).
- [ ] Audience: "No, it's not made for kids"; altered or synthetic content: "No" (rendered game footage, recorded human voice).
- [ ] Category Science & Technology; video language English; licence chosen by the user (Standard YouTube or Creative Commons).
- [ ] Upload as Unlisted; wait for HD processing; check the Copyright step in YouTube Studio shows no claim.
- [ ] End screen on the 16:9 videos linking each other; the Short's "related video" set to the 12-minute video.
- [ ] Switch to Public only after the user confirms in chat; record the three URLs in `configs/video/published.yaml`.

---

### Task 1: Footage set

**Files:**
- Create: `src/wut_murl_craftax/visualisation/footage.py`, `configs/video/footage.yaml`
- Test: `tests/visualisation/test_footage.py`

- [ ] **Step 1: Write the failing tests**: the YAML expands to one rollout spec per (kind, game, policy, checkpoint, seed); each spec's label matches the Footage table; a trained policy loads through the trainer interface of [[2026-10-20-craftax-infrastructure]] (recurrent policies keep their hidden state inside the `Policy` object); god-mode specs always carry the "god mode tour" label.
- [ ] **Step 2: Implement** `load_footage(path) -> list[FootageSpec]` and `make_rollouts(specs, out_dir)` writing `outputs/visualisation/runs/<id>.npz` in the `Rollout` format.
- [ ] **Step 3: Run** `uv run pytest tests/visualisation/test_footage.py -q`.

### Task 2: Masters, highlights and the reviewer CSV

- [ ] **Step 1: Run** `uv run python -m wut_murl_craftax.visualisation.footage configs/video/footage.yaml --out-dir outputs/visualisation/runs`.
- [ ] **Step 2: Run** `uv run python -m wut_murl_craftax.visualisation.render outputs/visualisation/runs/*.npz --out-dir outputs/visualisation/masters`.
- [ ] **Step 3: Run** `uv run python -m wut_murl_craftax.visualisation.highlights outputs/visualisation/runs/*.npz --top 120`; check that achievement, fight and breakthrough candidates of trained agents exist (rank within one batch, [[Highlight detection]]).
- [ ] **Step 4: Run** `uv run python -m wut_murl_craftax.visualisation.review outputs/visualisation/review/candidates.csv`, then serve it with `cd outputs/visualisation/review && python -m http.server 8765 --bind 127.0.0.1`.
- [ ] **Step 5: Hand** the page to the one reviewer (the user); they mark keep or drop, fix captions and speeds, and save `outputs/visualisation/review/review.csv`.
- [ ] **Step 6: Check** that the kept rows cover every storyboard scene that selects clips (counts and seconds per type) before Task 9.

### Task 3: Multi-method headline curve

**Files:**
- Modify: `src/wut_murl_craftax/visualisation/curves.py`
- Test: `tests/visualisation/test_curves.py`

- [ ] **Step 1: Write the failing test**: `render_curves` with 3 runs draws 3 score lines in `METHOD_COLORS` order with a legend, and the second panel shows achievements unlocked at least once per method over env steps; one run still renders the current single-run panel.
- [ ] **Step 2: Implement** a `runs: Sequence[Path]` argument and the multi-run achievements panel; the y axis stays % of max reward (226 Craftax, 22 Classic).
- [ ] **Step 3: Run** `uv run pytest tests/visualisation/test_curves.py -q`.

### Task 4: Stills and two-speed clips

**Files:**
- Create: `src/wut_murl_craftax/visualisation/stills.py`
- Modify: `src/wut_murl_craftax/visualisation/cuts.py` (`cut_clip` takes a list of `(start, end, speed)` segments)
- Test: `tests/visualisation/test_stills.py`, `tests/visualisation/test_cuts.py`

- [ ] **Step 1: Write the failing tests**: a figure PNG becomes a 1080p60 clip of the requested length with 0.4 s fades, letterboxed without distortion, in `land` and `vert`; a two-segment clip (speed 8 then 1) lasts the sum of its segment durations within one frame.
- [ ] **Step 2: Implement** `still_clip(png, seconds, layout, out)` and segment support in `cut_clip`; video-theme figures come from `analysis.figures` with `theme="video"` (vertical variants stack panels).
- [ ] **Step 3: Run** `uv run pytest tests/visualisation/test_stills.py tests/visualisation/test_cuts.py -q`.

### Task 5: Storyboard engine

**Files:**
- Create: `src/wut_murl_craftax/visualisation/storyboard.py`, `configs/video/short.yaml`, `configs/video/long.yaml`
- Test: `tests/visualisation/test_storyboard.py`

- [ ] **Step 1: Write the failing tests**: `short.yaml` sums to 175 s and `long.yaml` to 720 s; any scene list whose `vert` render exceeds 180 s is rejected; a scene with a missing source fails with the scene name; narration over 2.3 words per second fails; chapters start at 0:00, number at least 3 and last at least 10 s; the SRT has increasing, non-overlapping cues inside their scenes; the voice track lasts exactly the video length.
- [ ] **Step 2: Implement** scene kinds `card`, `still`, `clips` (kept rows filtered by `type`, `policy`, `game`, in score order, filling the scene length), `segments` (a case from `cases.csv` cut into `(start, end, speed)` pieces of its 16 px master, because a stall is longer than a 20 s highlight window), `grid`, `curves`, `timeline`, `overlay`, `montage`; Classic clips only in warm-up scenes (`cuts.order_for_cut`).
- [ ] **Step 3: Implement** outputs per video and layout: `out/<video>_<layout>.mp4`, `captions.srt`, `chapters.txt`, `voice.wav`; CLI `uv run python -m wut_murl_craftax.visualisation.storyboard configs/video/<video>.yaml --layout land|vert --csv outputs/visualisation/review/review.csv [--voice-dir ...] [--music ...]`.
- [ ] **Step 4: Run** `uv run pytest tests/visualisation/test_storyboard.py -q`.

### Task 6: Write the storyboards

- [ ] **Step 1: Write** `configs/video/short.yaml` and `configs/video/long.yaml` from the two storyboard tables (scene id, title, seconds, kind, source filter, chapter, narration placeholder).
- [ ] **Step 2: Render** silent drafts of both videos (`--layout land`, and `--layout vert` for the short one) and watch them once; adjust scene sources, not durations.

### Task 7: Voice-over script

- [ ] **Step 1: Write** the narration per scene in the YAML files: what the viewer sees, why it matters, one number at most per scene.
- [ ] **Step 2: Check** every number against `knowledge/results/Final results.md` and every game claim against [[Game]].
- [ ] **Step 3: Read** each scene aloud with a stopwatch; shorten any scene that runs over its seconds; rerun the storyboard tests.
- [ ] **Step 4: Ask** the user to approve both scripts before recording.

### Task 8: Recording

- [ ] **Step 1: Record** one take per scene as listed under Audio; the user records (their voice).
- [ ] **Step 2: Trim** leading and trailing silence; keep breaths short; no music under the takes.
- [ ] **Step 3: Run** the storyboard with `--voice-dir outputs/visualisation/audio/voice/<video>`; it fails on any take that is too long; re-record those.

### Task 9: Music and licence check

**Files:**
- Create: `configs/video/music.yaml`

- [ ] **Step 1: Pick** two candidate tracks per video; prefer YouTube Audio Library tracks marked "no attribution required", else CC0, else CC-BY.
- [ ] **Step 2: Check** for each track: use in YouTube videos allowed; attribution text and where it must appear; redistribution outside YouTube allowed or not (decides whether the Hugging Face release may carry the videos with music, [[2026-10-20-craftax-huggingface-release]]); no Content ID registration.
- [ ] **Step 3: Record** title, artist, source URL, licence, licence URL, attribution text, download date and SHA-256 of the file in `configs/video/music.yaml`; the storyboard refuses a music file whose hash is not listed.
- [ ] **Step 4: Ask** the user to confirm the chosen tracks.

### Task 10: Final renders

- [ ] **Step 1: Run** the kept-window re-render at final size for sharp HUDs: `uv run python -m wut_murl_craftax.visualisation.render --kept outputs/visualisation/review/review.csv --layout land --out-dir outputs/visualisation/hires_land` and the same with `--layout vert`.
- [ ] **Step 2: Run** the storyboard for `short` (`land`, `vert`) and `long` (`land`) with voice and music.
- [ ] **Step 3: Check** each file with `ffmpeg.check_youtube` and the integrated loudness with `ffmpeg.integrated_loudness` (-14 LUFS within 1 LU).
- [ ] **Step 4: Ask** the user to watch all three and list fixes; repeat Steps 2 and 3 until approved.

### Task 11: GIFs

- [ ] **Step 1: Make** these GIFs, each at most 20 s: stuck-then-solved, multi-enemy fight, deepest floor descent, checkpoint grid over the headline curve, first-unlock race (animated chart), Classic warm-up.
- [ ] **Step 2: Run** `uv run python -m wut_murl_craftax.visualisation.cuts gif outputs/visualisation/review/review.csv <id> --out outputs/visualisation/gifs/<name>.gif` for clip GIFs and `uv run python -m wut_murl_craftax.visualisation.montage <run> --updates <4 checkpoints>` for the grid GIF.
- [ ] **Step 3: Check** duration with `ffmpeg.duration` (<= 20 s) and size (<= 5 MB, else x2 scale, [[Encoding GIFs and YouTube videos]]).

### Task 12: YouTube package

- [ ] **Step 1: Write** `outputs/visualisation/youtube/<video>/` with `title.txt`, `description.txt` (chapters from `chapters.txt`), `tags.txt`, `captions.srt`, `thumbnail.png`.
- [ ] **Step 2: Walk** through the YouTube upload checklist with the user; the user uploads (an outward-facing action needing their account and confirmation).
- [ ] **Step 3: Write** the published URLs to `configs/video/published.yaml`.

### Task 13: Knowledge updates

**Files:**
- Create: `knowledge/architecture/visualisation.md` (template `knowledge/_templates/module.md`)
- Modify: `knowledge/visualisation/Visualisation.md` (move music, voice-over, chapters and trained-agent footage from "Not tested" to "Tested" with measured numbers)

- [ ] **Step 1: Write** the measured render times, sizes and loudness of the final videos into [[Visualisation]].
- [ ] **Step 2: Append** a dated section to [[Visualisation decisions]] for any choice that changed (for example two-speed clips, storyboard YAML).
- [ ] **Step 3: Run** `make knowledge` and `make check`.
