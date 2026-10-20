---
title: "Visualisation decisions"
type: decision-log
domain: visualisation
parent: "[[wut_murl_craftax]]"
tags: [decision-log, visualisation]
created: 2026-10-20
updated: 2026-10-20
---
# Visualisation decisions

## 2026-10-20 - Video formats, footage, audio and review

Decision: the 3-minute video is made twice, as a 16:9 video and as a vertical YouTube Short of at most 180 s; the 12-minute video is 16:9; GIFs are at most 20 s.

Why: the user wants both a channel video and a Short from the same footage.

Consequences: the cut step renders both aspect ratios ([[Assembling the 3 and 12 minute cuts]]).

Decision: upload at 1080p by default; 2160p stays an option.

Why: the user chose 1080p.

Consequences: final encodes default to 1920 x 1080 (or 1080 x 1920 for the Short) ([[Encoding GIFs and YouTube videos]]). Rejected: 2160p by default (higher YouTube bitrate, but larger and slower to encode).

Decision: besides the trained agent, the cuts may show early checkpoints, a random policy, a scripted agent and a god-mode dive, each labelled on screen.

Why: the user wants the contrast of "before learning" footage and a full game tour for the intro.

Consequences: every policy kind plugs into the same rollout and replay format ([[Rendering Craftax frames]]).

Decision: audio is a voice-over plus music, the music ducked under the voice and loudness normalised for YouTube.

Why: the user chose narration with music.

Consequences: the cut step mixes a voice track and a music track supplied by the team; a silent track is used when none is given. Rejected: captions only, silent video.

Decision: one reviewer marks keep or drop per highlight; a single CSV drives the cut.

Why: the user chose a single reviewer.

Consequences: [[Highlight review workflow]] reads one CSV; no vote merging.

Decision: one video per length covers both games, with Craftax as the main subject and Craftax-Classic as a short warm-up.

Why: Craftax is the project's target environment.

Consequences: rollouts, training curves and grids support both games. Rejected: separate videos per game, Craftax only.

Decision: the headline training curve shows reward as % of max (the scoreboard metric) with achievements unlocked as a second panel; checkpoint grid panels are labelled with the fixed-seed evaluation score.

Why: % of max reward is comparable with [[Existing Implementations]]; achievements are easier to explain on screen; fixed seeds make grid panels comparable.

Consequences: [[Animated training curves]] and [[Checkpoint comparison grid]].

## 2026-10-20 - Visualisation code lives in the package

Decision: every visualisation step that is used regularly is code in `src/wut_murl_craftax/visualisation/` (with the baseline PPO used to produce test material in `src/wut_murl_craftax/training/`), tested in `tests/`; notes link to the modules instead of carrying scripts.

Why: the user does not want ad-hoc scripts generated on the fly.

Consequences: charts are rendered as images and video; the only HTML is one small static review page template. Rejected: HTML dashboards and per-clip HTML files.
