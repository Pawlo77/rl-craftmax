---
title: "Highlight detection"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, highlights, events]
created: 2026-10-20
updated: 2026-10-20
related-code:
  - "[[src/wut_murl_craftax/visualisation/highlights.py]]"
  - "[[tests/visualisation/test_highlights.py]]"
---
# Highlight detection

Per-step features recorded with each rollout ([[Rendering Craftax frames]]) become events, each event a clip window, overlapping windows one candidate; candidates are ranked by score into `candidates.csv` for [[Highlight review workflow]]. Frame index == step index, so windows are frame ranges of the 16 px masters.

## Steps

1. Detect and rank (pure numpy, no env):

   ```bash
   uv run python -m wut_murl_craftax.visualisation.highlights outputs/visualisation/runs/*.npz --top 60
   ```

   Output: `outputs/visualisation/review/candidates.csv` (columns in [[Highlight review workflow]]).

2. Event rules (`detect`; `t` = step whose state shows the event; windows in steps, 10 steps = 1 s of master):

   | Event | Trigger from features | Window | Score |
   |---|---|---|---|
   | achievement | `ach[t] & ~ach[t-1]` | t-30 .. t+20 | tier x (2 - share of episodes of that game unlocking it) |
   | breakthrough | an unlock 150+ steps after the previous unlock | t-60 .. t+20 | 2 + gap / 100 |
   | floor | `level[t] != level[t-1]` | t-30 .. t+40 | 6 + new floor index |
   | boss | first step on floor 8; every rise of `boss_progress` | t .. t+80; t-30 .. t+20 | 6; 10 |
   | fight | run of steps with 2+ hostiles (melee + ranged + projectiles) on screen and 1+ within 2 tiles, gaps under 5 steps bridged, kept if damage taken or 6+ steps | start-10 .. end+10 | 1.5 x max enemies on screen + health lost |
   | death | `health <= 0` at the last step | T-50 .. T | 5 |
   | close call | health <= 2 and the episode lasts 50+ more steps | t-30 .. t+50 | 4 |

   Tiers: Craftax uses its own `ACHIEVEMENT_REWARD_MAP` (1, 3, 5, 8: 25, 18, 15 and 9 achievements); Classic (all +1 in the game) uses tech-tree depth (`CLASSIC_TIER`): 1 basic (wood, sapling, drink, plant, cow, wake up), 2 table / wood tools / zombie / eat plant, 3 stone, furnace, coal, skeleton, 5 iron and iron tools, 8 diamond. Floors: [[Craftax floors and progression]]; achievements: [[Craftax achievements]].

3. Merge (`merge`): events sorted by window start; the open candidate grows while the next window starts within 10 steps of its end (`MERGE_GAP`) and stays at most 200 steps (`MAX_LEN`, 20 s); score = best event + 0.3 x the others (`OTHERS_WEIGHT`); the best event (ties: kind, label, peak) gives type, caption and peak frame (thumbnail and freeze-frame).

4. Rank (`candidates`): all candidates of all rollouts by score; rows get `id` c001..., empty `keep`, `speed` 1 and the footage `label` of the rollout.

## Results

| Measurement | Result |
|---|---|
| Input | 23 test rollouts (heuristic, random, one god-mode dive), 14,986 steps |
| Events | 473 |
| Merged candidates | 81: achievement 31, fight 32, death 8, floor 5, breakthrough 5 (by best event) |
| Top 60 written | achievement 21, fight 25, death 6, floor 5, breakthrough 3; 703.2 s of footage |
| Time | 1.9 s |
| Highest ranked | god-mode floor dives (scores 35.9, 33.6, 26.3, 25.2), then Classic fights with 4 enemies and 4 to 8 damage taken (24.9, 24.1) |
| Longest drought caught | Classic `DEFEAT_SKELETON` after 655 steps without a new achievement |
| Boss hits (`boss_progress` rising) | none in the test data; only the "boss arena" entry event fired |
| Package vs first script on the 23 rollouts | the 60 rows identical in id, rollout, window, peak, type, caption, events, seconds; one displayed score differs (11.93 vs 11.92: Python vs numpy rounding of 11.925) |
| Repo smoke test (7 new rollouts) | top 30: achievement 7, fight 14, breakthrough 4, floor 2, death 2, close call 1 |

## Pitfalls

- Early-game unlocks come in bursts (10+ in the first 200 steps); summing all event scores ranks those bursts above single rare events, hence 0.3 x the rest and the 20 s cap.
- Projectiles count towards "busy" but not towards the enemy number in the caption; an arrow in flight alone with one zombie still makes a fight.
- Rarity weights are per input set: the same achievement scores differently in a batch of random episodes and in a batch of trained episodes; rank within one batch.
- Craftax has at most 3 melee and 2 ranged mobs per floor (`StaticEnvParams`), so "enemies at once" tops out at 5 plus projectiles.
- God-mode episodes have full inventories and walk through walls; their label (`god-mode dive`) is shown on screen.

## Not tested

- Boss-fight events (`boss_progress`, `DAMAGE_NECROMANCER`, `DEFEAT_NECROMANCER`): the code path ran, no episode hit the boss.
- Weighting by value-function surprise or advantage of a trained agent (needs a network; see [[Agent internals overlay]]).
- Variable speed inside one clip (time-lapse of the stuck stretch, then 1x for the unlock); speed is one value per clip.
