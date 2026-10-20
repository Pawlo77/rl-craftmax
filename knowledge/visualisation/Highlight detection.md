---
title: "Highlight detection"
type: handbook
domain: visualisation
parent: "[[Visualisation]]"
tags: [handbook, visualisation, highlights, events]
created: 2026-10-20
updated: 2026-10-20
---
# Highlight detection

Per-step features recorded by the rollout (see [[Rendering Craftax frames]]) are turned into events, each event into a clip window, overlapping windows into one candidate, and candidates are ranked by score into `candidates.csv` for [[Highlight review workflow]]. Frame index == step index, so windows are frame ranges of the 16 px master.

## Steps

1. Run `detect.py review/candidates.csv runs/*.npz --top 60` (pure numpy; 1.9 s for 23 rollouts).

2. Event rules (`t` = step whose state shows the event; windows in steps, 10 steps = 1 s of master):

   | Event | Trigger from features | Window | Score |
   |---|---|---|---|
   | achievement | `ach[t] & ~ach[t-1]` | t-30 .. t+20 | tier x (2 - share of episodes of that game unlocking it) |
   | breakthrough | an unlock 150+ steps after the previous unlock | t-60 .. t+20 | 2 + gap / 100 |
   | floor | `level[t] != level[t-1]` | t-30 .. t+40 | 6 + new floor index |
   | boss | first step on floor 8; every rise of `boss_progress` | t .. t+80; t-30 .. t+20 | 6; 10 |
   | fight | run of steps with 2+ hostiles (melee + ranged + projectiles) on screen and 1+ within 2 tiles, gaps under 5 steps bridged, kept if damage taken or 6+ steps | start-10 .. end+10 | 1.5 x max enemies on screen + health lost |
   | death | `health <= 0` at the last step | T-50 .. T | 5 |
   | close call | health <= 2 and the episode lasts 50+ more steps | t-30 .. t+50 | 4 |

   Tiers: Craftax uses its own `ACHIEVEMENT_REWARD_MAP` (1, 3, 5, 8: 25, 18, 15 and 9 achievements); Classic (all +1 in the game) uses tech-tree depth: 1 basic (wood, sapling, drink, plant, cow, wake up), 2 table / wood tools / zombie / eat plant, 3 stone, furnace, coal, skeleton, 5 iron and iron tools, 8 diamond. Floor names and order: [[Craftax floors and progression]]; achievements: [[Craftax achievements]].

3. Core of `detect.py`:

   ```python
   new = A[1:] & ~A[:-1]                               # A = achievements [T+1, n]; new[t-1] = unlocked at step t
   unlock_steps = [t for t in range(1, T + 1) if new[t - 1].any()]
   prev = 0
   for t in unlock_steps:
       for i in np.flatnonzero(new[t - 1]):
           ev.append((t - 30, t + 20, t, tier[i] * (2 - freq[kind].get(names[i], 0)), "achievement", names[i]))
       if t - prev >= 150 and prev > 0:
           ev.append((t - 60, t + 20, t, 2 + (t - prev) / 100, "breakthrough", f"{names[...]} after {t - prev} steps stuck"))
       prev = t
   for t in np.flatnonzero(np.diff(level)) + 1:
       ev.append((t - 30, t + 40, t, 6 + level[t], "floor", f"down to {FLOORS[level[t]]}"))
   dmg = np.r_[0, np.clip(hp[:-1] - hp[1:], 0, None)]
   busy = (melee + ranged + projectiles >= 2) & (hostile_close >= 1)   # fights: maximal runs of busy
   ```

4. Merge and rank: sort events by window start; extend the open candidate while the next window starts within 10 steps of its end and the result stays at most 200 steps (20 s); score = best event + 0.3 x sum of the others; the best event gives the clip its type, caption and peak frame (thumbnail and freeze-frame).

   ```python
   for s, e, p, sc, k, lab in sorted(evs):
       c = out[-1] if out else None
       if c and s <= c["end"] + 10 and max(e, c["end"]) - c["start"] <= 200:
           c["end"] = max(c["end"], e); c["events"].append((sc, k, lab, p))
       else:
           out.append(dict(start=s, end=e, events=[(sc, k, lab, p)]))
   ```

5. Output columns: `id, keep, score, type, caption, rollout, game, policy, start, end, peak, seconds, speed, events`; `keep` empty and `speed` 1 for the reviewer.

## Results

| Measurement | Result |
|---|---|
| Input | 23 test rollouts (heuristic, random, one god-mode dive), 14,986 steps |
| Events | 473 |
| Merged candidates | 81: achievement 31, fight 32, death 8, floor 5, breakthrough 5 (by best event) |
| Top 60 written | achievement 21, fight 25, death 6, floor 5, breakthrough 3; 703.2 s of footage |
| Highest ranked | god-mode floor dives (scores 35.9, 33.6, 26.3, 25.2), then Classic fights with 4 enemies and 4 to 8 damage taken (24.9, 24.1) |
| Longest drought caught | Classic `DEFEAT_SKELETON` after 655 steps without a new achievement |
| Boss hits (`boss_progress` rising) | none in the test data; only the "boss arena" entry event fired |

## Pitfalls

- Early-game unlocks come in bursts (10+ in the first 200 steps); summing all event scores ranks those bursts above single rare events, hence 0.3 x the rest and the 20 s cap.
- Projectiles count towards "busy" but not towards the enemy number in the caption; an arrow in flight alone with one zombie still makes a fight.
- Rarity weights are per input set: the same achievement scores differently in a batch of random episodes and in a batch of trained episodes; rank within one batch.
- Craftax has at most 3 melee and 2 ranged mobs per floor (`StaticEnvParams`), so "enemies at once" tops out at 5 plus projectiles.
- God-mode episodes have full inventories and walk through walls; label or drop them in public cuts.

## Not tested

- Boss-fight events (`boss_progress`, `DAMAGE_NECROMANCER`, `DEFEAT_NECROMANCER`): the code path ran, no episode hit the boss.
- Weighting by value-function surprise or advantage of a trained agent (needs a network; see [[Agent internals overlay]]).
- Variable speed inside one clip (time-lapse of the stuck stretch, then 1x for the unlock); today speed is one value per clip.
