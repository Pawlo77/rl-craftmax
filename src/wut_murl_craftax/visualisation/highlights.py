"""Highlight detection: features -> events -> scored windows -> ranked candidates.

``python -m wut_murl_craftax.visualisation.highlights RUNS/*.npz`` writes the candidates
CSV. Frame index == step index (``render``): windows are frame ranges of the masters.
"""

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP

from wut_murl_craftax.visualisation.review import Row, write_rows
from wut_murl_craftax.visualisation.rollout import FEATURES, Floats, Rollout

logger = logging.getLogger(__name__)

CLASSIC_TIER = (
    dict.fromkeys(
        (
            "COLLECT_WOOD",
            "COLLECT_SAPLING",
            "COLLECT_DRINK",
            "PLACE_PLANT",
            "EAT_COW",
            "WAKE_UP",
        ),
        1,
    )
    | dict.fromkeys(
        (
            "PLACE_TABLE",
            "MAKE_WOOD_PICKAXE",
            "MAKE_WOOD_SWORD",
            "DEFEAT_ZOMBIE",
            "EAT_PLANT",
        ),
        2,
    )
    | dict.fromkeys(
        (
            "COLLECT_STONE",
            "PLACE_STONE",
            "MAKE_STONE_PICKAXE",
            "MAKE_STONE_SWORD",
            "PLACE_FURNACE",
            "COLLECT_COAL",
            "DEFEAT_SKELETON",
        ),
        3,
    )
    | dict.fromkeys(("COLLECT_IRON", "MAKE_IRON_PICKAXE", "MAKE_IRON_SWORD"), 5)
    | {"COLLECT_DIAMOND": 8}
)
"""Classic tiers follow tech-tree depth (+1 per unlock in-game); Craftax has 1/3/5/8."""
FLOORS = (
    "Overworld",
    "Dungeon",
    "Gnomish Mines",
    "Sewers",
    "Vault",
    "Troll Mines",
    "Fire Realm",
    "Ice Realm",
    "Graveyard",
)
"""Craftax floor names by level index."""
PRE, POST = 30, 20
"""Pre/post roll around a point event, in steps (10 steps = 1 s of master)."""
STUCK_MIN = 150
"""Steps without a new achievement before an unlock counts as a breakthrough."""
MERGE_GAP, MAX_LEN = 10, 200
"""Merge windows closer than this; never grow a candidate past 20 s."""
OTHERS_WEIGHT = 0.3
"""Share of the other events' scores added to a candidate (early bursts stay modest)."""
DEFAULT_OUT = Path("outputs/visualisation/review/candidates.csv")
"""Default candidates CSV."""


@dataclass(frozen=True, slots=True)
class Event:
    """One detected moment and the clip window around it."""

    start: int
    end: int
    peak: int
    score: float
    kind: str
    label: str


def _rank(e: Event) -> tuple[float, str, str, int]:
    """Score first; ties broken by kind, label, peak (stable across runs)."""
    return e.score, e.kind, e.label, e.peak


@dataclass(slots=True)
class Candidate:
    """Merged events; the best one names the clip and sets its peak frame."""

    start: int
    end: int
    events: list[Event] = field(default_factory=list)

    def ranked(self) -> list[Event]:
        """Events best first."""
        return sorted(self.events, key=_rank, reverse=True)

    def _best(self) -> Event:
        return max(self.events, key=_rank)

    @property
    def score(self) -> float:
        """Best event + ``OTHERS_WEIGHT`` x the others."""
        best, *others = self.ranked()
        return best.score + OTHERS_WEIGHT * sum(e.score for e in others)

    @property
    def kind(self) -> str:
        """Type of the best event."""
        return self._best().kind

    @property
    def caption(self) -> str:
        """Label of the best event."""
        return self._best().label

    @property
    def peak(self) -> int:
        """Peak frame of the best event (thumbnail, freeze-frame)."""
        return self._best().peak


def _tiers(r: Rollout) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    if r.game == "craftax":
        return np.asarray(ACHIEVEMENT_REWARD_MAP, float)
    return np.array([CLASSIC_TIER.get(n, 1) for n in r.achievement_names], float)


def unlock_frequency(runs: Mapping[str, Rollout]) -> dict[str, dict[str, float]]:
    """Per game, the share of episodes unlocking each achievement (rare: up to 2x)."""
    out: dict[str, dict[str, float]] = {}
    for game in {r.game for r in runs.values()}:
        rs = [r for r in runs.values() if r.game == game]
        share = np.mean([r.achievements[-1] for r in rs], axis=0)
        out[game] = dict(zip(rs[0].achievement_names, map(float, share), strict=True))
    return out


def _unlock_events(r: Rollout, freq: dict[str, float]) -> list[Event]:
    tier, names, ev = _tiers(r), r.achievement_names, []
    new = r.achievements[1:] & ~r.achievements[:-1]  # new[t - 1] = unlocked by step t
    prev = 0
    for t in (np.flatnonzero(new.any(1)) + 1).tolist():
        idx = np.flatnonzero(new[t - 1])
        ev += [
            Event(
                t - PRE,
                t + POST,
                t,
                float(tier[i]) * (2 - freq.get(names[i], 0)),
                "achievement",
                names[i],
            )
            for i in idx
        ]
        if prev and t - prev >= STUCK_MIN:
            ev.append(
                Event(
                    t - 60,
                    t + POST,
                    t,
                    2 + (t - prev) / 100,
                    "breakthrough",
                    f"{names[idx[0]]} after {t - prev} steps stuck",
                )
            )
        prev = t
    return ev


def _fights(
    hostile: Floats, enemies: Floats, close: Floats, dmg: Floats
) -> list[Event]:
    busy = (hostile >= 2) & (
        close >= 1
    )  # several on screen, at least one within 2 tiles
    ev, t, last = [], 0, len(busy) - 1
    while t <= last:
        if not busy[t]:
            t += 1
            continue
        s = e = t
        while e < last and busy[e + 1 : e + 5].any():  # bridge gaps under 5 steps
            e += 1
        n, d = int(enemies[s : e + 1].max()), float(dmg[s : e + 1].sum())
        if d > 0 or e - s >= 5:
            peak = s + int(np.argmax(hostile[s : e + 1]))
            ev.append(
                Event(
                    s - 10,
                    e + 10,
                    peak,
                    1.5 * n + d,
                    "fight",
                    f"{n} enemies at once, {d:.0f} damage taken",
                )
            )
        t = e + 1
    return ev


def detect(r: Rollout, freq: Mapping[str, Mapping[str, float]]) -> list[Event]:
    """All events of one rollout, windows clamped to its frames."""
    f = {n: r.features[:, i] for i, n in enumerate(FEATURES)}
    t_end = r.steps
    ev = _unlock_events(r, dict(freq.get(r.game, {})))
    lvl = f["level"]
    for t in (np.flatnonzero(np.diff(lvl)) + 1).tolist():
        way = "down" if lvl[t] > lvl[t - 1] else "up"
        ev.append(
            Event(
                t - PRE,
                t + 40,
                t,
                6 + float(lvl[t]),
                "floor",
                f"{way} to {FLOORS[int(lvl[t])]}",
            )
        )
    if r.game == "craftax":
        boss = f["boss_progress"]
        ev += [
            Event(t - PRE, t + POST, t, 10, "boss", f"boss hit {int(boss[t])}")
            for t in (np.flatnonzero(np.diff(boss) > 0) + 1).tolist()
        ]
        if (arena := np.flatnonzero(lvl == 8)).size:
            ev.append(
                Event(
                    int(arena[0]),
                    int(arena[0]) + 80,
                    int(arena[0]) + 10,
                    6,
                    "boss",
                    "boss arena",
                )
            )
    hp = f["health"]
    dmg = np.r_[0, np.clip(hp[:-1] - hp[1:], 0, None)]
    enemies = f["melee_on_screen"] + f["ranged_on_screen"]
    ev += _fights(
        enemies + f["projectiles_on_screen"], enemies, f["hostile_close"], dmg
    )
    if hp[-1] <= 0:
        ev.append(
            Event(t_end - 50, t_end + 1, t_end, 5, "death", f"death at step {t_end}")
        )
    if (low := np.flatnonzero(hp <= 2)).size and low[
        0
    ] < t_end - 50:  # near death, then 50+ more steps
        t = int(low[0])
        ev.append(
            Event(t - PRE, t + 50, t, 4, "close call", f"survived at {hp[t]:.1f} HP")
        )
    return [
        Event(max(0, e.start), min(t_end, e.end), e.peak, e.score, e.kind, e.label)
        for e in ev
    ]


def merge(events: Sequence[Event]) -> list[Candidate]:
    """Join windows starting within ``MERGE_GAP`` of the open one, up to ``MAX_LEN``."""
    out: list[Candidate] = []
    for e in sorted(events, key=lambda e: (e.start, e.end, e.peak, *_rank(e))):
        c = out[-1] if out else None
        if (
            c
            and e.start <= c.end + MERGE_GAP
            and max(e.end, c.end) - c.start <= MAX_LEN
        ):
            c.end = max(c.end, e.end)
            c.events.append(e)
        else:
            out.append(Candidate(e.start, e.end, [e]))
    return out


def candidates(runs: Mapping[str, Rollout], top: int = 60, fps: int = 10) -> list[Row]:
    """Ranked review rows (``review.COLUMNS``): the ``top`` candidates of all runs."""
    freq = unlock_frequency(runs)
    scored = [(c, stem, r) for stem, r in runs.items() for c in merge(detect(r, freq))]
    scored.sort(key=lambda x: -x[0].score)
    rows = []
    for i, (c, stem, r) in enumerate(scored[:top]):
        events = "; ".join(f"{e.kind}:{e.label}" for e in c.ranked())
        rows.append({
            "id": f"c{i + 1:03d}",
            "keep": "",
            "score": f"{c.score:.2f}",
            "type": c.kind,
            "caption": c.caption,
            "rollout": stem,
            "game": r.game,
            "policy": r.policy,
            "label": r.label,
            "start": str(c.start),
            "end": str(c.end),
            "peak": str(c.peak),
            "seconds": f"{(c.end - c.start + 1) / fps:.1f}",
            "speed": "1",
            "events": events,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    """Detect, merge and rank highlights of the given rollouts into a candidates CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rollouts", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top", type=int, default=60)
    a = p.parse_args(argv)
    runs = {path.stem: Rollout.load(path) for path in a.rollouts}
    rows = candidates(runs, a.top)
    write_rows(rows, a.out)
    types: dict[str, int] = {}
    for r in rows:
        types[r["type"]] = types.get(r["type"], 0) + 1
    print(f"{len(runs)} rollouts -> top {len(rows)} candidates {types} in {a.out}")


if __name__ == "__main__":
    main()
