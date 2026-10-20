"""Tests for event detection, scoring, merging and ranking on synthetic rollouts."""

import numpy as np

from wut_murl_craftax.visualisation.highlights import (
    Event,
    candidates,
    detect,
    merge,
    unlock_frequency,
)
from wut_murl_craftax.visualisation.rollout import FEATURES, Rollout

NAMES = ("COLLECT_WOOD", "PLACE_TABLE", "COLLECT_DIAMOND")


def _rollout(steps: int = 400, game: str = "classic") -> Rollout:
    feats = np.zeros((steps + 1, len(FEATURES)), np.float32)
    feats[:, FEATURES.index("health")] = 9
    ach = np.zeros((steps + 1, len(NAMES)), bool)
    return Rollout(
        game=game,
        policy="heuristic",
        label="scripted agent",
        seed=0,
        key_scheme="fold_in",
        god=False,
        actions=np.zeros(steps, np.int16),
        rewards=np.zeros(steps, np.float32),
        features=feats,
        feature_names=FEATURES,
        achievements=ach,
        achievement_names=NAMES,
    )


def _col(r: Rollout, name: str) -> np.ndarray:
    return r.features[:, FEATURES.index(name)]


def test_unlocks_breakthrough_floor_fight_and_death() -> None:
    r = _rollout()
    r.achievements[50:, 0] = True
    r.achievements[300:, 2] = True  # 250 steps after the previous unlock
    _col(r, "level")[200:] = 1
    _col(r, "melee_on_screen")[100:110] = 3
    _col(r, "hostile_close")[100:110] = 1
    _col(r, "health")[105:] = 6
    _col(r, "health")[-1] = 0
    evs = detect(r, unlock_frequency({"r": r}))
    by_kind = {e.kind: e for e in evs}
    wood = next(e for e in evs if e.label == "COLLECT_WOOD")
    assert (wood.start, wood.end, wood.peak) == (20, 70, 50)
    assert by_kind["breakthrough"].label == "COLLECT_DIAMOND after 250 steps stuck"
    assert by_kind["floor"].label == "down to Dungeon"
    assert by_kind["fight"].label == "3 enemies at once, 3 damage taken"
    assert (by_kind["fight"].start, by_kind["fight"].end) == (90, 119)
    assert by_kind["death"].end == 400
    rare = next(e for e in evs if e.label == "COLLECT_DIAMOND")
    assert rare.score == 8.0  # tier 8 x (2 - unlocked in every episode)


def test_merge_joins_close_windows_and_caps_length() -> None:
    evs = [
        Event(0, 50, 30, 3.0, "achievement", "A"),
        Event(55, 90, 70, 5.0, "fight", "B"),
        Event(95, 250, 120, 1.0, "fight", "C"),
    ]
    out = merge(evs)
    assert [(c.start, c.end) for c in out] == [(0, 90), (95, 250)]
    assert out[0].score == 5.0 + 0.3 * 3.0
    assert (out[0].kind, out[0].caption, out[0].peak) == ("fight", "B", 70)


def test_candidates_rank_and_fill_review_columns() -> None:
    a, b = _rollout(), _rollout()
    a.achievements[50:, 2] = True
    b.achievements[60:, 0] = True
    rows = candidates({"a": a, "b": b}, top=10)
    assert [r["id"] for r in rows] == ["c001", "c002"]
    assert rows[0]["caption"] == "COLLECT_DIAMOND"
    assert float(rows[0]["score"]) > float(rows[1]["score"])
    assert rows[0]["keep"] == ""
    assert rows[0]["speed"] == "1"
    assert rows[0]["label"] == "scripted agent"
