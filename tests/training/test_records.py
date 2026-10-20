"""Tests for the per-update metrics and per-episode logs."""

from pathlib import Path

import numpy as np

from wut_murl_craftax.training.records import (
    RunLog,
    achievement_names,
    achievement_weights,
    first_unlocks,
    load_episodes,
    load_metrics,
)

CLASSIC = "Craftax-Classic-Symbolic-v1"


def test_achievement_tables() -> None:
    assert len(achievement_names(CLASSIC)) == 22
    assert achievement_weights(CLASSIC).sum() == 22
    assert len(achievement_names("Craftax-Symbolic-v1")) == 67
    assert achievement_weights("Craftax-Symbolic-v1").sum() == 226


def _log_two_updates(out: Path) -> RunLog:
    names = achievement_names(CLASSIC)
    log = RunLog(out, CLASSIC, names)
    done = np.zeros((4, 2), bool)
    done[1, 0] = done[3, 1] = True
    ach = np.zeros((4, 2, len(names)), bool)
    ach[1, 0, 0] = True  # collect_wood in the first finished episode
    ret = np.arange(8, dtype=np.float32).reshape(4, 2)
    length = np.full((4, 2), 10, np.int32)
    losses = np.zeros(5, np.float32)
    log.add_update(1, 8, 0.5, losses, done, ach, ret, length)
    ach2 = ach.copy()
    ach2[3, 1, 2] = True
    log.add_update(2, 16, 1.0, losses, done, ach2, ret, length)
    log.flush()
    return log


def test_metrics_rates_and_score(tmp_path: Path) -> None:
    _log_two_updates(tmp_path)
    m = load_metrics(tmp_path)
    assert m.steps.tolist() == [8, 16]
    assert m.rates[0, 0] == 50.0  # 1 of 2 finished episodes
    np.testing.assert_allclose(m.col("return"), [(2 + 7) / 2] * 2)
    np.testing.assert_allclose(m.score, m.col("return") / 22 * 100)


def test_episode_bits_round_trip_and_first_unlock(tmp_path: Path) -> None:
    _log_two_updates(tmp_path)
    ep = load_episodes(tmp_path)
    assert ep.update.tolist() == [1, 1, 2, 2]
    assert ep.ach.shape == (4, 22)
    assert ep.ach[:, 0].tolist() == [True, False, True, False]
    first = first_unlocks(ep)
    assert first[0] == 1
    assert first[2] == 2
    assert first[1] == -1
