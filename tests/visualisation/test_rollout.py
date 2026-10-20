"""Tests for the rollout record, policies and deterministic replay."""

from pathlib import Path

import jax
import numpy as np
import pytest
from _gameplay_run import classic_rollout

from wut_murl_craftax.visualisation.rollout import (
    FEATURES,
    HeuristicPolicy,
    NetworkPolicy,
    RandomPolicy,
    Rollout,
    game_for,
    keys,
    play,
    replay,
)


def test_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "r.npz"
    classic_rollout().save(path)
    back = Rollout.load(path)
    assert back == classic_rollout()
    assert back.steps == len(back.actions) == len(back.features) - 1
    assert back.feature_names == FEATURES


def test_replay_reproduces_recorded_features() -> None:
    chunks = list(replay(classic_rollout(), chunk=16))
    feats = np.concatenate([c.features for c in chunks])
    assert np.array_equal(feats, classic_rollout().features)


def test_replay_frames_are_deterministic_and_ranged() -> None:
    full = np.concatenate([
        c.frames for c in replay(classic_rollout(), bps=16, chunk=16)
    ])
    again = np.concatenate([
        c.frames for c in replay(classic_rollout(), bps=16, chunk=32)
    ])
    part = np.concatenate([
        c.frames for c in replay(classic_rollout(), bps=16, start=20, end=40, chunk=8)
    ])
    assert full.shape == (classic_rollout().steps + 1, 144, 144, 3)
    assert np.array_equal(full, again)
    assert np.array_equal(part, full[20:41])


def test_key_schemes_share_step_keys_but_not_the_reset_key() -> None:
    base = jax.random.PRNGKey(3)
    reset, env = keys(3)
    assert np.array_equal(reset, base)
    assert np.array_equal(env, jax.random.fold_in(base, 1))
    split_reset, split_env = keys(3, "split")
    assert np.array_equal(split_env, env)  # threefry: split(k)[1] == fold_in(k, 1)
    assert not np.array_equal(split_reset, reset)  # so only the world differs


def test_heuristic_policy_unlocks_achievements() -> None:
    unlocked = classic_rollout().achievements[-1]
    names = [
        n
        for n, x in zip(classic_rollout().achievement_names, unlocked, strict=True)
        if x
    ]
    assert "COLLECT_WOOD" in names


@pytest.mark.parametrize("policy_cls", [RandomPolicy, NetworkPolicy])
def test_other_policies_return_valid_actions(policy_cls: type) -> None:
    game = game_for("classic")
    r = play(game, policy_cls(game), seed=0, max_steps=10)
    assert r.actions.min() >= 0
    assert r.actions.max() < game.num_actions
    assert r.label  # shown on screen in the cuts


def test_craftax_features_track_floor_and_achievement_order() -> None:
    game = game_for("craftax", god_mode=True)
    names = game.achievement_names
    assert names[25] == "MAKE_DIAMOND_SWORD"  # enum order differs from value order
    r = play(game, HeuristicPolicy(game), seed=1, max_steps=60)
    level = r.features[:, FEATURES.index("level")]
    assert level.max() >= 1  # the god-mode dive descends at step 59
    assert r.label == "god-mode dive"
