"""Tests for fixed-seed eval rollouts: files, determinism and replay."""

import jax
import numpy as np
from _learning_run import EVAL_SEEDS, EVAL_T, tiny_run

from wut_murl_craftax.visualisation.eval_rollouts import (
    EvalRun,
    as_rollout,
    episode_frames,
    load_eval,
    make_eval_rollout,
    state_digest,
)
from wut_murl_craftax.visualisation.rollout import replay


def test_eval_files_cover_every_checkpoint() -> None:
    run = EvalRun.load(tiny_run())
    for update in run.updates:
        ev = load_eval(run.dir, update)
        assert ev["act"].shape == (len(EVAL_SEEDS), EVAL_T)
        assert ev["probs"].shape == (len(EVAL_SEEDS), EVAL_T, 17)
        assert ev["obs"].shape == (len(EVAL_SEEDS), EVAL_T, 1345)
        np.testing.assert_allclose(ev["probs"].sum(-1), 1, atol=1e-2)


def test_same_seed_same_actions_and_vmap_matches_single() -> None:
    run = EvalRun.load(tiny_run())
    params = run.params(run.updates[-1])
    roll = make_eval_rollout(run, 30)
    a, b = roll(params, 0), roll(params, 0)
    assert np.array_equal(a["act"], b["act"])
    batch = jax.vmap(roll, in_axes=(None, 0))(params, np.arange(4))
    assert np.array_equal(a["act"], batch["act"][0])
    assert not np.array_equal(a["act"], batch["act"][1])


def test_replay_of_stored_actions_reproduces_states() -> None:
    run = EvalRun.load(tiny_run())
    update = run.updates[-1]
    ev = load_eval(run.dir, update)
    forced = make_eval_rollout(run, EVAL_T, forced=True)
    other_params = run.params(0)  # states depend on the actions only
    tr = forced(other_params, 0, ev["act"][0])
    assert np.array_equal(state_digest(tr["state"]), ev["digest"][0])


def test_episode_frames_are_deterministic() -> None:
    run = EvalRun.load(tiny_run())
    update = run.updates[-1]
    a = episode_frames(run, update, seed=1)
    b = episode_frames(run, update, seed=1)
    n_alive = int(load_eval(run.dir, update)["alive"][1].sum())
    assert a.shape == (min(n_alive + 1, EVAL_T), 144, 144, 3)
    assert np.array_equal(a, b)


def test_as_rollout_replays_to_the_same_frames() -> None:
    run = EvalRun.load(tiny_run())
    update = run.updates[-1]
    r = as_rollout(run, update, seed=0)
    frames = np.concatenate([
        c.frames for c in replay(r, bps=16) if c.frames is not None
    ])
    assert np.array_equal(frames, episode_frames(run, update, seed=0))
    assert r.features.shape[0] == r.steps + 1
