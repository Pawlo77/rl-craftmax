"""Tests for fixed-seed evaluation: chunked rollouts, eval files, replay, schedule."""

import inspect
import json
import shutil
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _learning_run import EVAL_SEEDS, TINY, runner_config, tiny_run
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.events import StepEvents
from wut_murl_craftax.orchestration.queue import RunSpec
from wut_murl_craftax.training.checkpoints import save_checkpoint
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.ppo import PPOPolicy
from wut_murl_craftax.visualisation.eval_rollouts import (
    EVAL_SEEDS as PRODUCTION_SEEDS,
)
from wut_murl_craftax.visualisation.eval_rollouts import (
    MODES,
    EvalRun,
    as_rollout,
    episode_frames,
    eval_returns,
    evaluate,
    evaluate_spec,
    load_eval,
    replay_episode,
    rollout_seeds,
)
from wut_murl_craftax.visualisation.rollout import replay

ENV = TINY.env
EVENT_KEYS = {f"events/{f.name}" for f in fields(StepEvents)}
FILE_KEYS = {"seeds", "actions", "length", "method"} | EVENT_KEYS
MOVES = 5
"""The counting policy cycles NOOP and the four moves: it never eats, so it dies."""


class _CountingPolicy:
    """A recurrent stand-in: the carry counts steps, the action is the count mod 5."""

    def init_carry(self, n: int) -> dict[str, jax.Array]:
        return {"t": jnp.zeros(n, jnp.int32)}

    def step(
        self,
        params: Mapping[str, jax.Array],
        carry: Mapping[str, jax.Array],
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del obs, key, greedy
        return (carry["t"] + params["shift"]) % MOVES, {"t": carry["t"] + 1}

    def extras(self, carry: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """A two-level policy's running skill and skill step, read from the carry."""
        return {"skill": carry["t"] // 4, "skill_step": carry["t"] % 4}


class _CountingTrainer:
    """The registry entry of `_CountingPolicy`; its params are one shift."""

    def init(self, config: Mapping[str, Any], key: jax.Array) -> dict[str, jax.Array]:
        del config, key
        return {"shift": jnp.zeros((), jnp.int32)}

    def params(self, state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return state

    def policy(self, state: dict[str, jax.Array]) -> _CountingPolicy:
        return _CountingPolicy()


@pytest.fixture
def counting_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A run of the counting entry, registered in `METHODS`, checkpoints 0 to 2."""
    monkeypatch.setitem(METHODS, "counting", _CountingTrainer())
    run = tmp_path / "counting"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"method": "counting", "env": ENV}))
    for u in range(3):
        save_checkpoint(run, u, 100 * u, 0.0, {"shift": np.int32(u)})
    return run


@pytest.fixture
def fresh_ppo_run(tmp_path: Path) -> Path:
    """The tiny PPO run's config and checkpoints without its evaluations."""
    run = tmp_path / "ppo"
    shutil.copytree(tiny_run() / "ckpt", run / "ckpt")
    (run / "config.json").write_text(json.dumps(runner_config(TINY)))
    return run


def _spec(phase: str, eval_mode: str) -> RunSpec:
    cfg = runner_config(TINY)
    return RunSpec("ppo", "ppo", "1m", ENV, 0, cfg, "0" * 8, phase, eval_mode)


def test_production_seeds_are_10000_to_10063() -> None:
    assert tuple(range(10000, 10064)) == PRODUCTION_SEEDS
    assert MODES == ("stochastic", "greedy")


def test_the_same_checkpoint_twice_gives_the_same_rollouts() -> None:
    run = EvalRun.load(tiny_run())
    params = run.params(run.updates[-1])
    for greedy in (False, True):
        a = rollout_seeds(run, params, EVAL_SEEDS, greedy)
        b = rollout_seeds(run, params, EVAL_SEEDS, greedy)
        assert a is not None
        assert b is not None
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
    # a seed alone plays as inside the batch, so fewer seeds are a subset
    alone = rollout_seeds(run, params, EVAL_SEEDS[1:], True)
    assert alone is not None
    np.testing.assert_array_equal(alone["actions"][0], a["actions"][1])


def test_load_opens_any_registry_entry_and_threads_its_carry(
    counting_run: Path,
) -> None:
    run = EvalRun.load(counting_run)
    assert run.method == "counting"
    out = rollout_seeds(run, run.params(2), EVAL_SEEDS, greedy=False)
    assert out is not None
    for i, n in enumerate(out["length"]):
        t = np.arange(n)
        np.testing.assert_array_equal(out["actions"][i, :n], (t + 2) % MOVES)
        np.testing.assert_array_equal(out["extras/skill"][i, :n], (t + 1) // 4)
        np.testing.assert_array_equal(out["extras/skill_step"][i, :n], (t + 1) % 4)
    ppo = EvalRun.load(tiny_run())
    assert ppo.method == "ppo"
    assert isinstance(ppo.policy, PPOPolicy)
    assert str(load_eval(ppo.dir, 0)["method"]) == "ppo"


def test_the_scan_stops_after_the_chunk_where_the_last_episode_ended() -> None:
    run = EvalRun.load(tiny_run())
    out = rollout_seeds(run, run.params(4), EVAL_SEEDS, greedy=False, chunk=16)
    assert out is not None
    steps = out["actions"].shape[1]
    assert steps % 16 == 0
    assert steps - 16 < out["length"].max() <= steps
    for i, n in enumerate(out["length"]):
        assert (out["actions"][i, :n] >= 0).all()
        assert (out["actions"][i, n:] == -1).all()
        ep = replay_episode(run, EVAL_SEEDS[i], out["actions"][i, :n])
        assert ep.done[-1]  # the episode ended there: never cut
        assert not ep.done[:-1].any()


def test_the_eval_file_holds_actions_events_and_declared_extras_only(
    counting_run: Path,
) -> None:
    ev = load_eval(tiny_run(), 4, greedy=True)
    assert set(ev) == FILE_KEYS
    np.testing.assert_array_equal(ev["seeds"], EVAL_SEEDS)
    run = EvalRun.load(counting_run)
    evaluate(run, [2], "greedy", seeds=EVAL_SEEDS)
    got = load_eval(run.dir, 2, greedy=True)
    assert set(got) == FILE_KEYS | {"extras/skill", "extras/skill_step"}
    assert str(got["method"]) == "counting"
    assert list(inspect.signature(load_eval).parameters) == [
        "run_dir",
        "update",
        "greedy",
    ]


def test_replaying_seed_and_actions_rebuilds_observations_and_events() -> None:
    run = EvalRun.load(tiny_run())
    ev = load_eval(run.dir, 4)
    env = make_craftax_env_from_name(ENV, auto_reset=False)
    returns = eval_returns(ev, ENV)
    for i, seed in enumerate(EVAL_SEEDS):
        n = int(ev["length"][i])
        ep = replay_episode(run, seed, ev["actions"][i, :n])
        assert ep.obs.shape == (n, 1345)
        obs0, _ = env.reset(jax.random.PRNGKey(seed), env.default_params)
        np.testing.assert_array_equal(ep.obs[0], obs0)
        for f in fields(StepEvents):
            want = ev[f"events/{f.name}"][i, :n]
            np.testing.assert_array_equal(getattr(ep.events, f.name), want)
        np.testing.assert_allclose(ep.reward.sum(), returns[i], atol=1e-4)


@pytest.mark.parametrize(
    ("phase", "eval_mode", "updates"),
    [("tune-coarse", "final", [4]), ("final-1m", "log_spaced", [0, 1, 2, 4])],
)
def test_tuning_runs_evaluate_the_last_checkpoint_finals_every_one(
    fresh_ppo_run: Path, phase: str, eval_mode: str, updates: list[int]
) -> None:
    got = evaluate_spec(
        fresh_ppo_run, _spec(phase, eval_mode), lambda: False, seeds=EVAL_SEEDS
    )
    for mode in MODES:
        files = sorted((fresh_ppo_run / "eval" / mode).glob("*.npz"))
        assert [int(p.stem) for p in files] == updates
    assert [(s.update, s.mode) for s in got] == [(u, m) for u in updates for m in MODES]
    run = EvalRun.load(fresh_ppo_run)
    for s in got:
        assert s.env_steps == run.env_steps(s.update)
        assert s.ret == pytest.approx(
            eval_returns(load_eval(run.dir, s.update, s.mode == "greedy"), ENV).mean()
        )
        assert s.score_pct == pytest.approx(s.ret / 22 * 100)


def test_an_interrupted_evaluation_skips_the_files_already_written(
    fresh_ppo_run: Path,
) -> None:
    spec = _spec("final-1m", "log_spaced")
    first = evaluate_spec(
        fresh_ppo_run,
        spec,
        lambda: any((fresh_ppo_run / "eval").rglob("*.npz")),
        seeds=EVAL_SEEDS,
    )
    assert [(s.update, s.mode) for s in first] == [(0, "stochastic")]
    written = fresh_ppo_run / "eval" / "stochastic" / "00000.npz"
    stamp = written.stat().st_mtime_ns
    rest = evaluate_spec(fresh_ppo_run, spec, lambda: False, seeds=EVAL_SEEDS)
    assert (0, "stochastic") not in [(s.update, s.mode) for s in rest]
    assert len(rest) == 4 * len(MODES) - 1
    assert written.stat().st_mtime_ns == stamp


def test_episode_frames_show_every_state_of_the_episode() -> None:
    run = EvalRun.load(tiny_run())
    seed = EVAL_SEEDS[1]
    a = episode_frames(run, 4, seed)
    b = episode_frames(run, 4, seed)
    n = int(load_eval(run.dir, 4)["length"][1])
    assert a.shape == (n + 1, 144, 144, 3)
    assert np.array_equal(a, b)


def test_as_rollout_replays_to_the_same_frames() -> None:
    run = EvalRun.load(tiny_run())
    seed = EVAL_SEEDS[0]
    r = as_rollout(run, 4, seed)
    frames = np.concatenate([
        c.frames for c in replay(r, bps=16) if c.frames is not None
    ])
    assert np.array_equal(frames, episode_frames(run, 4, seed))
    assert r.features.shape[0] == r.steps + 1
    assert r.policy == "ppo"
