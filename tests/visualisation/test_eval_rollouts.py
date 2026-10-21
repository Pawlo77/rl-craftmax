"""Tests for fixed-seed evaluation: chunked rollouts, eval files, replay, schedule."""

import inspect
import json
import os
import shutil
import subprocess
import sys
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
from wut_murl_craftax.orchestration.runtime import device_roles
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
    network_outputs,
    replay_episode,
    rollout_seeds,
    stored_episode,
)
from wut_murl_craftax.visualisation.rollout import replay

ENV = TINY.env
EVENT_KEYS = {f"events/{f.name}" for f in fields(StepEvents)}
FILE_KEYS = {"seeds", "actions", "length", "method", "devices"} | EVENT_KEYS
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

    def env_wrapper(self, state: dict[str, jax.Array]) -> None:
        return None


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


_OPEN_THEN_TRAIN = """
import json, sys
from pathlib import Path
import jax
from wut_murl_craftax.orchestration.runtime import configure_runtime
configure_runtime()
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.visualisation.eval_rollouts import EvalRun
run = Path(sys.argv[1])
EvalRun.load(run)
METHODS["ppo"].init({**json.loads((run / "config.json").read_text()), "steps": 64},
                    jax.random.key(0))
print("ok")
"""


@pytest.mark.parametrize("env", ["Craftax-Symbolic-v1", "Craftax-Classic-Symbolic-v1"])
def test_opening_a_run_first_leaves_craftax_usable_outside_the_trace(
    tmp_path: Path, env: str
) -> None:
    """The queue evaluates a FINISHED run before training the next one in a process.

    ``load`` traces ``init`` with ``eval_shape``; Craftax imported for the first time
    inside that trace kept tracers as module constants and the next eager ``reset``
    raised ``UnexpectedTracerError``. A fresh interpreter makes ``load`` come first.
    """
    cfg = runner_config(TINY) | {"env": env, "envs": 2, "num_steps": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    out = subprocess.run(
        [sys.executable, "-c", _OPEN_THEN_TRAIN, str(tmp_path)],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split()[-1] == "ok"


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


def test_the_eval_file_records_the_devices_it_ran_on() -> None:
    devices = json.loads(str(load_eval(tiny_run(), 4)["devices"]))
    roles = device_roles()
    assert devices == {
        "actor": str(roles.actor),
        "learner": str(roles.learner),
        "shards": max(1, len(jax.devices("cpu"))),
    }


def test_replays_of_different_lengths_compile_once() -> None:
    run = EvalRun.load(tiny_run())
    ev = load_eval(run.dir, 4)
    assert len(set(ev["length"].tolist())) > 1
    for i, seed in enumerate(EVAL_SEEDS):
        ep, actions = stored_episode(run, 4, seed)
        assert len(ep.obs) == len(actions) == ev["length"][i]
        assert jax.tree.leaves(ep.states)[0].shape[0] == len(actions) + 1
    assert run._replay._cache_size() == 1


def test_a_short_episode_beside_a_long_one_replays_at_its_own_width(
    fresh_ppo_run: Path,
) -> None:
    # the file's width is set by its longest seed: widen it as a 2,500-step seed would
    ev = load_eval(tiny_run(), 4)
    width = 3 * 1000
    wide = {
        k: np.pad(
            v,
            [(0, 0), (0, width - v.shape[1]), *[(0, 0)] * (v.ndim - 2)],
            constant_values=-1 if k == "actions" else 0,
        )
        if v.ndim >= 2
        else v
        for k, v in ev.items()
    }
    path = fresh_ppo_run / "eval" / "stochastic" / "00004.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, **wide)
    run = EvalRun.load(fresh_ppo_run)
    real, widths = run._replay, []

    def spy(seed: jax.Array, actions: jax.Array) -> Any:
        widths.append(actions.shape[0])
        return real(seed, actions)

    run.__dict__["_replay"] = spy  # the cached_property's slot
    for i, seed in enumerate(EVAL_SEEDS):
        ep, actions = stored_episode(run, 4, seed)
        assert len(ep.obs) == len(actions) == ev["length"][i]
    replay_episode(run, EVAL_SEEDS[0], np.zeros(1000, np.int32))  # a full bucket
    assert widths == [1000, 1000, 1000]
    assert real._cache_size() == 1


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
    spec = _spec(phase, eval_mode)
    got = list(evaluate_spec(fresh_ppo_run, spec, lambda: False, seeds=EVAL_SEEDS))
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
    first = list(
        evaluate_spec(
            fresh_ppo_run,
            spec,
            lambda: any((fresh_ppo_run / "eval").rglob("*.npz")),
            seeds=EVAL_SEEDS,
        )
    )
    assert [(s.update, s.mode) for s in first] == [(0, "stochastic")]
    written = fresh_ppo_run / "eval" / "stochastic" / "00000.npz"
    stamp = written.stat().st_mtime_ns
    # the rerun reads the written file back (for logging) and plays the other 7
    rest = list(evaluate_spec(fresh_ppo_run, spec, lambda: False, seeds=EVAL_SEEDS))
    assert [(s.update, s.mode) for s in rest] == [
        (u, m) for u in (0, 1, 2, 4) for m in MODES
    ]
    assert rest[0] == first[0]
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


PQN_CONFIG = {"method": "pqn-rnn", "env": ENV, "envs": 8, "num_steps": 8}
"""A tiny PQN-RNN run config as the runner writes it (no ``steps``)."""
PQN_CONFIG |= {"hidden": 32, "minibatches": 2, "epochs": 1}


@pytest.fixture
def pqn_run(tmp_path: Path) -> Path:
    """4 PQN-RNN updates on Classic with checkpoints 0 and 4, as the runner saves."""
    trainer, run = METHODS["pqn-rnn"], tmp_path / "pqn"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(PQN_CONFIG))
    spu = trainer.steps_per_update(PQN_CONFIG)
    state = trainer.init(PQN_CONFIG | {"steps": 4 * spu}, jax.random.key(0))
    save_checkpoint(run, 0, 0, 0.0, trainer.params(state))
    state, _ = trainer.train_chunk(state, 4)
    save_checkpoint(run, 4, 4 * spu, 1.0, trainer.params(state))
    return run


def test_pqn_rnn_checkpoints_evaluate_in_both_modes_at_their_epsilon(
    pqn_run: Path,
) -> None:
    """Variables with BatchRenorm statistics and eps come from ``ckpt/``; the
    stochastic mode acts at the checkpoint's epsilon (1 at update 0, 0.005 at 4)."""
    run = EvalRun.load(pqn_run)
    first, last = run.params(0), run.params(4)
    assert set(last["variables"]) == {"params", "batch_stats"}
    assert float(first["eps"]) == 1.0
    np.testing.assert_allclose(float(last["eps"]), 0.005, rtol=1e-6)
    for mode in MODES:
        evaluate(run, [0, 4], mode, seeds=EVAL_SEEDS)
    for u in (0, 4):
        for greedy in (False, True):
            ev = load_eval(pqn_run, u, greedy)
            assert set(ev) == FILE_KEYS
            assert str(ev["method"]) == "pqn-rnn"
    stochastic, greedy = (load_eval(pqn_run, 0, g)["actions"] for g in (False, True))
    assert (stochastic != greedy).any()
    # at epsilon 0 the stochastic mode is the greedy one, so it reads params["eps"]
    calm = rollout_seeds(run, first | {"eps": np.float32(0.0)}, EVAL_SEEDS, False)
    assert calm is not None
    np.testing.assert_array_equal(calm["actions"], greedy)
    with pytest.raises(NotImplementedError, match="pqn-rnn"):
        network_outputs(run, 4, np.zeros((1, 1)), np.zeros(1), np.zeros(1), True)
