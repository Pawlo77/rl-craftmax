"""Tests for fixed-seed evaluation: chunked rollouts, eval files, replay, schedule."""

import inspect
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import fields, make_dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _learning_run import EVAL_SEEDS, TINY, runner_config, tiny_run
from _method_runs import CLASSIC, CONFIGS, FAMILY_RUNS, UPDATES, method_run
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.events import StepEvents
from wut_murl_craftax.orchestration.queue import RunSpec, load_method
from wut_murl_craftax.orchestration.runtime import device_roles
from wut_murl_craftax.training import count_bonus, interface, ppo_rnn, pqn_rnn, sil
from wut_murl_craftax.training.checkpoints import save_checkpoint
from wut_murl_craftax.training.dyna_q import DYNA_KEYS
from wut_murl_craftax.training.hooks import HookFields, HookMethod
from wut_murl_craftax.training.interface import METHODS, combo_trainer, run_trainer
from wut_murl_craftax.training.ppo import PPOPolicy
from wut_murl_craftax.training.ppo_rnn import PPORNNConfig, make_trainer
from wut_murl_craftax.training.records import achievement_weights
from wut_murl_craftax.training.skills import (
    NUM_SKILLS,
    PRIMITIVE,
    SKILL_IDS,
    step_limit,
)
from wut_murl_craftax.visualisation.eval_rollouts import (
    EVAL_SEEDS as PRODUCTION_SEEDS,
)
from wut_murl_craftax.visualisation.eval_rollouts import (
    MODES,
    Q_FAMILIES,
    EvalRun,
    as_rollout,
    episode_frames,
    episode_outputs,
    eval_returns,
    evaluate,
    evaluate_spec,
    load_eval,
    network_family,
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
        return {
            "t": jnp.zeros(n, jnp.int32),
            "r": jnp.zeros(n),
            "d": jnp.zeros(n, bool),
        }

    def step(
        self,
        params: Mapping[str, jax.Array],
        carry: Mapping[str, jax.Array],
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Also keeps the previous step's reward and done that evaluation passes."""
        del obs, key, greedy
        assert reward is not None
        assert done is not None
        nxt = {"t": carry["t"] + 1, "r": reward, "d": done}
        return (carry["t"] + params["shift"]) % MOVES, nxt

    def extras(self, carry: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """A two-level policy's running skill and skill step, read from the carry."""
        return {"skill": carry["t"] // 4, "skill_step": carry["t"] % 4}


class _InputsPolicy(_CountingPolicy):
    """`_CountingPolicy` whose extras are the reward and done it was handed."""

    def extras(self, carry: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        return {"reward": carry["r"], "done": carry["d"]}


class _CountingTrainer:
    """The registry entry of `_CountingPolicy`; its params are one shift."""

    policy_class: type[_CountingPolicy] = _CountingPolicy

    def init(self, config: Mapping[str, Any], key: jax.Array) -> dict[str, jax.Array]:
        del config, key
        return {"shift": jnp.zeros((), jnp.int32)}

    def params(self, state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return state

    def policy(self, state: dict[str, jax.Array]) -> _CountingPolicy:
        return self.policy_class()

    def env_wrapper(self, state: dict[str, jax.Array]) -> None:
        return None


@pytest.fixture
def counting_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A run of the counting entry, registered in `METHODS`, checkpoints 0 to 2."""
    # `METHODS` is read-only (it builds ``combo`` lazily) and reads `_ENTRIES` live
    monkeypatch.setitem(interface._ENTRIES, "counting", _CountingTrainer())
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


COMBO_1M = Path(__file__).parents[1] / "training/fixtures/combo-1m-config.json"
"""``config.json`` of the combo-1m runs: count-bonus-ppo-rnn + sil, no ``combine``."""


@pytest.mark.parametrize("recorded", [False, True])
def test_a_combo_run_loads_its_own_pair_after_combo_yaml_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: bool
) -> None:
    """Ruling R80a: the pair comes from config.json (recorded, else its keys)."""
    cfg = json.loads(COMBO_1M.read_text())
    if recorded:
        cfg["combine"] = {"base": "ppo-rnn", "methods": ["count-bonus-ppo-rnn", "sil"]}
    run = tmp_path / "combo"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(cfg))
    # combo.yaml now names another pair: sil + a toy hook method with key toy_k
    toy = HookMethod(
        make_dataclass("ToyConfig", [("toy_k", int, 1)], frozen=True),
        lambda c, b: HookFields(),
    )
    monkeypatch.setitem(
        interface._ENTRIES, "toy", make_trainer(PPORNNConfig(), method=toy)
    )
    (tmp_path / "combo.yaml").write_text(
        "combine: {base: ppo-rnn, methods: [sil, toy]}\n"
        "base: {base_from: ppo-rnn.tuned}\n"
        "grid: {coarse: [{name: combined}], fine: null}\n"
        "tuned: null\n"
    )
    monkeypatch.setattr(
        interface, "_combo", lambda: combo_trainer(interface._ENTRIES, tmp_path)
    )
    pairs = {"live": METHODS["combo"], "built": run_trainer(cfg)}
    hooked = {k: getattr(t, "method", None) for k, t in pairs.items()}
    assert hooked["live"].keys == sil.METHOD.keys | {"toy_k"}  # pair Y is the live one
    assert hooked["built"].keys == count_bonus.METHOD.keys | sil.METHOD.keys  # pair X
    loaded = EvalRun.load(run)
    assert loaded.cfg == cfg
    assert loaded.method == "combo"


def test_each_step_gets_the_reward_and_done_of_the_step_before(
    counting_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zeros at the episode start, then the transition that produced ``obs``."""
    inputs = _CountingTrainer()
    inputs.policy_class = _InputsPolicy
    monkeypatch.setitem(interface._ENTRIES, "counting", inputs)
    run = EvalRun.load(counting_run)
    out = rollout_seeds(run, run.params(0), EVAL_SEEDS, greedy=True)
    assert out is not None
    unlocks = out["events/unlocked"] @ achievement_weights(ENV)
    rewards = unlocks + 0.1 * out["events/health_delta"]
    for i, n in enumerate(out["length"]):
        np.testing.assert_allclose(out["extras/reward"][i, 0], 0.0)
        np.testing.assert_allclose(
            out["extras/reward"][i, 1:n], rewards[i, : n - 1], atol=1e-5
        )
        assert not out["extras/done"][i, :n].any()


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
    # the network re-runs at the checkpoint's epsilon: 1 at update 0, so uniform
    for mode, want in (("greedy", greedy[0]), ("stochastic", stochastic[0])):
        probs, _ = episode_outputs(run, 0, mode, EVAL_SEEDS[0])
        if mode == "greedy":
            np.testing.assert_array_equal(probs.argmax(1), want[: len(probs)])
        else:
            np.testing.assert_allclose(probs, 1 / probs.shape[1], atol=1e-6)


# --- network outputs of every method ------------------------------------------------

N_ACTIONS = 17
"""Craftax-Classic actions."""
FAMILIES = {
    "ppo": "ppo",
    "ppo-rnn": "ppo-rnn",
    "count-bonus-ppo-rnn": "ppo-rnn",
    "sil": "ppo-rnn",
    "combo": "ppo-rnn",
    "pqn-rnn": "pqn-rnn",
    "count-bonus-pqn-rnn": "pqn-rnn",
    "smdp-skills": "skills",
    "dyna-q": "skills",
}
"""The network family each registry entry's policy re-runs through."""


def test_every_registry_entry_has_a_network_family() -> None:
    """Combo goes through its own pair (`run_trainer`), never the live combo.yaml."""
    assert set(METHODS) == FAMILIES.keys()
    key, combo = jax.random.key(0), json.loads(COMBO_1M.read_text())
    for name, family in FAMILIES.items():
        hooked = getattr(interface._ENTRIES.get(name), "method", None)
        own = {} if hooked is None else {k: combo[k] for k in hooked.keys}
        if name == "dyna-q":  # planning keys have no defaults
            own = {k: load_method(name).base[k] for k in DYNA_KEYS}
        cfg = combo if name == "combo" else {"method": name, "env": CLASSIC} | own
        trainer = run_trainer(cfg)
        config = {k: v for k, v in cfg.items() if k != "combine"} | {"steps": 4096}
        state = jax.eval_shape(lambda t=trainer, c=config: t.init(c, key))
        assert network_family(trainer.policy(state)) == family, name
    assert {"pqn-rnn", "skills"} == Q_FAMILIES


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", FAMILY_RUNS)
def test_episode_outputs_are_a_distribution_led_by_the_greedy_action(
    name: str, mode: str
) -> None:
    run, u = EvalRun.load(method_run(name)), UPDATES[name]
    ev = load_eval(run.dir, u, mode == "greedy")
    for i, seed in enumerate(EVAL_SEEDS):
        n = int(ev["length"][i])
        probs, value = episode_outputs(run, u, mode, seed)
        assert probs.dtype == value.dtype == np.float32
        assert probs.shape == (n, N_ACTIONS)
        assert value.shape == (n,)
        assert (probs >= 0).all()
        assert np.isfinite(value).all()
        np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-5)
        if mode == "greedy":
            np.testing.assert_array_equal(probs.argmax(1), ev["actions"][i, :n])


def test_episode_outputs_refuse_an_unknown_mode_or_seed() -> None:
    run = EvalRun.load(method_run("ppo"))
    with pytest.raises(ValueError, match="mode"):
        episode_outputs(run, 4, "sampled", EVAL_SEEDS[0])
    with pytest.raises(ValueError, match="seed 9"):
        episode_outputs(run, 4, "greedy", 9)


def _stepwise(
    run: EvalRun, step: Any, obs: np.ndarray, extra: np.ndarray | None = None
) -> list[Any]:
    """Outputs of ``step(carry, obs, t) -> (carry, out)`` applied one step at a time."""
    carry, outs = run.policy.init_carry(1), []
    for t, o in enumerate(obs):
        x = o if extra is None else np.concatenate([o, extra[t]])
        carry, out = step(carry, jnp.asarray(x)[None, None], t)
        outs.append(out)
    return outs


def test_ppo_rnn_outputs_equal_a_live_rollout_of_the_seed() -> None:
    """The GRU carry is threaded from zeros over the episode, as the policy acts."""
    run, u, seed = EvalRun.load(method_run("ppo-rnn")), 2, EVAL_SEEDS[0]
    ep, _ = stored_episode(run, u, seed)
    params, apply = run.params(u), jax.jit(ppo_rnn.apply_network, static_argnums=0)
    no_reset = jnp.zeros((1, 1), bool)

    def step(h: jax.Array, x: jax.Array, t: int) -> Any:
        h, logits, v = apply(run.policy.net, params, h, x, no_reset)
        return h, (jax.nn.softmax(logits[0, 0]), v[0, 0])

    probs, value = map(np.stack, zip(*_stepwise(run, step, ep.obs), strict=True))
    got = episode_outputs(run, u, "stochastic", seed)
    np.testing.assert_allclose(got[0], probs, atol=1e-5)
    np.testing.assert_allclose(got[1], value, atol=1e-5)


def _eps_greedy(q: np.ndarray, eps: float) -> np.ndarray:
    return eps / q.shape[-1] + (1 - eps) * np.eye(q.shape[-1])[q.argmax(-1)]


def _live_q(
    run: EvalRun, u: int, obs: np.ndarray, actions: np.ndarray, extra: Any = None
) -> np.ndarray:
    """Q-values of a live PQN-RNN rollout: the previous action is an input."""
    variables = run.params(u)["variables"]
    apply = jax.jit(pqn_rnn.apply_network, static_argnums=0)
    no_reset = jnp.zeros((1, 1), bool)
    last = np.r_[pqn_rnn.NO_ACTION, actions[:-1]].astype(np.int32)

    def step(c: Any, x: jax.Array, t: int) -> Any:
        lstm, q = apply(
            run.policy.net, variables, c[0], x, no_reset, last[t, None, None]
        )
        return (lstm, c[1]), np.asarray(q[0, 0])

    if extra is None:
        pad = run.policy.net.obs_dim - obs.shape[1]
        extra = np.zeros((len(obs), pad), obs.dtype)
    return np.stack(_stepwise(run, step, obs, extra))


@pytest.mark.parametrize("u", [0, 2])
def test_pqn_rnn_outputs_are_max_q_and_epsilon_greedy_at_the_checkpoint(
    u: int,
) -> None:
    run, seed = EvalRun.load(method_run("pqn-rnn")), EVAL_SEEDS[0]
    ep, actions = stored_episode(run, u, seed)
    q = _live_q(run, u, ep.obs, actions)
    eps = float(run.params(u)["eps"])
    assert (eps == 1.0) if u == 0 else (0.1 < eps < 0.9)
    probs, value = episode_outputs(run, u, "stochastic", seed)
    np.testing.assert_allclose(value, q.max(1), atol=1e-5)
    np.testing.assert_allclose(probs, _eps_greedy(q, eps), atol=1e-5)
    ep, actions = stored_episode(run, u, seed, greedy=True)
    probs, _ = episode_outputs(run, u, "greedy", seed)
    np.testing.assert_array_equal(probs, np.eye(N_ACTIONS)[actions])


def _skill_inputs(ev: dict[str, np.ndarray], i: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(ev["length"][i])
    return ev["extras/skill"][i, :n], ev["extras/steps_in_skill"][i, :n]


def test_the_skill_network_reads_the_stored_skill_and_its_step_fraction() -> None:
    """Input: observation, skill one-hot, steps / step limit at the run's timeout."""
    run, u, seed = EvalRun.load(method_run("smdp-skills")), 2, EVAL_SEEDS[0]
    ev = load_eval(run.dir, u)
    skill, steps = _skill_inputs(ev, 0)
    assert ev["extras/skill"].dtype == np.uint8
    assert ev["extras/steps_in_skill"].dtype == np.uint16
    assert (steps > 0).any()  # a learned skill ran for more than one step
    ep, actions = stored_episode(run, u, seed)
    limit = np.asarray(step_limit(skill.astype(np.int32), 0.5))
    assert CONFIGS["smdp-skills"]["timeout_mult"] == 0.5
    frac = (steps / limit).astype(np.float32)[:, None]
    extra = np.concatenate([np.eye(NUM_SKILLS)[skill], frac], 1).astype(np.float32)
    q = _live_q(run, u, ep.obs, actions, extra)
    eps = float(run.params(u)["eps"])
    probs, value = episode_outputs(run, u, "stochastic", seed)
    np.testing.assert_allclose(value, q.max(1), atol=1e-5)
    np.testing.assert_allclose(probs, _eps_greedy(q, eps), atol=1e-5)
    with pytest.raises(ValueError, match="skill"):
        network_outputs(run, u, ep.obs, actions, ep.done, False)


def test_primitive_skill_steps_are_one_hot_on_the_stored_action() -> None:
    """``read_book`` and ``level_up`` act with a fixed action, never the network's."""
    run, u, seed = EvalRun.load(method_run("smdp-skills")), 2, EVAL_SEEDS[0]
    ev = load_eval(run.dir, u)
    skill, steps = (x.copy() for x in _skill_inputs(ev, 0))
    skill[::3], steps[::3] = SKILL_IDS["read_book"], 0
    skill[1::7] = SKILL_IDS["level_up"]
    ep, actions = stored_episode(run, u, seed)
    probs, _ = network_outputs(run, u, ep.obs, actions, ep.done, False, skill, steps)
    prim = PRIMITIVE[skill]
    assert prim.any()
    assert not prim.all()
    np.testing.assert_array_equal(probs[prim], np.eye(N_ACTIONS)[actions[prim]])
    assert (probs[~prim].max(1) < 0.9).all()  # epsilon about 0.5 elsewhere
