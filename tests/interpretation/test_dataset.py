"""Tests for the distillation datasets: seed splits, rollouts, labels and features."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _toys import CLASSIC, toy_run

from wut_murl_craftax.interpretation import (
    FEATURES,
    MAX_STATES,
    MEMORY_FEATURES,
    SKILL_FEATURES,
    SPLITS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    Dataset,
    collect,
    dataset,
    dataset_path,
    featurize,
    relabel,
)
from wut_murl_craftax.training.checkpoints import save_checkpoint
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.skills import NUM_SKILLS
from wut_murl_craftax.visualisation.eval_rollouts import (
    EvalRun,
    evaluate,
    load_eval,
    network_outputs,
    rollout_seeds,
    stored_episode,
)

N_ACTIONS = 17
MOVES = 5
"""The toy network's greedy action cycles NOOP and the four moves."""
SKILL_ACTIONS = (0, 5)
"""Toy skill 0 always plays NOOP, skill 1 always DO."""
SKILL_LEN = 7
"""The toy top level switches skill every 7 steps."""
SEEDS = TRAIN_SEEDS[:2]


class _CountingPolicy:
    """Recurrent toy: the carry counts steps; greedy plays the count mod 5."""

    def init_carry(self, n: int) -> dict[str, jax.Array]:
        return {"t": jnp.zeros(n, jnp.int32)}

    def step(
        self,
        params: Any,
        carry: Mapping[str, jax.Array],
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del params, obs, reward, done
        t = carry["t"]
        sampled = jax.random.randint(key, t.shape, 0, MOVES)
        return jnp.where(greedy, t % MOVES, sampled), {"t": t + 1}


class _SkillPolicy:
    """Two-level toy: skill ``(t // 7) % 2`` runs, each a fixed primitive action."""

    def init_carry(self, n: int) -> dict[str, jax.Array]:
        zeros = jnp.zeros(n, jnp.int32)
        return {"t": zeros, "skill": zeros, "steps": zeros}

    def step(
        self,
        params: Any,
        carry: Mapping[str, jax.Array],
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del params, obs, key, greedy, reward, done
        t = carry["t"]
        skill = (t // SKILL_LEN) % 2
        action = jnp.asarray(SKILL_ACTIONS)[skill]
        return action, {"t": t + 1, "skill": skill, "steps": t % SKILL_LEN}

    def extras(self, carry: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        """The skill and skill step of the step just taken, as `SMDPPolicy.extras`."""
        return {
            "skill": carry["skill"].astype(jnp.uint8),
            "steps_in_skill": carry["steps"].astype(jnp.uint16),
        }


def _mixed(action: np.ndarray, greedy: bool) -> np.ndarray:
    """Greedy one-hot, else half uniform: rows sum to 1, argmax is ``action``."""
    eps = 0.0 if greedy else 0.5
    return (eps / N_ACTIONS + (1 - eps) * np.eye(N_ACTIONS)[action]).astype(np.float32)


def _counting_outputs(
    run: EvalRun,
    update: int,
    obs: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    greedy: bool,
    skill: np.ndarray | None = None,
    skill_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A recurrent network: the hidden state counts steps since the last reset."""
    del run, update, actions, skill, skill_steps
    h = np.zeros(len(obs), np.int64)
    for t in range(1, len(obs)):
        h[t] = 0 if dones[t - 1] else h[t - 1] + 1
    return _mixed(h % MOVES, greedy), np.zeros(len(obs), np.float32)


def _skill_outputs(
    run: EvalRun,
    update: int,
    obs: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    greedy: bool,
    skill: np.ndarray | None = None,
    skill_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The skill network: the running skill's fixed primitive action."""
    del run, update, actions, dones, skill_steps
    assert skill is not None
    action = np.asarray(SKILL_ACTIONS)[skill]
    return _mixed(action, greedy), np.zeros(len(obs), np.float32)


@pytest.fixture
def counting_run(tmp_path: Path) -> EvalRun:
    return toy_run(tmp_path / "counting", _CountingPolicy())


@pytest.fixture
def skill_run(tmp_path: Path) -> EvalRun:
    return toy_run(tmp_path / "skills", _SkillPolicy())


@pytest.fixture(scope="module")
def ppo_run(tmp_path_factory: pytest.TempPathFactory) -> EvalRun:
    """An untrained feed-forward PPO checkpoint on Craftax-Classic."""
    root = tmp_path_factory.mktemp("ppo")
    cfg = {"method": "ppo", "env": CLASSIC, "envs": 16, "num_steps": 8}
    cfg |= {"minibatches": 2, "width": 32}
    (root / "config.json").write_text(json.dumps(cfg))
    trainer = METHODS["ppo"]
    state = trainer.init(
        cfg | {"steps": trainer.steps_per_update(cfg)}, jax.random.key(0)
    )
    save_checkpoint(root, 0, 0, 0.0, trainer.params(state))
    return EvalRun.load(root)


def _equal(a: Dataset, b: Dataset) -> None:
    assert (a.columns, a.family, a.recurrent) == (b.columns, b.family, b.recurrent)
    for name in ("features", "label", "probs", "episode", "step", "floor"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), name)


def test_the_seed_splits_are_the_plans_and_disjoint() -> None:
    assert tuple(range(30000, 30256)) == TRAIN_SEEDS
    assert tuple(range(10000, 10064)) == VAL_SEEDS
    assert tuple(range(20000, 20064)) == TEST_SEEDS
    assert SPLITS == {"train": TRAIN_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS}
    sets = [set(s) for s in SPLITS.values()]
    assert sum(map(len, sets)) == len(set.union(*sets))


def test_two_collections_of_a_network_policy_are_identical(ppo_run: EvalRun) -> None:
    a, b = (collect(ppo_run, 0, SEEDS, max_states_per_episode=40) for _ in range(2))
    _equal(a, b)
    assert a.columns == (*FEATURES["classic"], *MEMORY_FEATURES)
    assert a.features.shape == (len(a.label), len(a.columns))
    assert a.features.dtype == a.probs.dtype == np.float32
    np.testing.assert_allclose(a.probs.sum(1), 1, rtol=1e-5)
    # greedy-mode outputs of a feed-forward softmax policy rank as the stochastic ones
    np.testing.assert_array_equal(a.label, a.probs.argmax(1))
    assert set(a.episode) == set(SEEDS)
    assert not a.floor.any()  # Classic has one floor
    assert (a.family, a.recurrent) == ("ppo", False)  # memory proxies unused


def test_a_failed_save_leaves_no_file_and_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = Dataset(
        features=np.zeros((2, 3), np.float32),
        columns=("a", "b", "c"),
        label=np.zeros(2, np.int8),
        probs=np.full((2, 4), 0.25, np.float32),
        episode=np.zeros(2, np.int32),
        step=np.arange(2, dtype=np.int32),
        floor=np.zeros(2, np.int8),
        family="ppo",
        recurrent=False,
    )

    def boom(*a: Any, **k: Any) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError, match="disk full"):
        ds.save(tmp_path / "train.npz")
    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    ds.save(tmp_path / "train.npz")
    assert [p.name for p in tmp_path.iterdir()] == ["train.npz"]


def test_an_empty_seed_list_is_refused(ppo_run: EvalRun) -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        collect(ppo_run, 0, ())
    with pytest.raises(ValueError, match="at least one seed"):
        collect(ppo_run, 0, np.array([], np.int64))


def test_seeds_may_be_a_numpy_array(ppo_run: EvalRun) -> None:
    as_list = collect(ppo_run, 0, SEEDS[:2])
    as_array = collect(ppo_run, 0, np.array(SEEDS[:2]))
    np.testing.assert_array_equal(as_array.features, as_list.features)
    np.testing.assert_array_equal(as_array.label, as_list.label)


def test_probs_and_labels_are_network_outputs_over_each_stochastic_episode(
    ppo_run: EvalRun,
) -> None:
    ds = collect(ppo_run, 0, SEEDS[:1], max_states_per_episode=None)
    evaluate(ppo_run, [0], "stochastic", seeds=SEEDS[:1])  # the same episode, stored
    ep, actions = stored_episode(ppo_run, 0, SEEDS[0])
    probs, _ = network_outputs(ppo_run, 0, ep.obs, actions, ep.done, greedy=False)
    np.testing.assert_array_equal(ds.probs, probs)
    np.testing.assert_array_equal(ds.step, np.arange(len(actions)))
    np.testing.assert_array_equal(
        ds.features[:, : len(FEATURES["classic"])], featurize(ep.obs, "classic")
    )
    since, last = [], -1  # steps since the last unlock, from the episode start
    for t, unlocked in enumerate(ep.events.unlocked.any(1)):
        since.append(t - last - 1)
        last = t if unlocked else last
    np.testing.assert_array_equal(ds.features[:, -1], since)


def test_episodes_keep_at_most_the_cap_sampled_uniformly(counting_run: EvalRun) -> None:
    assert MAX_STATES == 2000
    full = collect(counting_run, 0, SEEDS, None, _counting_outputs, "ppo-rnn")
    capped = collect(counting_run, 0, SEEDS, 50, _counting_outputs, "ppo-rnn")
    for seed in SEEDS:
        steps = full.step[full.episode == seed]
        np.testing.assert_array_equal(steps, np.arange(len(steps)))
        kept = capped.step[capped.episode == seed]
        assert len(kept) == 50 < len(steps)
        assert (np.diff(kept) > 0).all()  # sorted, no repeats
        assert kept[-1] > len(steps) // 2  # spread over the episode, not its head


def _stored_actions(run: EvalRun, seeds: tuple[int, ...]) -> list[np.ndarray]:
    """Each seed's stochastic actions, from a stored eval file written for the test."""
    evaluate(run, [0], "stochastic", seeds=seeds)
    ev = load_eval(run.dir, 0)
    return [ev["actions"][i, : ev["length"][i]] for i in range(len(seeds))]


def test_a_recurrent_policys_labels_follow_its_carried_state(
    counting_run: EvalRun,
) -> None:
    """Outputs run over the whole episode before subsampling, carry threaded."""
    ds = collect(counting_run, 0, SEEDS, 30, _counting_outputs, "ppo-rnn")
    np.testing.assert_array_equal(ds.label, ds.step % MOVES)
    assert ds.recurrent
    np.testing.assert_allclose(ds.probs.max(1), 0.5 + 0.5 / N_ACTIONS, rtol=1e-6)
    ev = _stored_actions(counting_run, SEEDS)
    prev = ds.features[:, ds.columns.index("prev_action")]
    for i, seed in enumerate(SEEDS):
        rows = ds.episode == seed
        steps = ds.step[rows]
        expected = np.r_[-1, ev[i]][steps]  # the action taken one step earlier
        np.testing.assert_array_equal(prev[rows], expected)
    # the random stochastic actions differ from the greedy labels somewhere
    assert any(
        (ev[i][ds.step[ds.episode == s]] != ds.label[ds.episode == s]).any()
        for i, s in enumerate(SEEDS)
    )


def test_a_two_level_policy_labels_primitive_actions_with_the_skill_as_features(
    skill_run: EvalRun,
) -> None:
    ds = collect(skill_run, 0, SEEDS, 60, _skill_outputs, "skills")
    skill = (ds.step // SKILL_LEN) % 2
    np.testing.assert_array_equal(ds.label, np.asarray(SKILL_ACTIONS)[skill])
    assert (ds.family, ds.recurrent) == ("skills", True)  # an LSTM skill network
    assert ds.columns == (*FEATURES["classic"], *MEMORY_FEATURES, *SKILL_FEATURES)
    assert len(SKILL_FEATURES) == NUM_SKILLS + 1
    one_hot = ds.features[:, -NUM_SKILLS - 1 : -1]
    np.testing.assert_array_equal(one_hot, np.eye(NUM_SKILLS)[skill])
    np.testing.assert_array_equal(ds.features[:, -1], ds.step % SKILL_LEN)


def test_a_stored_eval_file_with_every_seed_is_reused(
    counting_run: EvalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = collect(counting_run, 0, SEEDS, 40, _counting_outputs, "ppo-rnn")
    evaluate(counting_run, [0], "stochastic", seeds=SEEDS[::-1])  # rows by seed

    def no_rollout(*args: object, **kwargs: object) -> None:
        msg = "rolled out although the eval file holds the seeds"
        raise AssertionError(msg)

    monkeypatch.setattr(dataset, "rollout_seeds", no_rollout)
    _equal(collect(counting_run, 0, SEEDS, 40, _counting_outputs, "ppo-rnn"), fresh)
    with pytest.raises(AssertionError, match="rolled out"):
        collect(
            counting_run, 0, (SEEDS[0], TEST_SEEDS[0]), 40, _counting_outputs, "ppo-rnn"
        )


def test_a_dataset_saves_under_budget_method_run_and_split(
    counting_run: EvalRun, tmp_path: Path
) -> None:
    path = dataset_path("1m", "ppo", "abc123", "train", tmp_path)
    assert path == tmp_path / "1m" / "ppo" / "abc123" / "train.npz"
    assert dataset_path("1m", "ppo", "abc123", "test") == Path(
        "outputs/interpretation/1m/ppo/abc123/test.npz"
    )
    ds = collect(counting_run, 0, SEEDS, 20, _counting_outputs, "ppo-rnn")
    ds.save(path)
    _equal(Dataset.load(path), ds)


def test_relabel_labels_the_states_of_any_rollout_as_collect_does(
    counting_run: EvalRun,
) -> None:
    """DAgger: another policy's eval arrays, labelled by the network of the run."""
    played = rollout_seeds(counting_run, counting_run.params(0), SEEDS, greedy=True)
    assert played is not None
    ds = relabel(counting_run, 0, played, 30, _counting_outputs, "ppo-rnn")
    np.testing.assert_array_equal(ds.label, ds.step % MOVES)  # the network's labels
    prev = ds.features[:, ds.columns.index("prev_action")]
    for i, seed in enumerate(SEEDS):
        rows = ds.episode == seed
        acted = played["actions"][i, : played["length"][i]]
        np.testing.assert_array_equal(prev[rows], np.r_[-1, acted][ds.step[rows]])
    stochastic = rollout_seeds(counting_run, counting_run.params(0), SEEDS, False)
    assert stochastic is not None
    _equal(
        relabel(counting_run, 0, stochastic, 30, _counting_outputs, "ppo-rnn"),
        collect(counting_run, 0, SEEDS, 30, _counting_outputs, "ppo-rnn"),
    )
