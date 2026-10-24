"""Tests for tree policies: both protocols, memory proxies, two levels, closed loop."""

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _toys import CLASSIC, toy_run
from craftax.craftax_classic.constants import BlockType
from sklearn.tree import DecisionTreeClassifier

from wut_murl_craftax.interpretation import (
    FEATURES,
    MEMORY_FEATURES,
    TEST_SEEDS,
    Dataset,
    FittedTree,
    TreePolicy,
    closed_loop,
    collect,
    featurize,
    tree_rollouts,
)
from wut_murl_craftax.interpretation.tree_policy import _JaxTree
from wut_murl_craftax.training.checkpoints import save_checkpoint
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.skills import PRIMITIVE, SKILLS
from wut_murl_craftax.visualisation.eval_rollouts import EvalRun, replay_episode
from wut_murl_craftax.visualisation.rollout import game_for, play

N_ACTIONS = 17
NOOP, LEFT, RIGHT, UP, DOWN, DO = range(6)
GRASS, TREE = BlockType.GRASS.value, BlockType.TREE.value
SEEDS = TEST_SEEDS[:2]
SKILL_LEN = 7
LEARNED = 0
PRIM = int(np.flatnonzero(PRIMITIVE)[0])
"""The toy top level alternates a learned skill and a primitive one."""


def _facing_rule(direction: Any, facing: Any) -> Any:
    """DO on grass and trees, else turn clockwise; numpy or jax arrays."""
    return np.where((facing == GRASS) | (facing == TREE), DO, 1 + direction % 4)


class _FacingPolicy:
    """Feed-forward toy network: greedy `_facing_rule` of the observation."""

    def init_carry(self, n: int) -> tuple[()]:
        del n
        return ()

    def step(
        self,
        params: Any,
        carry: tuple[()],
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[()]]:
        del params, reward, done
        names = FEATURES["classic"]
        f = featurize(obs, "classic", jnp)
        d, b = (f[:, names.index(c)] for c in ("direction", "facing_block"))
        rule = jnp.where((b == GRASS) | (b == TREE), DO, 1 + d % 4).astype(jnp.int32)
        sampled = jax.random.randint(key, rule.shape, 0, N_ACTIONS)
        return jnp.where(greedy, rule, sampled), carry


class _TopLevel:
    """Two-level toy: learned skill 0 (NOOP) and a primitive skill (DO) take turns
    for 7 steps each; the carry keeps the reward and done it was handed."""

    def init_carry(self, n: int) -> dict[str, jax.Array]:
        zeros = jnp.zeros(n, jnp.int32)
        return {"t": zeros, "skill": zeros, "steps": zeros} | {
            "reward": jnp.zeros(n),
            "done": jnp.zeros(n, bool),
        }

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
        del params, obs, key, greedy
        assert reward is not None
        assert done is not None
        t = carry["t"]
        skill = jnp.where((t // SKILL_LEN) % 2 == 0, LEARNED, PRIM)
        action = jnp.where(skill == LEARNED, NOOP, DO)
        new = {"t": t + 1, "skill": skill, "steps": t % SKILL_LEN}
        return action, new | {"reward": reward, "done": done}

    def extras(self, carry: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        return {
            "skill": carry["skill"].astype(jnp.uint8),
            "steps_in_skill": carry["steps"].astype(jnp.uint16),
            "reward_in": carry["reward"],
            "done_in": carry["done"],
        }


def _grid(*values: range) -> np.ndarray:
    return np.array(list(itertools.product(*values)), np.float32)


def _fit(
    columns: tuple[str, ...], x: np.ndarray, y: np.ndarray, leaves: int
) -> FittedTree:
    model = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(x, y)
    return FittedTree("leaves", leaves, model, columns, 1.0)


def _move_tree() -> FittedTree:
    """4 leaves: DO next to a tree, else a move chosen by the facing direction."""
    x = _grid(range(1, 5), range(9))
    d, near = x[:, 0], x[:, 1]
    turn = np.where(d <= 2, UP, np.where(d <= 3, LEFT, RIGHT))
    return _fit(("direction", "tree_dist"), x, np.where(near <= 1, DO, turn), 4)


def _memory_tree() -> FittedTree:
    """DO after a move, then a move chosen by the steps since the last unlock."""
    x = _grid(range(1, 5), range(-1, N_ACTIONS), range(40))
    prev, since = x[:, 1], x[:, 2]
    y = np.where(prev == DO, 1 + (since // 5) % 4, DO)
    return _fit(("direction", *MEMORY_FEATURES), x, y, 16)


def _skill_tree() -> FittedTree:
    """Moves only, chosen by the running skill and its step."""
    columns = ("direction", f"skill_{SKILLS[LEARNED].name}", "skill_steps")
    x = _grid(range(1, 5), range(2), range(SKILL_LEN))
    y = 1 + (x[:, 0] + x[:, 1] + x[:, 2]).astype(int) % 4
    return _fit(columns, x, y, 16)


def _facing_tree() -> FittedTree:
    """`_facing_rule` exactly, at the closed-loop size."""
    x = _grid(range(1, 5), range(len(BlockType)))
    return _fit(("direction", "facing_block"), x, _facing_rule(x[:, 0], x[:, 1]), 256)


def _uniform(run: EvalRun, update: int, obs: np.ndarray, *args: Any) -> Any:
    """Stand-in network outputs: `collect` is used here for its features only."""
    del run, update, args
    return np.full((len(obs), N_ACTIONS), 1 / N_ACTIONS, np.float32), np.zeros(len(obs))


def _offline(run: EvalRun, played: Mapping[str, np.ndarray], family: Any) -> Dataset:
    """`collect` over the tree's own episodes, stored as the eval file it reads."""
    path = run.dir / "eval" / "stochastic" / "00000.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **played)
    return collect(run, 0, tuple(played["seeds"].tolist()), None, _uniform, family)


def _predict(tree: FittedTree, features: np.ndarray, columns: tuple[str, ...]) -> Any:
    return tree.model.predict(features[:, [columns.index(c) for c in tree.columns]])


def _expected(tree: FittedTree, obs: np.ndarray) -> Any:
    """The tree's actions on observations, through scikit-learn."""
    return _predict(tree, featurize(obs, "classic"), tuple(FEATURES["classic"]))


@pytest.fixture
def facing_run(tmp_path: Path) -> EvalRun:
    return toy_run(tmp_path / "facing", _FacingPolicy())


def test_a_tree_policy_acts_on_the_observation_of_a_state(facing_run: EvalRun) -> None:
    tree = _move_tree()
    policy = TreePolicy(tree, "classic")
    assert policy.name == "tree"
    acts = np.random.default_rng(0).integers(0, 5, 200)
    ep = replay_episode(facing_run, SEEDS[0], acts)
    rng = np.random.default_rng(0)
    got = [
        policy(jax.tree.map(lambda x, t=t: x[t], ep.states), rng)
        for t in range(0, 200, 10)
    ]
    expected = _expected(tree, ep.obs[::10])
    np.testing.assert_array_equal(got, expected)
    assert len(set(got)) > 1


def test_a_four_leaf_tree_plays_100_classic_steps(facing_run: EvalRun) -> None:
    tree = _move_tree()
    assert tree.leaves == 4
    r = play(game_for("classic"), TreePolicy(tree, "classic"), SEEDS[0], 100)
    assert (r.steps, r.policy) == (100, "tree")
    obs = replay_episode(facing_run, SEEDS[0], r.actions).obs
    np.testing.assert_array_equal(r.actions, _expected(tree, obs))


@pytest.fixture(scope="module")
def memory_play(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[EvalRun, FittedTree, dict[str, np.ndarray]]:
    """A tree over memory proxies played on two seeds through the eval rollout."""
    run = toy_run(tmp_path_factory.mktemp("memory") / "run", _FacingPolicy())
    tree = _memory_tree()
    return run, tree, tree_rollouts(tree, run, 0, SEEDS)


def test_memory_proxies_online_are_the_datasets_offline(
    memory_play: tuple[EvalRun, FittedTree, dict[str, np.ndarray]],
) -> None:
    run, tree, played = memory_play
    used = {tree.columns[i] for i in tree.model.tree_.feature if i >= 0}
    assert set(MEMORY_FEATURES) <= used
    ds = _offline(run, played, "ppo-rnn")
    since = ds.features[:, ds.columns.index("steps_since_unlock")]
    for i, seed in enumerate(SEEDS):
        rows, n = ds.episode == seed, played["length"][i]
        for name in MEMORY_FEATURES:
            online = played[f"extras/{name}"][i, :n]
            np.testing.assert_array_equal(
                ds.features[rows, ds.columns.index(name)], online
            )
        np.testing.assert_array_equal(
            _predict(tree, ds.features[rows], ds.columns), played["actions"][i, :n]
        )
    assert played["events/unlocked"].any()
    assert (since[ds.step > 0] == 0).any()  # unlocks reset the count, not only starts


def test_both_protocols_play_the_same_episode(
    memory_play: tuple[EvalRun, FittedTree, dict[str, np.ndarray]],
) -> None:
    _, tree, played = memory_play
    policy = TreePolicy(tree, "classic")
    for seed, i in ((SEEDS[0], 0), (SEEDS[1], 1)):  # timestep 0 resets the memory
        r = play(game_for("classic"), policy, seed, 300)
        assert r.steps == min(300, played["length"][i])
        np.testing.assert_array_equal(r.actions, played["actions"][i, : r.steps])


def test_memory_proxies_need_the_reward() -> None:
    policy = TreePolicy(_memory_tree(), "classic")
    obs = jnp.zeros((1, 1345))
    with pytest.raises(ValueError, match="reward"):
        policy.step(None, policy.init_carry(1), obs, jax.random.key(0), True)


def test_columns_the_policy_cannot_compute_are_refused() -> None:
    with pytest.raises(ValueError, match="top_level"):
        TreePolicy(_skill_tree(), "classic")
    model = DecisionTreeClassifier().fit(np.zeros((2, 1)), [0, 1])
    with pytest.raises(ValueError, match="mana"):
        TreePolicy(FittedTree("leaves", 2, model, ("mana",), 1.0), "classic")


def test_a_two_level_tree_keeps_the_top_levels_skills(tmp_path: Path) -> None:
    run = toy_run(tmp_path / "two", _TopLevel())
    tree = _skill_tree()
    played = tree_rollouts(tree, run, 0, SEEDS[:1])
    n = played["length"][0]
    t, acts = np.arange(n), played["actions"][0, :n]
    skill = played["extras/skill"][0, :n]
    np.testing.assert_array_equal(skill, np.where((t // SKILL_LEN) % 2, PRIM, LEARNED))
    np.testing.assert_array_equal(played["extras/steps_in_skill"][0, :n], t % SKILL_LEN)
    learned = skill == LEARNED
    assert (acts[~learned] == DO).all()  # a primitive skill keeps its fixed action
    ds = _offline(run, played, "skills")
    np.testing.assert_array_equal(
        acts[learned], _predict(tree, ds.features, ds.columns)[learned]
    )
    assert set(acts[learned]) <= {LEFT, RIGHT, UP, DOWN}
    # the top level gets the reward and done of the step that produced each observation
    ep = replay_episode(run, SEEDS[0], acts)
    assert (ep.reward != 0).any()
    np.testing.assert_allclose(
        played["extras/reward_in"][0, :n], np.r_[0, ep.reward[:-1]]
    )
    assert not played["extras/done_in"][0, :n].any()


def test_a_skill_methods_top_level_sees_the_trees_actions(tmp_path: Path) -> None:
    root = tmp_path / "smdp"
    root.mkdir()
    cfg = {"method": "smdp-skills", "env": CLASSIC, "envs": 16, "num_steps": 8}
    cfg |= {"minibatches": 2, "hidden": 16, "epochs": 1, "log2_rows": 12}
    cfg |= {"extra_obs_dim": 26}
    (root / "config.json").write_text(json.dumps(cfg))
    trainer = METHODS["smdp-skills"]
    state = trainer.init(
        cfg | {"steps": trainer.steps_per_update(cfg)}, jax.random.key(0)
    )
    save_checkpoint(root, 0, 0, 0.0, trainer.params(state))
    run = EvalRun.load(root)
    tree = _skill_tree()
    policy = TreePolicy(tree, "classic", top_level=run.policy)
    obs = run._reset(jnp.asarray(SEEDS[:1]))[0]
    zeros = jnp.zeros(1)
    act, carry = policy.step(
        run.params(0),
        policy.init_carry(1),
        obs,
        jax.random.key(0),
        True,
        zeros,
        zeros > 0,
    )
    assert int(carry.top[1][0]) == int(act[0])  # the skill network's last action input
    result = closed_loop(
        tree, run, 0, SEEDS[:1]
    )  # skill columns: the top level is used
    assert np.isfinite(result.tree).all()
    assert np.isfinite(result.network).all()


def test_a_tree_equal_to_the_network_has_closed_loop_fidelity_one(
    facing_run: EvalRun,
) -> None:
    result = closed_loop(_facing_tree(), facing_run, 0, SEEDS)
    np.testing.assert_array_equal(result.seeds, SEEDS)
    np.testing.assert_array_equal(result.tree, result.network)
    assert result.network.mean() > 0
    assert result.fidelity == 1.0
    other = closed_loop(_move_tree(), facing_run, 0, SEEDS, network=result.network)
    assert other.network is result.network
    assert other.fidelity == pytest.approx(other.tree.mean() / result.network.mean())
    assert not np.array_equal(other.tree, result.tree)


def test_float32_features_split_as_scikit_learn_splits_them() -> None:
    """Thresholds round down to float32, so values next to one go the same way."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3)).astype(np.float32)
    y = (x[:, 0] > x[:, 1]) + 2 * (x[:, 2] > 0.3)
    model = DecisionTreeClassifier(max_leaf_nodes=64, random_state=0).fit(x, y)
    reach = model.decision_path(x).tocsc()
    edges = []
    for node in np.flatnonzero(model.tree_.feature >= 0):
        f, near = model.tree_.feature[node], np.float32(model.tree_.threshold[node])
        row = x[reach[:, node].indices[0]]  # a row that reaches the split
        for v in (np.nextafter(near, -np.inf), near, np.nextafter(near, np.inf)):
            edges.append(row.copy())
            edges[-1][f] = v
    rows = np.concatenate([x, np.stack(edges)])
    got = _JaxTree.of(model)(jnp.asarray(rows))
    np.testing.assert_array_equal(got, model.predict(rows))
