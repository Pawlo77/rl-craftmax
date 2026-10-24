"""Decision trees played as policies, and their closed-loop fidelity.

`TreePolicy` plays a `FittedTree` through the trainer `Policy` protocol (``step``,
jitted by evaluation) and the gameplay one (``__call__(state, rng)``, `rollout.play`).
It computes the tree's columns online: `featurize` of the observation, the memory
proxies `dataset.collect` adds offline, and for two-level policies the running skill
of the top level, which keeps choosing and ending skills (the tree replaces only the
skill network). `closed_loop` plays the tree greedily with the eval keys and compares
its score with the network's. Plan: [[2026-10-20-craftax-interpretation-and-ruleset]].
"""

import dataclasses
import functools
import math
from collections.abc import Sequence
from typing import Any, NamedTuple, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..evaluation.episode_metrics import score_pct
from ..training.interface import Policy
from ..training.skills import NUM_SKILLS, PRIMITIVE
from ..training.smdp import SMDPPolicy
from ..visualisation.eval_rollouts import (
    EvalRun,
    eval_returns,
    game_kind,
    rollout_seeds,
)
from ..visualisation.rollout import GameKind, game_for
from .dataset import MEMORY_FEATURES, SKILL_FEATURES, TEST_SEEDS
from .distill import FittedTree
from .features import FEATURES, featurize

CLOSED_LOOP_LEAVES = 256
"""The leaf-grid size played in the closed loop: one tree per policy (preflight I4)."""
HEALTH_REWARD = 0.1
"""Reward per health point; the rest of a step's reward is unlock weights."""
_UNLOCK = 0.5
"""Smallest unlock weight is 1: a step whose reward minus the health part exceeds
this unlocked something (float32 rounding is far below it)."""


class Memory(NamedTuple):
    """Per env: the memory proxies of the step just taken and its action and health."""

    prev_action: jax.Array
    """The ``prev_action`` column the tree read (-1 at an episode start)."""
    since_unlock: jax.Array
    """The ``steps_since_unlock`` column the tree read."""
    action: jax.Array
    """The action played: the next step's ``prev_action``."""
    health: jax.Array
    """Health of the observation: the next step's unlock test."""
    started: jax.Array


class TreeCarry(NamedTuple):
    """`TreePolicy` carry: the memory proxies and the top level's carry (or ())."""

    memory: Memory
    top: Any


class _JaxTree(NamedTuple):
    """A fitted scikit-learn tree as arrays, evaluated in JAX."""

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    action: np.ndarray
    """The predicted class of each node (used at leaves)."""
    depth: int

    @classmethod
    def of(cls, model: DecisionTreeClassifier) -> Self:
        """The arrays of ``model``.

        scikit-learn tests ``x <= threshold`` in float64; for float32 ``x`` that is
        ``x <= t``, with ``t`` the largest float32 not above the threshold.
        """
        t = model.tree_
        thr = t.threshold.astype(np.float32)
        above = thr.astype(np.float64) > t.threshold
        thr = np.where(above, np.nextafter(thr, np.float32(-np.inf)), thr)
        action = model.classes_[t.value[:, 0].argmax(1)].astype(np.int32)
        feature = np.maximum(t.feature, 0)  # leaves hold -2; never read
        return cls(t.children_left, t.children_right, feature, thr, action, t.max_depth)

    def __call__(self, x: jax.Array) -> jax.Array:
        """The action of each row of ``x`` (columns in the model's order)."""
        rows = jnp.arange(x.shape[0])
        left, right, feature, thr = map(
            jnp.asarray, (self.left, self.right, self.feature, self.threshold)
        )

        def down(_: Any, node: jax.Array) -> jax.Array:
            nxt = jnp.where(
                x[rows, feature[node]] <= thr[node], left[node], right[node]
            )
            return jnp.where(left[node] < 0, node, nxt)

        start = jnp.zeros(x.shape[0], jnp.int32)
        return jnp.asarray(self.action)[jax.lax.fori_loop(0, self.depth, down, start)]


class TreePolicy:
    """A fitted tree acting deterministically through both policy protocols."""

    name = "tree"

    def __init__(
        self, tree: FittedTree, game: GameKind, top_level: Policy | None = None
    ) -> None:
        """Index the tree's columns among those the policy computes.

        ``top_level`` is a two-level policy with ``extras`` (`SMDPPolicy`); it picks
        and ends skills and supplies the skill columns.

        Raises:
            ValueError: If the tree reads a column the policy cannot compute.
        """
        skills = SKILL_FEATURES if top_level is not None else ()
        available = (*FEATURES[game], *MEMORY_FEATURES, *skills)
        missing = [c for c in tree.columns if c not in available]
        if missing:
            hint = (
                " (skill columns need a top_level)"
                if set(missing) & set(SKILL_FEATURES)
                else ""
            )
            msg = f"a {game} tree policy cannot compute {missing}{hint}"
            raise ValueError(msg)
        self.tree, self.game, self.top_level = tree, game, top_level
        self._columns = np.array([available.index(c) for c in tree.columns])
        self._health = FEATURES[game].index("health")
        self._needs_reward = "steps_since_unlock" in tree.columns
        self._model = _JaxTree.of(tree.model)
        self._state: tuple[TreeCarry, jax.Array] | None = None
        """``__call__``'s carry and the achievements of the last state."""

    def init_carry(self, n: int) -> TreeCarry:
        """No previous action, no health yet, the top level's carry; ``n`` envs."""
        none = jnp.full(n, -1, jnp.int32)
        zero = jnp.zeros(n, jnp.int32)
        mem = Memory(none, zero, none, jnp.zeros(n), jnp.zeros(n, bool))
        top = () if self.top_level is None else self.top_level.init_carry(n)
        return TreeCarry(mem, top)

    def _advance(
        self,
        mem: Memory,
        feats: jax.Array,
        unlocked: jax.Array,
        new: jax.Array,
        skill: list[jax.Array],
    ) -> tuple[jax.Array, Memory]:
        """The tree's action and the memory of this step (as `dataset.collect`)."""
        prev = jnp.where(new, -1, mem.action)
        since = jnp.where(new | unlocked, 0, mem.since_unlock + 1)
        row = jnp.concatenate(
            [feats, prev[:, None], since[:, None], *skill], 1, dtype=jnp.float32
        )
        action = self._model(row[:, self._columns])
        health = feats[:, self._health]
        return action, Memory(prev, since, action, health, jnp.ones_like(new))

    def step(
        self,
        params: Any,
        carry: TreeCarry,
        obs: jax.Array,
        key: jax.Array,
        greedy: bool,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, TreeCarry]:
        """The tree's actions for a batch of observations and the next carry.

        ``reward`` and ``done`` belong to the step that produced ``obs``. An unlock is
        the reward the health change does not explain, so ``steps_since_unlock``
        needs ``reward``. The top level gets ``params``, ``key``, ``greedy``,
        ``reward`` and ``done``; primitive skills keep their fixed action.

        Raises:
            ValueError: If the tree reads ``steps_since_unlock`` without ``reward``.
        """
        if reward is None and self._needs_reward:
            msg = "steps_since_unlock needs the reward of each step"
            raise ValueError(msg)
        feats = cast(jax.Array, featurize(obs, self.game, jnp))
        mem, top = carry
        new = ~mem.started if done is None else ~mem.started | done
        r = jnp.zeros(obs.shape[0]) if reward is None else reward
        unlocked = r - HEALTH_REWARD * (feats[:, self._health] - mem.health) > _UNLOCK
        if self.top_level is None:
            action, mem = self._advance(mem, feats, unlocked, new, [])
            return action, TreeCarry(mem, top)
        fixed, top = self.top_level.step(params, top, obs, key, greedy, reward, done)
        extras = cast(Any, self.top_level).extras(top)
        skill = extras["skill"].astype(jnp.int32)
        cols = [jax.nn.one_hot(skill, NUM_SKILLS), extras["steps_in_skill"][:, None]]
        action, mem = self._advance(mem, feats, unlocked, new, cols)
        action = jnp.where(jnp.asarray(PRIMITIVE)[skill], fixed, action)
        if isinstance(self.top_level, SMDPPolicy):
            top = (top[0], action, *top[2:])  # its next step reads the played action
        return action, TreeCarry(mem._replace(action=action), top)

    def extras(self, carry: TreeCarry) -> dict[str, jax.Array]:
        """Per step for the eval file: memory proxies read, the top level's extras."""
        mem = carry.memory
        out = {"prev_action": mem.prev_action.astype(jnp.int8)}
        out["steps_since_unlock"] = mem.since_unlock
        if self.top_level is not None:
            out |= cast(Any, self.top_level).extras(carry.top)
        return out

    @functools.cached_property
    def _from_state(self) -> Any:
        """Jitted ``(carry, achievements, state) -> (action, carry, achievements)``."""
        env = game_for(self.game).env

        def act(carry: TreeCarry, ach: jax.Array, state: Any) -> Any:
            feats = cast(jax.Array, featurize(env.get_obs(state)[None], self.game, jnp))
            new = ~carry.memory.started | (state.timestep == 0)
            unlocked = jnp.any(state.achievements & ~ach)[None]
            action, mem = self._advance(carry.memory, feats, unlocked, new, [])
            return action[0], TreeCarry(mem, ()), state.achievements

        return jax.jit(act)

    def __call__(self, state: Any, rng: np.random.Generator) -> int:  # noqa: ARG002
        """The tree's action for ``state``; a state at timestep 0 starts an episode.

        Raises:
            ValueError: With a top level, which needs params: play through `step`.
        """
        if self.top_level is not None:
            msg = "a two-level tree plays through step: its top level needs params"
            raise ValueError(msg)
        carry, ach = self._state or (self.init_carry(1), state.achievements)
        action, carry, ach = self._from_state(carry, ach, state)
        self._state = (carry, ach)
        return int(action)


def _rollouts(
    tree: FittedTree, run: EvalRun, params: Any, seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    top = run.policy if hasattr(run.policy, "extras") else None
    policy = TreePolicy(tree, game_kind(run), top_level=top)
    played = rollout_seeds(dataclasses.replace(run, policy=policy), params, seeds, True)
    return cast(dict[str, np.ndarray], played)  # no should_stop: never None


def tree_rollouts(
    tree: FittedTree, run: EvalRun, update: int, seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    """Eval-file arrays (`rollout_seeds`) of ``tree`` played greedily on ``seeds``.

    A run whose policy has ``extras`` is two-level: that policy, with the params of
    ``update``, is the tree's top level.
    """
    return _rollouts(tree, run, run.params(update), seeds)


class ClosedLoop(NamedTuple):
    """Greedy scores (% of the maximum reward) per seed of a tree and its network."""

    seeds: np.ndarray
    tree: np.ndarray
    network: np.ndarray

    @property
    def fidelity(self) -> float:
        """Mean tree score over mean network score; NaN if the network's is <= 0."""
        net = float(self.network.mean())
        return float(self.tree.mean()) / net if net > 0 else math.nan


def _scores(run: EvalRun, arrays: dict[str, np.ndarray] | None) -> np.ndarray:
    ret = eval_returns(cast(dict[str, np.ndarray], arrays), run.cfg["env"])
    return score_pct(ret, run.max_reward)


def closed_loop(
    tree: FittedTree,
    run: EvalRun,
    update: int,
    seeds: Sequence[int] = TEST_SEEDS,
    network: np.ndarray | None = None,
) -> ClosedLoop:
    """``tree`` and the network of ``update`` played greedily on ``seeds``.

    Both use the eval keys of each seed ([[Recording training data]]). ``network``
    reuses the network's scores of an earlier call on the same seeds.
    """
    params = run.params(update)
    if network is None:
        network = _scores(run, rollout_seeds(run, params, seeds, greedy=True))
    played = _rollouts(tree, run, params, seeds)
    return ClosedLoop(np.asarray(seeds), _scores(run, played), network)
