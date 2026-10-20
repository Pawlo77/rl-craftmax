"""Rollout record (seed, key scheme, actions, per-step features), policies, replay.

Step ``t`` uses ``fold_in(env_key, t)``: seed + actions replay an episode bit for bit;
``keys`` matches the training recorder.
CLI: ``python -m wut_murl_craftax.visualisation.rollout --game classic --seeds 1 2``.
"""

import argparse
import functools
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from craftax.craftax import constants as craftax_constants
from craftax.craftax.renderer import make_craftax_pixel_renderer as craftax_renderer
from craftax.craftax_classic import constants as classic_constants
from craftax.craftax_classic.renderer import (
    make_craftax_pixel_renderer as classic_renderer,
)
from craftax.craftax_env import make_craftax_env_from_name

logger = logging.getLogger(__name__)

type GameKind = Literal["craftax", "classic"]
type KeyScheme = Literal["fold_in", "split"]
type Frames = npt.NDArray[np.uint8]
type Floats = npt.NDArray[np.float32]

ENV_NAMES: dict[str, str] = {
    "craftax": "Craftax-Symbolic-v1",
    "classic": "Craftax-Classic-Symbolic-v1",
}
"""Symbolic envs: cheapest to step; frames come from the pixel renderer on demand."""
FEATURES = (
    "health",
    "food",
    "drink",
    "energy",
    "level",
    "sleeping",
    "melee_on_screen",
    "ranged_on_screen",
    "projectiles_on_screen",
    "hostile_close",
    "boss_progress",
    "row",
    "col",
)
"""Per-step feature columns recorded for highlight detection."""
LABELS = {
    "random": "random policy",
    "heuristic": "scripted agent",
    "network": "trained agent",
}
"""On-screen footage label per policy; god-mode episodes say ``god-mode dive``."""
DEFAULT_RUNS = Path("outputs/visualisation/runs")
"""Default rollout folder (git-ignored)."""
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
"""LEFT, RIGHT, UP, DOWN as (row, col) offsets (Craftax ``DIRECTIONS``)."""


def keys(seed: int, scheme: KeyScheme = "fold_in") -> tuple[jax.Array, jax.Array]:
    """(reset key, env-step base key); ``fold_in`` is the training recorder's scheme."""
    base = jax.random.PRNGKey(seed)
    if scheme == "fold_in":
        return base, jax.random.fold_in(base, 1)
    reset, step = jax.random.split(base)
    return reset, step


class Game:
    """One interface over Craftax and Craftax-Classic: env, renderer, features."""

    def __init__(self, kind: GameKind, god_mode: bool = False) -> None:
        """Build the no-auto-reset env (keeps the death frame) and jit its functions."""
        self.kind, self.god_mode = kind, god_mode
        self.env: Any = make_craftax_env_from_name(ENV_NAMES[kind], auto_reset=False)
        params = self.env.default_params
        self.params: Any = params.replace(god_mode=True) if god_mode else params
        self.constants: ModuleType = (
            craftax_constants if kind == "craftax" else classic_constants
        )
        enum = sorted(
            self.constants.Achievement, key=lambda a: a.value
        )  # enum order != value order
        self.achievement_names: tuple[str, ...] = tuple(a.name for a in enum)
        self.obs_dim: tuple[int, int] = tuple(self.constants.OBS_DIM)
        self.num_actions: int = int(self.env.num_actions)
        self.reset: Callable[..., Any] = jax.jit(self.env.reset)
        self.step: Callable[..., Any] = jax.jit(self.env.step)
        self.features: Callable[[Any], tuple[jax.Array, jax.Array]] = jax.jit(
            self._features
        )

    @functools.cache  # noqa: B019 - one Game per (kind, god) lives for the process
    def renderer(self, bps: int) -> Callable[[Any], jax.Array]:
        """Jittable ``state -> (H, W, 3) uint8`` at ``bps`` screen pixels per tile.

        The package caches textures for 7/10, 16 and 64 px; other sizes are built, added
        to the cached dict the renderer reads (80 px = Craftax 880x1040).
        """
        textures = self.constants.load_all_textures()
        if bps not in textures:
            textures[bps] = self.constants.load_all_textures_given_size(bps)
        draw = (
            classic_renderer(bps)
            if self.kind == "classic"
            else craftax_renderer(bps, do_night_noise=True)
        )
        return lambda s: draw(s).astype(jnp.uint8)

    def _mobs(self, s: Any) -> list[tuple[jax.Array, jax.Array]]:
        if self.kind == "craftax":
            lv = s.player_level
            return [
                (m.position[lv], m.mask[lv])
                for m in (s.melee_mobs, s.ranged_mobs, s.mob_projectiles)
            ]
        return [(m.position, m.mask) for m in (s.zombies, s.skeletons, s.arrows)]

    def _features(self, s: Any) -> tuple[jax.Array, jax.Array]:
        half = jnp.array(self.obs_dim) // 2
        counts, close = [], jnp.int32(0)
        for pos, mask in self._mobs(s):
            d = jnp.abs(pos - s.player_position)
            counts.append(jnp.sum(mask & (d <= half).all(-1)))
            close = close + jnp.sum(mask & (d.max(-1) <= 2))
        ck = self.kind == "craftax"
        f = jnp.stack([
            jnp.asarray(s.player_health, jnp.float32),
            s.player_food,
            s.player_drink,
            s.player_energy,
            s.player_level if ck else 0,
            s.is_sleeping,
            *counts,
            close,
            s.boss_progress if ck else 0,
            s.player_position[0],
            s.player_position[1],
        ]).astype(jnp.float32)
        return f, s.achievements

    def map2d(self, s: Any) -> npt.NDArray[np.int32]:
        """Block ids of the player's current floor."""
        return np.asarray(s.map[s.player_level] if self.kind == "craftax" else s.map)


@functools.cache
def game_for(kind: GameKind, god_mode: bool = False) -> Game:
    """Shared ``Game`` per (kind, god mode), so jit caches are reused."""
    return Game(kind, god_mode)


class Policy(Protocol):
    """Picks an action from the env state; trained agents use ``NetworkPolicy``."""

    name: str

    def __call__(self, state: Any, rng: np.random.Generator) -> int:
        """Action index for ``state``."""
        ...


class RandomPolicy:
    """Uniform random actions."""

    name = "random"

    def __init__(self, game: Game) -> None:
        """Remember the action count."""
        self.n = game.num_actions

    def __call__(self, state: Any, rng: np.random.Generator) -> int:  # noqa: ARG002
        """Uniform action."""
        return int(rng.integers(self.n))


class NetworkPolicy:
    """Greedy action of ``apply(params, obs)``; random linear weights by default."""

    name = "network"

    def __init__(
        self,
        game: Game,
        apply: Callable[[Any, jax.Array], jax.Array] | None = None,
        params: Any = None,
    ) -> None:
        """Jit ``argmax(apply(params, env.get_obs(state)))``."""
        obs_dim = game.env.observation_space(game.params).shape[0]
        if apply is None:
            params = (
                jax.random.normal(jax.random.PRNGKey(0), (obs_dim, game.num_actions))
                * 0.1
            )
            self._fwd = jax.jit(lambda p, s: jnp.argmax(game.env.get_obs(s) @ p))
        else:
            self._fwd = jax.jit(lambda p, s: jnp.argmax(apply(p, game.env.get_obs(s))))
        self.params = params

    def __call__(self, state: Any, rng: np.random.Generator) -> int:  # noqa: ARG002
        """Greedy action of the network."""
        return int(self._fwd(self.params, state))


class HeuristicPolicy:
    """Scripted agent: BFS to the nearest target; survive, climb the tech tree, descend.

    In god mode it also descends every 60 steps to the Graveyard (every floor on film).
    """

    name = "heuristic"
    WALK = frozenset({"GRASS", "SAND", "PATH", "FIRE_GRASS", "ICE_GRASS", "GRAVEL"})
    """Blocks the BFS may walk on."""

    def __init__(self, game: Game, descend_after: int = 600) -> None:
        """Precompute walkability; Craftax descends after ``descend_after`` steps."""
        self.g, self.B, self.A = game, game.constants.BlockType, game.constants.Action
        self.walkable = np.array([b.name in self.WALK for b in self.B])
        self.descend_after, self.last_pos, self.still = descend_after, (-1, -1), 0

    def _cells(self, s: Any, groups: list[Any]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for m in groups:
            pos, mask = np.asarray(m.position), np.asarray(m.mask)
            if self.g.kind == "craftax":
                pos, mask = pos[int(s.player_level)], mask[int(s.player_level)]
            out += [(int(p[0]), int(p[1])) for p, k in zip(pos, mask, strict=True) if k]
        return out

    def _bfs(
        self,
        grid: npt.NDArray[np.int32],
        start: tuple[int, int],
        goal: Callable[[tuple[int, int]], bool],
        blocked: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """(first move, path length) of the shortest walk to a cell next to a goal."""
        h, w = grid.shape
        prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
        queue = [start]
        for cur in queue:
            for a, (dr, dc) in MOVES.items():
                n = (cur[0] + dr, cur[1] + dc)
                if not (0 <= n[0] < h and 0 <= n[1] < w) or n in prev:
                    continue
                if goal(n):
                    path, c = [], cur
                    while (link := prev[c]) is not None:
                        c, step = link
                        path.append(step)
                    return (path[-1] if path else a), len(path)
                if self.walkable[grid[n]] and n not in blocked:
                    prev[n] = (cur, a)
                    queue.append(n)
        return None

    def _approach(
        self,
        s: Any,
        goal: Callable[[tuple[int, int]], bool],
        blocked: set[tuple[int, int]],
        act: int,
    ) -> int | None:
        """Walk to a goal cell and face it; ``act`` once adjacent and facing."""
        me = (int(s.player_position[0]), int(s.player_position[1]))
        found = self._bfs(self.g.map2d(s), me, goal, blocked)
        if found is None:
            return None
        a, dist = found
        dr, dc = MOVES[a]
        facing_goal = (
            dist == 0
            and goal((me[0] + dr, me[1] + dc))
            and int(s.player_direction) == a
        )
        return act if facing_goal else a

    def _block(
        self, s: Any, blocked: set[tuple[int, int]], *names: str, act: int | None = None
    ) -> int | None:
        grid = self.g.map2d(s)
        ids = {self.B[n].value for n in names}
        return self._approach(
            s,
            lambda n: int(grid[n]) in ids,
            blocked,
            self.A.DO.value if act is None else act,
        )

    def _survive(
        self,
        s: Any,
        hostile: list[tuple[int, int]],
        passive: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> int | None:
        me = (int(s.player_position[0]), int(s.player_position[1]))
        for a, (dr, dc) in MOVES.items():
            if (me[0] + dr, me[1] + dc) in hostile:
                return self.A.DO.value if int(s.player_direction) == a else a
        if int(s.player_energy) < 3:
            return int(self.A.SLEEP.value)
        if (
            int(s.player_drink) < 5
            and (act := self._block(s, blocked, "WATER")) is not None
        ):
            return act
        if int(s.player_food) < 5 and passive:
            cows = set(passive)
            return self._approach(
                s, lambda n: n in cows, blocked - cows, self.A.DO.value
            )
        return None

    def _tools(self, s: Any) -> tuple[bool, bool, bool, bool, bool]:
        inv = s.inventory
        if self.g.kind == "classic":
            return (
                bool(inv.wood_pickaxe),
                bool(inv.wood_sword),
                bool(inv.stone_pickaxe),
                bool(inv.stone_sword),
                bool(inv.iron_pickaxe),
            )
        pick, sword = int(inv.pickaxe), int(inv.sword)
        return pick >= 1, sword >= 1, pick >= 2, sword >= 2, pick >= 3

    def _craft(
        self, s: Any, blocked: set[tuple[int, int]], rng: np.random.Generator
    ) -> int | None:
        ac, grid, inv = self.A, self.g.map2d(s), s.inventory
        wood, stone, coal, iron = (
            int(inv.wood),
            int(inv.stone),
            int(inv.coal),
            int(inv.iron),
        )
        wp, ws, sp, ss, ip = self._tools(s)
        me = np.array([int(s.player_position[0]), int(s.player_position[1])])

        def near(name: str) -> bool:
            cells = np.argwhere(grid == self.B[name].value)
            return bool(len(cells) and (np.abs(cells - me).max(-1) <= 1).any())

        table_seen = bool((grid == self.B.CRAFTING_TABLE.value).any())
        if near("CRAFTING_TABLE"):
            for ok, act in (
                (not wp and wood >= 1, ac.MAKE_WOOD_PICKAXE),
                (not ws and wood >= 1, ac.MAKE_WOOD_SWORD),
                (wp and not sp and wood >= 1 and stone >= 1, ac.MAKE_STONE_PICKAXE),
                (sp and not ss and wood >= 1 and stone >= 1, ac.MAKE_STONE_SWORD),
                (
                    not ip and min(wood, stone, coal, iron) >= 1 and near("FURNACE"),
                    ac.MAKE_IRON_PICKAXE,
                ),
            ):
                if ok:
                    return int(act.value)
            if stone >= 1 and not (grid == self.B.FURNACE.value).any():
                return int(ac.PLACE_FURNACE.value)
        need_wood = 3 if not table_seen else (2 if not (wp and ws and sp and ss) else 0)
        if wood < need_wood:
            return self._block(s, blocked, "TREE") or int(rng.choice([1, 2, 3, 4]))
        if not table_seen:
            return int(ac.PLACE_TABLE.value)
        if not (wp and ws):
            return self._block(s, blocked, "CRAFTING_TABLE", act=ac.NOOP.value) or int(
                ac.NOOP.value
            )
        return None

    def _progress(
        self, s: Any, blocked: set[tuple[int, int]], rng: np.random.Generator
    ) -> int | None:
        ac, inv, ach = self.A, s.inventory, np.asarray(s.achievements)
        got = {
            n: bool(ach[self.g.constants.Achievement[n].value])
            for n in ("COLLECT_SAPLING", "PLACE_PLANT", "PLACE_STONE")
        }
        wp, _, sp, ss, ip = self._tools(s)
        if (
            not got["COLLECT_SAPLING"]
            and rng.random() < 0.5
            and (act := self._block(s, blocked, "GRASS")) is not None
        ):
            return act
        if int(inv.sapling) > 0 and not got["PLACE_PLANT"]:
            return int(ac.PLACE_PLANT.value)
        if (
            wp
            and int(inv.stone) < 3
            and (act := self._block(s, blocked, "STONE")) is not None
        ):
            return act
        if int(inv.stone) >= 2 and not got["PLACE_STONE"]:
            return int(ac.PLACE_STONE.value)
        if int(inv.stone) >= 1 and not (sp and ss):  # back to the table for stone tools
            act = self._block(s, blocked, "CRAFTING_TABLE", act=ac.NOOP.value)
            return act or int(ac.NOOP.value)
        if self.g.kind == "craftax" and int(s.timestep) > self.descend_after:
            target = tuple(
                int(x) for x in np.asarray(s.down_ladders)[int(s.player_level)]
            )
            if (int(s.player_position[0]), int(s.player_position[1])) == target:
                return int(ac.DESCEND.value)
            me = (int(s.player_position[0]), int(s.player_position[1]))
            if (
                found := self._bfs(self.g.map2d(s), me, lambda n: n == target, blocked)
            ) is not None:
                return found[
                    0
                ]  # step onto the ladder; DESCEND fires once standing on it
        for ok, name in (
            (sp and int(inv.coal) < 1, "COAL"),
            (sp and int(inv.iron) < 1, "IRON"),
            (ip, "DIAMOND"),
        ):
            if ok and (act := self._block(s, blocked, name)) is not None:
                return act
        return None

    def __call__(self, state: Any, rng: np.random.Generator) -> int:
        """Next action: dive (god mode), unstick, survive, craft, progress, wander."""
        s = state
        me = (int(s.player_position[0]), int(s.player_position[1]))
        self.still = self.still + 1 if me == self.last_pos else 0
        self.last_pos = me
        if self.g.god_mode and int(s.player_level) < 8 and int(s.timestep) % 60 == 59:
            return int(
                self.A.DESCEND.value
            )  # god mode descends anywhere (the player still takes damage)
        if self.still > 12 or rng.random() < 0.03:
            self.still = 0
            return int(rng.choice([1, 2, 3, 4]))
        if self.g.kind == "craftax":
            hostile, passive = (
                self._cells(s, [s.melee_mobs, s.ranged_mobs]),
                self._cells(s, [s.passive_mobs]),
            )
        else:
            hostile, passive = (
                self._cells(s, [s.zombies, s.skeletons]),
                self._cells(s, [s.cows]),
            )
        blocked = set(hostile + passive)
        steps: tuple[Callable[[], int | None], ...] = (
            lambda: self._survive(s, hostile, passive, blocked),
            lambda: self._craft(s, blocked, rng),
            lambda: self._progress(s, blocked, rng),
        )
        for step in steps:
            if (act := step()) is not None:
                return act
        return int(rng.choice([1, 2, 3, 4]))


POLICIES: dict[str, Callable[[Game], Policy]] = {
    "random": RandomPolicy,
    "heuristic": HeuristicPolicy,
    "network": NetworkPolicy,
}
"""Policy constructors by CLI name."""


@dataclass(frozen=True, slots=True, eq=False)
class Rollout:
    """One episode: enough to replay it exactly and to detect its highlights."""

    game: str
    policy: str
    label: str
    seed: int
    key_scheme: str
    god: bool
    actions: npt.NDArray[np.int16]
    rewards: Floats
    features: Floats
    """``(steps + 1, len(FEATURES))``: row t is the state after t steps (0 = reset)."""
    feature_names: tuple[str, ...]
    achievements: npt.NDArray[np.bool_]
    achievement_names: tuple[str, ...]

    @property
    def steps(self) -> int:
        """Number of env steps."""
        return len(self.actions)

    def __eq__(self, other: object) -> bool:
        """Field-wise equality, arrays compared by value."""
        if not isinstance(other, Rollout):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name))
            for f in fields(self)
        )

    __hash__ = None  # type: ignore[assignment]  # mutable arrays inside: not hashable

    def save(self, path: Path) -> Path:
        """Write a compressed ``.npz`` (8.5 KB for a 2,047-step Craftax episode)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            f.name: np.asarray(getattr(self, f.name)) for f in fields(self)
        }
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read ``save`` output (files without ``key_scheme``: early ``split`` runs)."""
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        policy = str(d["policy"])
        god = bool(d["god"])
        default_label = "god-mode dive" if god else LABELS.get(policy, policy)
        return cls(
            game=str(d["kind" if "kind" in d else "game"]),
            policy=policy,
            label=str(d.get("label", default_label)),
            seed=int(d["seed"]),
            key_scheme=str(d.get("key_scheme", "split")),
            god=god,
            actions=d["actions"].astype(np.int16),
            rewards=d["rewards"].astype(np.float32),
            features=d["features"].astype(np.float32),
            feature_names=tuple(map(str, d["feature_names"])),
            achievements=d["achievements"].astype(bool),
            achievement_names=tuple(map(str, d["achievement_names"])),
        )


def play(
    game: Game,
    policy: Policy,
    seed: int,
    max_steps: int = 10_000,
    key_scheme: KeyScheme = "fold_in",
    label: str | None = None,
) -> Rollout:
    """Run ``policy`` until death or ``max_steps``; record actions and features."""
    k_reset, k_step = keys(seed, key_scheme)
    rng = np.random.default_rng(seed)
    _, s = game.reset(k_reset, game.params)

    @jax.jit
    def step(
        t: int, s: Any, a: int
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array]:
        _, s, r, d, _ = game.step(jax.random.fold_in(k_step, t), s, a, game.params)
        f, ach = game.features(s)
        return s, r, d, f, ach

    f0, a0 = game.features(s)
    acts: list[int] = []
    rews: list[float] = []
    feats, achs = [np.asarray(f0)], [np.asarray(a0)]
    for t in range(max_steps):
        a = policy(s, rng)
        s, r, d, f, ach = step(t, s, a)
        acts.append(a)
        rews.append(float(r))
        feats.append(np.asarray(f))
        achs.append(np.asarray(ach))
        if bool(d):
            break
    name = getattr(policy, "name", type(policy).__name__)
    return Rollout(
        game=game.kind,
        policy=name,
        label=label or ("god-mode dive" if game.god_mode else LABELS.get(name, name)),
        seed=seed,
        key_scheme=key_scheme,
        god=game.god_mode,
        actions=np.array(acts, np.int16),
        rewards=np.array(rews, np.float32),
        features=np.stack(feats).astype(np.float32),
        feature_names=FEATURES,
        achievements=np.stack(achs).astype(bool),
        achievement_names=game.achievement_names,
    )


@dataclass(frozen=True, slots=True)
class ReplayChunk:
    """Consecutive replayed steps ``start .. start + len(features) - 1``."""

    start: int
    features: Floats
    frames: Frames | None


@functools.cache
def _replay_fns(
    game: Game, bps: int | None
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[[Any], Any]]:
    render = game.renderer(bps) if bps else (lambda _: jnp.zeros((), jnp.uint8))

    def step(k: jax.Array, s: Any, ta: tuple[jax.Array, jax.Array]) -> Any:
        _, s, *_ = game.step(jax.random.fold_in(k, ta[0]), s, ta[1], game.params)
        return s

    @jax.jit
    def skip(k: jax.Array, s: Any, ts: jax.Array, acts: jax.Array) -> Any:
        return jax.lax.scan(lambda s, ta: (step(k, s, ta), None), s, (ts, acts))[0]

    @jax.jit
    def run(k: jax.Array, s: Any, ts: jax.Array, acts: jax.Array) -> Any:
        def body(
            s: Any, ta: tuple[jax.Array, jax.Array]
        ) -> tuple[Any, tuple[jax.Array, jax.Array]]:
            s = step(k, s, ta)
            return s, (render(s), game.features(s)[0])

        return jax.lax.scan(body, s, (ts, acts))

    return skip, run, jax.jit(render)


def replay(
    rollout: Rollout,
    bps: int | None = None,
    start: int = 0,
    end: int | None = None,
    chunk: int = 256,
) -> Iterator[ReplayChunk]:
    """Re-run ``rollout``; stream steps ``start..end`` (frames at ``bps`` px per tile).

    Steps before ``start`` replay without rendering; the last chunk is padded so one
    compiled shape serves every chunk.
    """
    game = game_for(cast(GameKind, rollout.game), rollout.god)
    skip, run, render = _replay_fns(game, bps)
    k_reset, k_step = keys(rollout.seed, cast(KeyScheme, rollout.key_scheme))
    actions = rollout.actions.astype(np.int32)
    end = rollout.steps if end is None else min(end, rollout.steps)
    _, s = game.reset(k_reset, game.params)
    if start:
        s = skip(k_step, s, jnp.arange(start), jnp.asarray(actions[:start]))
    yield ReplayChunk(
        start,
        np.asarray(game.features(s)[0])[None],
        np.asarray(render(s))[None] if bps else None,
    )
    for i in range(start, end, chunk):
        n = min(chunk, end - i)
        s, (frames, f) = run(
            k_step,
            s,
            jnp.arange(i, i + chunk),
            jnp.asarray(np.pad(actions[i : i + n], (0, chunk - n))),
        )
        yield ReplayChunk(
            i + 1, np.asarray(f[:n]), np.asarray(frames[:n]) if bps else None
        )


def main(argv: list[str] | None = None) -> None:
    """Play episodes and save them as ``<game>_<policy>_<seed>.npz``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", choices=list(ENV_NAMES), default="classic")
    p.add_argument("--policy", choices=list(POLICIES), default="heuristic")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument(
        "--god",
        action="store_true",
        help="god mode: descend anywhere (game intro footage)",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RUNS)
    a = p.parse_args(argv)
    game = game_for(a.game, a.god)
    short = (
        "god"
        if a.god
        else {"heuristic": "heur", "random": "rand", "network": "net"}[a.policy]
    )
    for seed in a.seeds:
        r = play(game, POLICIES[a.policy](game), seed, a.max_steps)
        path = r.save(a.out_dir / f"{a.game}_{short}_{seed}.npz")
        got = [
            n for n, x in zip(r.achievement_names, r.achievements[-1], strict=True) if x
        ]
        print(
            f"{path}: {r.steps} steps, floor max {int(r.features[:, 4].max())},"
            f" {len(got)} achievements"
        )


if __name__ == "__main__":
    main()
