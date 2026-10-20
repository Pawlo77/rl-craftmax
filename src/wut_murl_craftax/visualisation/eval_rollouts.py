"""Fixed-seed eval rollouts of every checkpoint, stored as replayable trajectories.

Writes ``<run>/eval/<update>.npz``: ``act, probs, value, reward, done, alive, ach, obs``
(seed on axis 0) and a per-step state ``digest``. Frames are never stored: replaying the
actions with the same seed rebuilds the states bit-exactly (``episode_frames``).
Usage: ``python -m wut_murl_craftax.visualisation.eval_rollouts RUN --seeds 0-15``
"""

import argparse
import functools
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..orchestration import configure_runtime
from ..training.checkpoints import CheckpointEntry, load_index, load_params
from ..training.ppo import ActorCritic, make_network, policy_value
from ..training.records import achievement_weights, is_classic
from .rollout import FEATURES, GameKind, Rollout, game_for, keys

logger = logging.getLogger(__name__)

EvalFn = Callable[..., dict[str, Any]]
"""Jitted ``(params, seed[, actions]) -> per-step arrays`` (``state`` included)."""


@dataclass(frozen=True, slots=True)
class EvalRun:
    """A training run opened for evaluation: config, checkpoints, network, env."""

    dir: Path
    cfg: dict[str, Any]
    index: list[CheckpointEntry]
    net: ActorCritic
    env: Any
    env_params: Any
    template: Any
    """Initialised params with the checkpoint structure."""

    @classmethod
    def load(cls, run_dir: Path) -> "EvalRun":
        """Open ``run_dir`` written by ``training.ppo``."""
        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / "config.json").read_text())
        net, env, env_params = make_network(cfg["env"], cfg["width"], auto_reset=False)
        obs = jnp.zeros((1, *env.observation_space(env_params).shape))
        template = net.init(jax.random.key(0), obs)
        return cls(run_dir, cfg, load_index(run_dir), net, env, env_params, template)

    @property
    def classic(self) -> bool:
        """Whether the run is on Craftax-Classic."""
        return is_classic(self.cfg["env"])

    @property
    def max_reward(self) -> float:
        """22 for Classic, 226 for Craftax."""
        return float(achievement_weights(self.cfg["env"]).sum())

    @property
    def updates(self) -> list[int]:
        """Checkpointed updates, ascending."""
        return [e["update"] for e in self.index]

    def env_steps(self, update: int) -> int:
        """Environment steps trained at a checkpoint."""
        return next(e["env_steps"] for e in self.index if e["update"] == update)

    def params(self, update: int) -> Any:
        """Params of a checkpoint."""
        return load_params(self.dir, update, self.template)


def make_eval_rollout(
    run: EvalRun, steps: int, greedy: bool = False, forced: bool = False
) -> EvalFn:
    """Rollout of one seed for ``steps``; ``forced`` replays given actions instead.

    ``state`` at ``t`` is the state the action at ``t`` was chosen in; after ``done``
    the env keeps stepping, masked by ``alive``.
    """
    env, pe, net = run.env, run.env_params, run.net

    def rollout(params: Any, seed: int, actions: jax.Array | None = None) -> Any:
        k_reset, k_env = keys(seed)  # same streams as rollout.replay
        k_act = jax.random.fold_in(k_reset, 2)
        obs, st = env.reset(k_reset, pe)

        def step(c: tuple[Any, ...], t: jax.Array) -> tuple[tuple[Any, ...], Any]:
            obs, st, alive = c
            logits, v = policy_value(net, params, obs[None])
            if forced:
                a = cast(jax.Array, actions)[t]
            elif greedy:
                a = jnp.argmax(logits[0])
            else:
                a = jax.random.categorical(jax.random.fold_in(k_act, t), logits[0])
            obs2, st2, r, d, _ = env.step(jax.random.fold_in(k_env, t), st, a, pe)
            rec = {
                "state": st,
                "obs": obs,
                "act": a,
                "probs": jax.nn.softmax(logits[0]),
            }
            rec |= {"value": v[0], "reward": r * alive, "done": d, "alive": alive}
            rec["ach"] = st2.achievements
            return (obs2, st2, alive & ~d), rec

        init = (obs, st, jnp.bool_(True))
        return jax.lax.scan(step, init, jnp.arange(steps))[1]

    return cast(EvalFn, jax.jit(rollout))


def state_digest(states: Any) -> np.ndarray:
    """Per-step checksum over every leaf of the stacked env states."""
    leaves = [np.asarray(x).reshape(x.shape[0], -1) for x in jax.tree.leaves(states)]
    flat = np.concatenate(leaves, 1).astype(np.float64)
    return flat @ np.cos(np.arange(flat.shape[1]))


def evaluate(
    run_dir: Path,
    seeds: Sequence[int] = tuple(range(16)),
    steps: int = 1000,
    updates: Sequence[int] | None = None,
    greedy: bool = False,
) -> Path:
    """Evaluate checkpoints (default all) on ``seeds``; return the eval directory."""
    run = EvalRun.load(run_dir)
    out = run.dir / ("eval_greedy" if greedy else "eval")
    out.mkdir(exist_ok=True)
    roll = jax.jit(jax.vmap(make_eval_rollout(run, steps, greedy), in_axes=(None, 0)))
    seed_arr = np.asarray(seeds)
    for u in updates or run.updates:
        t0 = time.time()
        tr = jax.tree.map(np.asarray, roll(run.params(u), seed_arr))
        states = tr.pop("state")
        digest = np.stack([
            state_digest(jax.tree.map(lambda x, i=i: x[i], states))
            for i in range(len(seed_arr))
        ])
        obs, probs = (
            tr.pop("obs").astype(np.float16),
            tr.pop("probs").astype(np.float16),
        )
        np.savez_compressed(
            out / f"{u:05d}.npz",
            seeds=seed_arr,
            digest=digest,
            obs=obs,
            probs=probs,
            **tr,
        )
        ret = tr["reward"].sum(1).mean()
        logger.info("update %d: %.1fs, mean return %.2f", u, time.time() - t0, ret)
    return out


def load_eval(
    run_dir: Path, update: int, greedy: bool = False
) -> dict[str, np.ndarray]:
    """Arrays of ``eval/<update>.npz``."""
    name = "eval_greedy" if greedy else "eval"
    with np.load(Path(run_dir) / name / f"{update:05d}.npz") as z:
        return {k: z[k] for k in z.files}


def game_kind(run: EvalRun) -> GameKind:
    """``classic`` or ``craftax``, as the gameplay ``rollout`` module names games."""
    return "classic" if run.classic else "craftax"


@functools.cache
def _batch_renderer(kind: GameKind, bps: int) -> Callable[[Any], jax.Array]:
    return cast(
        Callable[[Any], jax.Array], jax.jit(jax.vmap(game_for(kind).renderer(bps)))
    )


def _replayed_states(
    run: EvalRun, update: int, seed: int, greedy: bool
) -> tuple[Any, int, np.ndarray]:
    """States of steps ``0..n`` of a stored episode (``n`` = steps taken), actions."""
    ev = load_eval(run.dir, update, greedy)
    i = int(np.flatnonzero(ev["seeds"] == seed)[0])
    steps = ev["act"].shape[1]
    tr = make_eval_rollout(run, steps, forced=True)(
        run.params(update), seed, ev["act"][i]
    )
    n = min(int(ev["alive"][i].sum()), steps - 1)
    return jax.tree.map(lambda x: x[: n + 1], tr["state"]), n, ev["act"][i, :n]


def episode_frames(
    run: EvalRun, update: int, seed: int, bps: int = 16, greedy: bool = False
) -> np.ndarray:
    """Frames ``(n + 1, H, W, 3)`` of a stored episode at ``bps`` px per tile.

    Rebuilt by replaying the stored actions; the last frame shows the final state.
    """
    states, _, _ = _replayed_states(run, update, seed, greedy)
    render = _batch_renderer(game_kind(run), bps)
    n = jax.tree.leaves(states)[0].shape[0]
    chunks = [
        np.asarray(render(jax.tree.map(lambda x, i=i: x[i : i + 128], states)))
        for i in range(0, n, 128)
    ]
    return np.concatenate(chunks)


def as_rollout(run: EvalRun, update: int, seed: int, greedy: bool = False) -> Rollout:
    """The stored episode as a gameplay ``Rollout`` (highlights, masters, cuts)."""
    states, n, actions = _replayed_states(run, update, seed, greedy)
    game = game_for(game_kind(run))
    feats, ach = jax.tree.map(np.asarray, jax.vmap(game.features)(states))
    ev = load_eval(run.dir, update, greedy)
    i = int(np.flatnonzero(ev["seeds"] == seed)[0])
    label = f"PPO after {run.env_steps(update) / 1e6:.1f}M steps"
    return Rollout(
        game=game.kind,
        policy="ppo",
        label=label,
        seed=seed,
        key_scheme="fold_in",
        god=False,
        actions=actions.astype(np.int16),
        rewards=ev["reward"][i, :n],
        features=feats.astype(np.float32),
        feature_names=FEATURES,
        achievements=ach.astype(bool),
        achievement_names=game.achievement_names,
    )


def main(argv: list[str] | None = None) -> None:
    """Command line: ``RUN [--seeds 0-15] [--steps 1000] [--updates 0,9]``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--seeds", default="0-15", help="inclusive range lo-hi")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--updates", help="comma list; default every checkpoint")
    p.add_argument("--greedy", action="store_true")
    a = p.parse_args(argv)
    lo, hi = map(int, a.seeds.split("-"))
    updates = [int(u) for u in a.updates.split(",")] if a.updates else None
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_runtime()
    out = evaluate(a.run, range(lo, hi + 1), a.steps, updates, a.greedy)
    print(f"eval written to {out}")


if __name__ == "__main__":
    main()
