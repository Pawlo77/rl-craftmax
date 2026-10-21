"""Training records on disk: per-update metrics, per-episode log, achievement tables.

Layout of a run directory (see ``knowledge/visualisation/Recording training data.md``):
``config.json``, ``metrics.npz`` (one row per update), ``episodes.npz`` (one row per
finished training episode, achievements bit-packed) and ``ckpt/``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.episode_metrics import update_summary
from ..evaluation.milestones import achievement_names as _names
from ..evaluation.milestones import constants, game_of

BASE_COLUMNS = ("update", "env_steps", "wall_s", "n_episodes", "return", "length")
"""Leading columns of ``metrics.npz``; the losses and ``ach/<name>`` columns follow."""
LOSS_COLUMNS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "explained_var",
    "clip_frac",
    "grad_norm",
)
"""Loss columns, in the order the PPO learner returns them."""


def is_classic(env_name: str) -> bool:
    """Whether ``env_name`` is a Craftax-Classic variant."""
    return game_of(env_name) == "classic"


def achievement_names(env_name: str) -> list[str]:
    """Lower-case achievement names in ``.value`` order (22 Classic, 67 Craftax)."""
    return list(_names(game_of(env_name)))


def achievement_weights(env_name: str) -> np.ndarray:
    """Reward per achievement in ``.value`` order: 1 Classic, tiers (sum 226) Craftax.

    The constants are imported on first use: they build JAX arrays, which starts the
    backends; deferring it keeps ``configure_runtime`` able to split the CPU.
    """
    consts = constants(game_of(env_name))
    if is_classic(env_name):
        return np.ones(len(consts.Achievement))
    return np.asarray(consts.ACHIEVEMENT_REWARD_MAP, float)


class RunLog:
    """Per-update rows and finished episodes; ``flush`` rewrites both files."""

    def __init__(self, out: Path, env_name: str, names: list[str]) -> None:
        """Start an empty log for a run written to ``out``."""
        self.out, self.env_name, self.names = out, env_name, names
        self.rows: list[list[float]] = []
        self.episodes: list[dict[str, np.ndarray]] = []

    def add_update(
        self,
        update: int,
        env_steps: int,
        wall_s: float,
        losses: np.ndarray,
        done: np.ndarray,
        ach: np.ndarray,
        ret: np.ndarray,
        length: np.ndarray,
    ) -> None:
        """Record one update from its ``(steps, envs)`` rollout arrays.

        ``ret``, ``length`` and ``ach`` hold the running episode at every step; only
        steps with ``done`` count, where they are the finished episode.
        """
        d = done.astype(bool)
        got = ach[d]
        s = update_summary(d, ret, length, ach)
        self.episodes.append({
            "update": np.full(s.n_episodes, update, np.int32),
            "ret": ret[d].astype(np.float32),
            "length": length[d].astype(np.int32),
            "ach": np.packbits(got, axis=-1),
        })
        self.rows.append([
            update,
            env_steps,
            wall_s,
            s.n_episodes,
            s.ret,
            s.length,
            *losses.tolist(),
            *s.success.tolist(),
        ])

    def flush(self) -> None:
        """Write ``metrics.npz`` and ``episodes.npz`` with everything logged so far."""
        cols = [*BASE_COLUMNS, *LOSS_COLUMNS, *(f"ach/{n}" for n in self.names)]
        np.savez_compressed(
            self.out / "metrics.npz",
            data=np.array(self.rows, np.float64),
            cols=np.array(cols),
            env=np.array(self.env_name),
        )
        eps = {
            k: np.concatenate([e[k] for e in self.episodes]) for k in self.episodes[0]
        }
        arrays: dict[str, Any] = {**eps, "names": np.array(self.names)}
        np.savez_compressed(self.out / "episodes.npz", **arrays)


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Per-update metrics of a run, with the score as % of the maximum reward."""

    env: str
    cols: list[str]
    data: np.ndarray
    names: list[str]
    weights: np.ndarray
    """Reward per achievement, aligned with ``names``."""

    def col(self, name: str) -> np.ndarray:
        """One column by name."""
        return self.data[:, self.cols.index(name)]

    @property
    def steps(self) -> np.ndarray:
        """Environment steps at the end of each update."""
        return self.col("env_steps")

    @property
    def rates(self) -> np.ndarray:
        """Success rate per achievement, % of episodes, (updates, achievements)."""
        return np.stack([self.col(f"ach/{n}") for n in self.names], 1)

    @property
    def max_reward(self) -> float:
        """22 for Classic, 226 for Craftax."""
        return float(self.weights.sum())

    @property
    def score(self) -> np.ndarray:
        """Episode return (achievements and health term) as % of the maximum reward."""
        return self.col("return") / self.max_reward * 100


def load_metrics(run_dir: Path) -> TrainingMetrics:
    """Read ``metrics.npz`` of a run."""
    z = np.load(Path(run_dir) / "metrics.npz")
    env = str(z["env"])
    cols = [str(c) for c in z["cols"]]
    names = [c.removeprefix("ach/") for c in cols if c.startswith("ach/")]
    # paired by name: files written before the switch to value order hold enum order
    reward = dict(zip(achievement_names(env), achievement_weights(env), strict=True))
    weights = np.array([reward[n] for n in names])
    return TrainingMetrics(env, cols, z["data"], names, weights)


@dataclass(frozen=True, slots=True)
class Episodes:
    """Finished training episodes, one row each."""

    update: np.ndarray
    ret: np.ndarray
    length: np.ndarray
    ach: np.ndarray
    """Achievement flags, bool, shape (episodes, achievements)."""
    names: list[str]


def load_episodes(run_dir: Path) -> Episodes:
    """Read ``episodes.npz`` and unpack the achievement bits."""
    z = np.load(Path(run_dir) / "episodes.npz")
    names = [str(n) for n in z["names"]]
    ach = np.unpackbits(z["ach"], axis=-1, count=len(names)).astype(bool)
    return Episodes(z["update"], z["ret"], z["length"], ach, names)


def first_unlocks(ep: Episodes) -> np.ndarray:
    """First update in which any episode earned each achievement; -1 if never."""
    hit = ep.ach.any(0)
    first_row = ep.ach.argmax(0)
    return np.where(hit, ep.update[first_row], -1)
