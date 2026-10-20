"""Training records on disk: per-update metrics, per-episode log, achievement tables.

Layout of a run directory (see ``knowledge/visualisation/Recording training data.md``):
``config.json``, ``metrics.npz`` (one row per update), ``episodes.npz`` (one row per
finished training episode, achievements bit-packed) and ``ckpt/``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from craftax.craftax import constants as craftax_constants
from craftax.craftax_classic import constants as classic_constants

BASE_COLUMNS = ("update", "env_steps", "wall_s", "n_episodes", "return", "length")
"""Leading columns of ``metrics.npz``; the losses and ``ach/<name>`` columns follow."""
LOSS_COLUMNS = ("policy_loss", "value_loss", "entropy", "approx_kl", "explained_var")
"""Loss columns, in the order the PPO learner returns them."""


def is_classic(env_name: str) -> bool:
    """Whether ``env_name`` is a Craftax-Classic variant."""
    return "Classic" in env_name


def achievement_names(env_name: str) -> list[str]:
    """Lower-case achievement names in enum order (22 Classic, 67 Craftax)."""
    consts = classic_constants if is_classic(env_name) else craftax_constants
    return [a.name.lower() for a in consts.Achievement]


def achievement_weights(env_name: str) -> np.ndarray:
    """Reward per achievement: 1 each in Classic, tier rewards (sum 226) in Craftax."""
    if is_classic(env_name):
        return np.ones(len(classic_constants.Achievement))
    return np.asarray(craftax_constants.ACHIEVEMENT_REWARD_MAP, float)


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

        ``ach`` holds the achievement flags of every step; only steps with ``done``
        count, since Craftax reports achievements on the final step only.
        """
        d = done.astype(bool)
        n_ep = int(d.sum())
        got = ach[d]
        self.episodes.append({
            "update": np.full(n_ep, update, np.int32),
            "ret": ret[d].astype(np.float32),
            "length": length[d].astype(np.int32),
            "ach": np.packbits(got, axis=-1),
        })
        nan = float("nan")
        rates = got.mean(0) * 100 if n_ep else np.full(len(self.names), nan)
        self.rows.append([
            update,
            env_steps,
            wall_s,
            n_ep,
            float(ret[d].mean()) if n_ep else nan,
            float(length[d].mean()) if n_ep else nan,
            *losses.tolist(),
            *rates.tolist(),
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
    """Reward per achievement."""

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
    return TrainingMetrics(env, cols, z["data"], names, achievement_weights(env))


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
