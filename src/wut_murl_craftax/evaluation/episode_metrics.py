"""Online score and achievement metrics of training episodes, read from step events.

``episode_step`` adds one transition to the running episode of every env inside the
rollout scan: return, length and the achievements unlocked so far, taken from
``StepEvents.unlocked`` of the pre-reset next state, so an unlock on the done step is
kept and no env state is compared across a done step. ``update_summary`` turns the
stacked per-step values of one update into the finished-episode means and success
rates; ``first_unlock_offsets`` finds when each achievement was first unlocked in it.
Definitions: "Scoreboard and basics" of the evaluation plan.
"""

from collections.abc import Mapping
from typing import Any, NamedTuple, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .events import StepEvents


@struct.dataclass
class EpisodeAcc:
    """Running episode of every env; leaves lead with the env axis."""

    ret: jax.Array
    """Return so far, float32."""
    length: jax.Array
    """Steps so far, int32."""
    ach: jax.Array
    """Achievements unlocked so far, bool (envs, achievements) by ``.value``."""


class UpdateSummary(NamedTuple):
    """Means over the episodes that finished in one update; NaN when none did."""

    n_episodes: int
    ret: float
    length: float
    success: np.ndarray
    """Per achievement, % of the finished episodes that unlocked it."""


def init_acc(envs: int, n_ach: int) -> EpisodeAcc:
    """Empty episodes for ``envs`` envs and ``n_ach`` achievements."""
    return EpisodeAcc(
        jnp.zeros(envs), jnp.zeros(envs, jnp.int32), jnp.zeros((envs, n_ach), bool)
    )


def events_of(info: Mapping[str, Any]) -> StepEvents:
    """The ``StepEvents`` of a wrapper step's ``info``.

    Raises:
        ValueError: Without ``info["events"]`` (Craftax's own auto-reset), where the
            achievements of a done step are not available: better no metrics than
            silent zeros.
    """
    events = info.get("events")
    if events is None:
        msg = (
            'step info has no "events": episode metrics need the optimistic-reset '
            "wrapper (resets='optimistic'); auto-reset runs cannot log them"
        )
        raise ValueError(msg)
    return events  # type: ignore[no-any-return]  # info is untyped


def episode_step(
    acc: EpisodeAcc, reward: jax.Array, done: jax.Array, unlocked: jax.Array
) -> tuple[EpisodeAcc, EpisodeAcc]:
    """Add one transition per env; return ``(next accumulators, totals)``.

    ``totals`` include this step and are the finished episode where ``done``; the
    next accumulators restart empty there.
    """
    total = EpisodeAcc(acc.ret + reward, acc.length + 1, acc.ach | unlocked)
    live = ~done
    nxt = EpisodeAcc(
        jnp.where(live, total.ret, 0.0),
        jnp.where(live, total.length, 0),
        total.ach & live[:, None],
    )
    return nxt, total


def first_unlock_offsets(unlocked: jax.Array) -> jax.Array:
    """Env steps into an update at which any env first unlocked each achievement.

    ``unlocked`` is ``(steps, envs, achievements)``; env steps count the whole batch
    (step ``t`` of ``envs`` envs ends at ``(t + 1) * envs``); -1 where none unlocked.
    """
    envs = unlocked.shape[1]
    any_env = unlocked.any(1)
    first = jnp.argmax(any_env, 0)
    return jnp.where(any_env.any(0), (first + 1) * envs, -1).astype(jnp.int32)


def update_summary(
    done: np.ndarray, ret: np.ndarray, length: np.ndarray, ach: np.ndarray
) -> UpdateSummary:
    """Finished-episode statistics of one update from its ``(steps, envs)`` arrays.

    ``ret``, ``length`` and ``ach`` (steps, envs, achievements) are read on done
    steps only, where `episode_step` holds the whole episode.
    """
    d = np.asarray(done, bool)
    n = int(d.sum())
    got = np.asarray(ach)[d]
    nan = float("nan")
    if not n:
        return UpdateSummary(0, nan, nan, np.full(got.shape[-1], nan))
    ret_mean = float(np.asarray(ret)[d].mean())
    return UpdateSummary(
        n, ret_mean, float(np.asarray(length)[d].mean()), got.mean(0) * 100
    )


@overload
def score_pct(ret: float, max_reward: float) -> float: ...
@overload
def score_pct(ret: np.ndarray, max_reward: float) -> np.ndarray: ...
def score_pct(ret: Any, max_reward: float) -> Any:
    """Return as % of the maximum reward (226 on Craftax, 22 on Classic)."""
    return ret / max_reward * 100
