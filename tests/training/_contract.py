"""Contract check every trainer's test calls: a done-step unlock is in its metrics."""

from typing import Any

import jax
import numpy as np
import pytest
from craftax.craftax_classic.constants import Achievement

from wut_murl_craftax.evaluation.episode_metrics import ONLINE_LAYOUT
from wut_murl_craftax.training import wrappers

MARKER = Achievement.PLACE_TABLE.value


def assert_done_step_unlock_counted(
    trainer: Any,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tries: int = 40,
) -> None:
    """Fail unless finished episodes carry the unlock and death of their done step.

    The wrapper's step events are patched to unlock ``PLACE_TABLE`` exactly on the
    step the player dies (health <= 0, which ends a Classic episode), so only a
    trainer that takes ``ach`` from ``StepEvents.unlocked`` at the done step sees it.
    ``config`` must be a Craftax-Classic config no other test has compiled (change a
    size, e.g. ``width``): trainers cache their compiled steps per config, which
    would skip the patch or keep it afterwards.
    """
    real, calls = wrappers.step_events, []

    def patched(game: Any, s: Any, action: Any, s2: Any, **kw: Any) -> Any:
        calls.append(game)
        events = real(game, s, action, s2, **kw)
        dying = s2.player_health <= 0
        return events.replace(unlocked=events.unlocked.at[MARKER].set(dying))

    monkeypatch.setattr(wrappers, "step_events", patched)
    state = trainer.init(config, jax.random.key(0))
    for _ in range(tries):
        state, metrics = trainer.train_chunk(state, 1)
        done = np.asarray(metrics["done"], bool)
        if done.any():
            break
    assert calls, "the patch was never traced: use a config this process has not run"
    assert done.any(), f"no episode finished in {tries} updates; use a longer rollout"
    ach = np.asarray(metrics["ach"])[done]
    assert ach[:, MARKER].all(), "a done-step unlock is missing from a finished episode"
    online = np.asarray(metrics["online"])[0]
    episodes = online[ONLINE_LAYOUT["episodes"]][0]
    assert episodes == done.sum(), "the online sums count other episodes"
    # no Classic episode times out this early: each one ended in the death step
    assert online[ONLINE_LAYOUT["death"]][0] == episodes, "a done-step death is lost"
    assert np.asarray(metrics["first_unlock"])[0, MARKER] >= 0
