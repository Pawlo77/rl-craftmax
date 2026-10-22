"""Tests for the online score and achievement metrics of training episodes."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Achievement, Action
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.episode_metrics import (
    ONLINE_LAYOUT,
    close_update,
    episode_step,
    events_of,
    first_unlock_offsets,
    init_acc,
    metric_keys,
    new_first_unlocks,
    score_pct,
    update_metrics,
    update_summary,
)
from wut_murl_craftax.evaluation.events import LAVA, NO_SOURCE, StepEvents
from wut_murl_craftax.evaluation.milestones import CREATURE_OFFSET, CREATURES
from wut_murl_craftax.training.ppo import PPOConfig, init_state
from wut_murl_craftax.training.ppo import train_chunk as ppo_train_chunk
from wut_murl_craftax.training.records import achievement_weights
from wut_murl_craftax.training.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)

CLASSIC, CRAFTAX = "Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1"
ENVS, EPISODE = 16, 200
TABLE = Achievement.PLACE_TABLE.value
WOOD, STONE = Achievement.COLLECT_WOOD.value, Achievement.PLACE_STONE.value
ZOMBIE, ARROW = CREATURE_OFFSET["melee"], CREATURE_OFFSET["projectile"]
KINDS = len(CREATURES)
F16 = jnp.float16


def _events(envs: int, n_ach: int = 22, **fields: Any) -> StepEvents:
    """Step events of ``envs`` envs where nothing happens, except ``fields``."""

    def z(*shape: int, dtype: Any = bool) -> jax.Array:
        return jnp.zeros((envs, *shape), dtype)

    quiet = {
        "unlocked": z(n_ach), "floor": z(dtype=jnp.int8),
        "health_delta": z(dtype=jnp.float32), "health_loss": z(dtype=F16),
        "drain": z(), "combat_damage": z(dtype=F16),
        "damage_source": jnp.full(envs, NO_SOURCE, jnp.int8),
        "kills": z(len(CREATURES), dtype=jnp.int8), "passive_kills": z(dtype=jnp.int8),
        "damage_dealt": z(dtype=F16), "projectile_damage": z(dtype=F16),
        "ate": z(), "drank": z(), "slept": z(), "stat_low": z(3), "stat_zero": z(3),
        "forced": z(), "effective": z(), "hostiles_on_screen": z(dtype=jnp.int8),
        "hostiles_close": z(dtype=jnp.int8),
    }  # fmt: skip
    given = {k: jnp.asarray(v, quiet[k].dtype) for k, v in fields.items()}
    return StepEvents(**quiet | given)


def _hit(envs: int, env: int, loss: float, source: int, **more: Any) -> StepEvents:
    """Events where ``env`` loses ``loss`` health to ``source`` (and ``more``)."""
    col = np.zeros(envs, np.float32)
    col[env] = loss
    src = np.full(envs, NO_SOURCE)
    src[env] = source
    return _events(
        envs, health_delta=-col, health_loss=col, combat_damage=col,
        damage_source=src, **more,
    )  # fmt: skip


def _flag(envs: int, env: int, width: int, at: int) -> np.ndarray:
    """``(envs, width)`` bools with ``[env, at]`` set."""
    out = np.zeros((envs, width), bool)
    out[env, at] = True
    return out


def _play(
    acc: Any, events: list[StepEvents], done: list[list[bool]], act: Any = None
) -> tuple[Any, list[Any]]:
    """Feed ``events[t]`` and ``done[t]`` step by step; the totals of every step."""
    step, totals = jax.jit(episode_step), []
    envs = len(done[0])
    for t, ev in enumerate(events):
        a = jnp.zeros(envs, jnp.int32) if act is None else jnp.asarray(act[t])
        r = ev.unlocked.sum(-1) + 0.1 * ev.health_delta
        acc, total = step(acc, r, jnp.asarray(done[t]), ev, a)
        totals.append(total)
    return acc, totals


def _metrics(out: dict[str, Any], totals: list[Any], done: Any, game: Any) -> Any:
    """`update_metrics` of a played update closed into ``out``."""
    keys = ("ret", "length", "ach")
    stack = [np.stack([np.asarray(getattr(t, k)) for t in totals]) for k in keys]
    summary = update_summary(np.asarray(done, bool), *stack)
    return update_metrics(np.asarray(out["online"]), summary, game)


@pytest.fixture(scope="module")
def scripted() -> dict[str, Any]:
    """200 steps of 16 Classic envs through the optimistic-reset wrapper.

    Nothing spawns, so every episode lasts exactly 200 steps. Env 1 places a table at
    step 100; env 0 places one on step 200, the done step, where the wrapper has
    already replaced its state with a fresh episode. The others do nothing.
    """
    env = LogWrapper(make_craftax_env_from_name(CLASSIC, auto_reset=False))
    vec = OptimisticResetVecEnvWrapper(env, ENVS, 16)
    params = vec.default_params.replace(
        max_timesteps=EPISODE,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    _, st = vec.reset(jax.random.key(0), params)
    s = st.env_state
    wood = s.inventory.wood.at[:2].set(5)
    s = s.replace(
        inventory=s.inventory.replace(wood=wood),
        zombies=s.zombies.replace(mask=jnp.zeros_like(s.zombies.mask)),
        skeletons=s.skeletons.replace(mask=jnp.zeros_like(s.skeletons.mask)),
    )
    st = st.replace(env_state=s)
    acc, step = init_acc(ENVS, 22), jax.jit(episode_step)
    out: dict[str, list[Any]] = {k: [] for k in ("done", "ret", "len", "ach", "reward")}
    for t in range(1, EPISODE + 1):
        act = np.zeros(ENVS, np.int32)
        act[1] = Action.PLACE_TABLE.value if t == 100 else 0
        act[0] = Action.PLACE_TABLE.value if t == EPISODE else 0
        _, st, r, d, info = vec.step(jax.random.key(t), st, jnp.asarray(act), params)
        acc, end = step(acc, r, d, events_of(info), jnp.asarray(act))
        for k, v in zip(out, (d, end.ret, end.length, end.ach, r), strict=True):
            out[k].append(np.asarray(v))
    stacked = {k: np.stack(v) for k, v in out.items()}
    stacked["logged"] = np.asarray(info["returned_episode_returns"])
    stacked["fresh"] = np.asarray(st.env_state.achievements)
    return stacked


def test_the_unlock_on_the_done_step_is_counted(scripted: dict[str, Any]) -> None:
    done, ach = scripted["done"], scripted["ach"]
    assert np.flatnonzero(done.any(1)).tolist() == [EPISODE - 1]
    assert ach[-1, 0, TABLE], "env 0 unlocked on its last step"
    assert ach[-1, 1, TABLE], "env 1 unlocked 100 steps earlier and keeps it"
    assert ach[-1].sum() == 2
    # what a comparison of states across the done step sees: the fresh episode
    assert not scripted["fresh"].any()


def test_episode_return_and_length_cover_the_done_step(
    scripted: dict[str, Any],
) -> None:
    ret, length = scripted["ret"][-1], scripted["len"][-1]
    assert (length == EPISODE).all()
    np.testing.assert_allclose(ret, scripted["reward"].sum(0), atol=1e-5)
    np.testing.assert_allclose(ret, scripted["logged"], atol=1e-5)
    assert ret[:2].tolist() == [1.0, 1.0]  # one achievement, reward 1 in Classic


def test_the_update_summary_counts_finished_episodes(scripted: dict[str, Any]) -> None:
    got = update_summary(
        scripted["done"], *(scripted[k] for k in ("ret", "len", "ach"))
    )
    assert got.n_episodes == ENVS
    assert got.ret == pytest.approx(2 / ENVS)
    assert got.length == EPISODE
    want = np.zeros(22)
    want[TABLE] = 100 * 2 / ENVS
    np.testing.assert_allclose(got.success, want)
    pct = score_pct(got.ret, achievement_weights(CLASSIC).sum())
    assert pct == pytest.approx(2 / ENVS / 22 * 100)


def test_a_step_before_the_end_only_counts_finished_episodes(
    scripted: dict[str, Any],
) -> None:
    mid = update_summary(
        scripted["done"][:-1], *(scripted[k][:-1] for k in ("ret", "len", "ach"))
    )
    assert mid.n_episodes == 0
    assert np.isnan([mid.ret, mid.length]).all()
    assert np.isnan(mid.success).all()


def test_the_achievements_match_craftaxs_own_report_on_done_steps() -> None:
    """A random policy for 400 steps: our episode flags equal ``Achievements/*``."""
    env = LogWrapper(make_craftax_env_from_name(CLASSIC, auto_reset=False))
    vec = OptimisticResetVecEnvWrapper(env, ENVS, 16)
    params = vec.default_params
    _, st = vec.reset(jax.random.key(0), params)
    acc, step = init_acc(ENVS, 22), jax.jit(episode_step)
    names = [a.name.lower() for a in sorted(Achievement, key=lambda a: a.value)]
    finished = 0
    for t in range(400):
        k_a, k_s = jax.random.split(jax.random.key(t))
        act = jax.random.randint(k_a, (ENVS,), 0, 17)
        _, st, r, d, info = vec.step(k_s, st, act, params)
        acc, end = step(acc, r, d, events_of(info), act)
        theirs = np.stack(
            [np.asarray(info[f"Achievements/{n}"]) > 0 for n in names], -1
        )
        d = np.asarray(d)
        assert np.array_equal(np.asarray(end.ach)[d], theirs[d]), t
        finished += d.sum()
    assert finished > 0
    assert np.asarray(end.ach).any(), "the random policy unlocked something"


def test_accumulators_restart_after_a_done_step() -> None:
    """Env 0 ends on its 2nd step with a new unlock; env 1 runs on."""
    acc, step = init_acc(2, 4), jax.jit(episode_step)
    noop = jnp.zeros(2, jnp.int32)
    unlocked = np.zeros((2, 4), bool)
    unlocked[0, 1] = True
    ev = _events(2, 4, unlocked=unlocked)
    acc, end = step(acc, jnp.array([1.0, 0.5]), jnp.array([False, False]), ev, noop)
    unlocked = np.zeros((2, 4), bool)
    unlocked[0, 3] = True
    ev = _events(2, 4, unlocked=unlocked)
    acc, end = step(acc, jnp.array([2.0, 0.0]), jnp.array([True, False]), ev, noop)
    assert np.asarray(end.ach).tolist() == [[0, 1, 0, 1], [0, 0, 0, 0]]
    assert np.asarray(end.ret).tolist() == [3.0, 0.5]
    assert np.asarray(end.length).tolist() == [2, 2]
    ev = _events(2, 4)
    acc, end = step(acc, jnp.array([1.0, 1.0]), jnp.array([False, True]), ev, noop)
    assert np.asarray(end.ach).tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert np.asarray(end.ret).tolist() == [1.0, 1.5]
    assert np.asarray(end.length).tolist() == [1, 3]


def test_first_unlock_is_the_earliest_env_step_of_the_update() -> None:
    """(steps 3, envs 2): achievement 1 unlocks first at step 2, 0 at step 1."""
    unlocked = np.zeros((3, 2, 4), bool)
    unlocked[0, 1, 0] = unlocked[1, 0, 1] = unlocked[2, 1, 1] = True
    got = jax.jit(first_unlock_offsets)(unlocked)
    assert np.asarray(got).tolist() == [2, 4, -1, -1]  # (t + 1) * envs


@pytest.mark.parametrize(("env", "top"), [(CLASSIC, 22), (CRAFTAX, 226)])
def test_score_is_the_return_as_a_share_of_the_maximum(env: str, top: int) -> None:
    assert achievement_weights(env).sum() == top
    assert score_pct(top / 2, achievement_weights(env).sum()) == 50.0
    both = score_pct(np.array([0.0, top / 4]), top)
    np.testing.assert_allclose(both, [0.0, 25.0])


def test_info_without_events_fails_loudly() -> None:
    with pytest.raises(ValueError, match=r"info.*events.*optimistic"):
        events_of({"returned_episode_returns": 0.0})


def test_auto_reset_ppo_refuses_to_log_rather_than_logging_zeros() -> None:
    """Craftax's own auto-reset gives no events, so the first update raises."""
    cfg = PPOConfig(env=CLASSIC, envs=8, num_steps=4, width=8, resets="auto")
    state = init_state(cfg, jax.random.key(0))
    with pytest.raises(ValueError, match="optimistic"):
        ppo_train_chunk(state, 1)


# --- online families: synthetic event streams ----------------------------------


def _first_update() -> tuple[Any, dict[str, Any], list[Any], list[list[bool]]]:
    """3 envs, 6 steps. Env 0 dies on step 3 with a kill, an unlock and an arrow hit
    on that step (a zombie hit before); env 1 ends alive on step 6; env 2 runs on."""
    events = [
        _hit(3, 0, 2.0, ZOMBIE, unlocked=_flag(3, 2, 22, WOOD)),
        _events(3, unlocked=_flag(3, 1, 22, WOOD), kills=_flag(3, 2, KINDS, ZOMBIE)),
        _hit(
            3, 0, 7.0, ARROW,
            kills=_flag(3, 0, KINDS, ZOMBIE), unlocked=_flag(3, 0, 22, TABLE),
        ),
        _events(3),
        _events(3),
        _events(3),
    ]  # fmt: skip
    done = [[False] * 3, [False] * 3, [True, False, False]]
    done += [[False] * 3, [False] * 3, [False, True, False]]
    acc, totals = _play(init_acc(3, 22), events, done)
    acc, out = jax.jit(close_update)(acc)
    return acc, out, totals, done


def test_the_done_step_kill_unlock_and_damage_count_in_the_ended_episode() -> None:
    _, out, totals, done = _first_update()
    online, lay = np.asarray(out["online"]), ONLINE_LAYOUT
    assert online[lay["episodes"]].tolist() == [2]
    assert online[lay["steps"]].tolist() == [3 + 6]
    assert online[lay["kills"]][ZOMBIE] == 1, "env 2's kill is in no finished episode"
    assert online[lay["damage_taken"]].tolist() == [9.0]
    assert online[lay["death"]].tolist() == [1]
    assert online[lay["cause"]][ARROW] == 1, "the arrow took more in the last 20 steps"
    m = _metrics(out, totals, done, "classic")
    assert m["train/kills/zombie"] == 0.5
    assert m["train/kills_per_1k/zombie"] == pytest.approx(1000 / 9)
    assert m["train/kills_per_1k/skeleton"] == 0.0
    assert m["train/damage_taken"] == 4.5
    assert (m["train/end/death"], m["train/end/timeout"]) == (50.0, 50.0)
    assert (m["train/death/arrow"], m["train/death/zombie"]) == (100.0, 0.0)
    assert m["train/funnel/depth_1"] == 100.0
    assert (m["train/funnel/depth_2"], m["train/funnel/depth_3"]) == (50.0, 0.0)
    assert m["train/funnel/floor_0"] == 100.0
    assert m["train/ach_per_episode"] == 1.0
    assert m["train/ach_reward_share"] == pytest.approx(2 / (0.1 + 1.0))
    assert (m["train/action/noop"], m["train/action_entropy"]) == (1.0, 0.0)
    assert m["train/wasted_share"] == 0.0


def test_accumulators_of_an_ended_episode_restart() -> None:
    acc, _, totals, _ = _first_update()
    assert int(totals[2].length[0]) == 3
    assert bool(totals[2].ach[0, TABLE])
    assert int(totals[5].length[0]) == 3, "env 0's second episode"
    assert not np.asarray(totals[5].ach[0]).any()
    lay = ONLINE_LAYOUT
    assert float(acc.counts[0, lay["steps"]][0]) == 3
    assert not np.asarray(acc.counts[0, lay["kills"]]).any(), "a fresh episode"
    assert not np.asarray(acc.counts[0, lay["damage_taken"]]).any()
    assert float(acc.health[0]) == 9.0
    assert float(acc.counts[2, ONLINE_LAYOUT["kills"].start + ZOMBIE]) == 1


def test_the_summary_keys_are_the_metric_keys() -> None:
    _, out, totals, done = _first_update()
    m = _metrics(out, totals, done, "classic")
    assert list(m) == list(metric_keys("classic"))


def test_an_update_without_finished_episodes_is_all_nan_and_first_unlocks_stay() -> (
    None
):
    acc, out, _, _ = _first_update()
    first = np.asarray(out["first_unlock"])
    assert (first[WOOD], first[TABLE]) == (3, 9)  # (step + 1) x 3 envs
    assert (np.delete(first, [WOOD, TABLE]) == -1).all()
    events = [
        _events(3, unlocked=_flag(3, 1, 22, WOOD)),  # env 1's new episode
        _events(3, unlocked=_flag(3, 1, 22, STONE)),
        _events(3),
    ]
    acc, totals = _play(acc, events, [[False] * 3] * 3)
    acc, out = jax.jit(close_update)(acc)
    first = np.asarray(out["first_unlock"])
    assert (first[WOOD], first[STONE]) == (3, 24), "never later, new ones at 8 x 3"
    assert not np.asarray(out["online"]).any()
    m = _metrics(out, totals, [[False] * 3] * 3, "classic")
    assert np.isnan(list(m.values())).all()


def test_close_calls_need_50_more_steps_and_stalls_300_without_unlock() -> None:
    """Env 0: health 2 at steps 10 and 320, ends at 350 with no unlock; env 1:
    unlocks at 299 and 600 (gaps 299 and 301), ends at 700."""
    steps = 700
    delta = np.zeros((steps, 2), np.float32)
    delta[9, 0], delta[99, 0], delta[319, 0] = -7.0, 3.0, -3.0
    unlocked = np.zeros((steps, 2, 22), bool)
    unlocked[298, 1, WOOD] = unlocked[599, 1, TABLE] = True
    done = np.zeros((steps, 2), bool)
    done[349, 0] = done[699, 1] = True
    events = [
        _events(
            2, health_delta=delta[t], health_loss=np.maximum(-delta[t], 0),
            unlocked=unlocked[t],
        )
        for t in range(steps)
    ]  # fmt: skip
    acc, totals = _play(init_acc(2, 22), events, done.tolist())
    _, out = close_update(acc)
    online = np.asarray(out["online"])
    assert online[ONLINE_LAYOUT["close_calls"]].tolist() == [1]
    assert online[ONLINE_LAYOUT["stalls"]].tolist() == [2]
    assert online[ONLINE_LAYOUT["death"]].tolist() == [0], "both end alive"
    m = _metrics(out, totals, done, "classic")
    assert (m["train/close_calls"], m["train/stalls_per_episode"]) == (0.5, 1.0)


def test_the_cause_of_death_is_the_largest_loss_of_the_last_20_steps() -> None:
    """Env 0: a zombie takes 5 at step 1, needs drain 4 x 1 in steps 21-40 while
    food is 0, death at 40; env 1: lava at step 40 without a health change."""
    zero_food = _flag(2, 0, 3, 0)
    drain = _events(
        2, health_delta=[-1.0, 0], health_loss=[1.0, 0], drain=[True, False],
        stat_zero=zero_food, stat_low=zero_food,
    )  # fmt: skip
    events = [_events(2)] * 40
    events[0] = _hit(2, 0, 5.0, ZOMBIE)
    events[29] = events[37] = events[38] = drain
    events[39] = drain.replace(damage_source=jnp.array([NO_SOURCE, LAVA], jnp.int8))
    done = [[False, False]] * 39 + [[True, True]]
    acc, totals = _play(init_acc(2, 22), events, done)
    _, out = close_update(acc)
    m = _metrics(out, totals, done, "classic")
    assert m["train/end/death"] == 100.0
    assert (m["train/death/starvation"], m["train/death/lava"]) == (50.0, 50.0)
    assert m["train/death/zombie"] == 0.0
    assert m["train/survival/food_zero"] == pytest.approx(4 / 80)


def test_action_usage_skips_forced_steps_and_counts_wasted_ones() -> None:
    """LEFT (moved), LEFT (blocked: wasted), NOOP, then a forced step; 1 episode."""
    effective, forced = [True, False, False, False], [False, False, False, True]
    events = [
        _events(1, effective=[e], forced=[f])
        for e, f in zip(effective, forced, strict=True)
    ]
    act = [[Action.LEFT.value], [Action.LEFT.value], [0], [Action.DO.value]]
    done = [[False]] * 3 + [[True]]
    acc, totals = _play(init_acc(1, 22), events, done, act)
    _, out = close_update(acc)
    m = _metrics(out, totals, done, "classic")
    assert m["train/action/left"] == pytest.approx(2 / 3)
    assert (m["train/action/noop"], m["train/action/do"]) == (pytest.approx(1 / 3), 0.0)
    assert m["train/wasted_share"] == pytest.approx(1 / 3)
    assert m["train/forced_share"] == 0.25
    h = -(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3))
    assert m["train/action_entropy"] == pytest.approx(h)


def test_first_unlocks_are_reported_in_the_update_that_holds_their_step() -> None:
    """Updates 2-4 of 100 env steps: 150 is in update 2, 310 in update 4, 50 earlier."""
    stamps = np.array([[-1, 150, 50], [-1, 150, 50], [310, 150, 50]])
    assert new_first_unlocks(stamps, 1, 100) == [(1, 150, 0), (0, 310, 2)]


def test_craftax_keys_add_floors_creatures_and_advanced_mechanics() -> None:
    classic, craftax = set(metric_keys("classic")), set(metric_keys("craftax"))
    assert "train/funnel/floor_8" in craftax
    assert "train/funnel/floor_1" not in classic
    assert "train/funnel/depth_10" in craftax
    assert "train/funnel/depth_10" not in classic
    assert "train/adv/cast_fireball" in craftax
    assert not any(k.startswith("train/adv/") for k in classic)
    assert {"train/kills/zombie", "train/kills/skeleton"} <= classic
    assert "train/kills/troll" in craftax
    assert {"train/kills_per_1k/troll", "train/kills_per_1k/passive"} <= craftax
    assert "train/kills/troll" not in classic
    assert sum(k.startswith("train/action/") for k in craftax) == 43
    assert sum(k.startswith("train/action/") for k in classic) == 17
