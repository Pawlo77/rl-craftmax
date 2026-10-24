"""Tests for per-episode records built from synthetic event, action and trace arrays."""

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax.constants import Achievement, Action, BlockType
from craftax.craftax_classic.constants import Action as ClassicAction

from wut_murl_craftax.evaluation.episode_metrics import (
    CAUSES,
    ONLINE_LAYOUT,
    episode_step,
    init_acc,
)
from wut_murl_craftax.evaluation.episode_records import (
    Trace,
    episode_record,
    stored_records,
    trace_of,
)
from wut_murl_craftax.evaluation.events import NO_SOURCE, StepEvents
from wut_murl_craftax.evaluation.milestones import CREATURE_OFFSET, CREATURES

ZOMBIE, ARROW = CREATURE_OFFSET["melee"], CREATURE_OFFSET["projectile"]
FOOD, DRINK = 0, 1
WOOD, DIAMOND = Achievement.COLLECT_WOOD.value, Achievement.COLLECT_DIAMOND.value


def quiet(n: int, n_ach: int = 67) -> dict[str, np.ndarray]:
    """Event fields of ``n`` steps where nothing happens."""

    def z(*shape: int, dtype: Any = bool) -> np.ndarray:
        return np.zeros((n, *shape), dtype)

    return {
        "unlocked": z(n_ach), "floor": z(dtype=np.int8),
        "health_delta": z(dtype=np.float32), "health_loss": z(dtype=np.float16),
        "drain": z(), "combat_damage": z(dtype=np.float16),
        "damage_source": np.full(n, NO_SOURCE, np.int8),
        "kills": z(len(CREATURES), dtype=np.int8), "passive_kills": z(dtype=np.int8),
        "damage_dealt": z(dtype=np.float16), "projectile_damage": z(dtype=np.float16),
        "ate": z(), "drank": z(), "slept": z(), "stat_low": z(3), "stat_zero": z(3),
        "forced": z(), "effective": z(), "hostiles_on_screen": z(dtype=np.int8),
        "hostiles_close": z(dtype=np.int8),
    }  # fmt: skip


def hit(ev: dict[str, np.ndarray], t: int, loss: float, source: int) -> None:
    ev["health_delta"][t] -= loss
    ev["health_loss"][t] += loss
    ev["combat_damage"][t] += loss
    ev["damage_source"][t] = source


def drain(ev: dict[str, np.ndarray], t: int, stat: int) -> None:
    ev["health_delta"][t] -= 1
    ev["health_loss"][t] += 1
    ev["drain"][t] = True
    ev["stat_zero"][t, stat] = True


def record(ev: dict[str, np.ndarray], actions: Any = None, **kw: Any) -> Any:
    n = len(ev["floor"])
    acts = np.zeros(n, np.int32) if actions is None else np.asarray(actions)
    return episode_record(StepEvents(**ev), acts, game=kw.pop("game", "craftax"), **kw)


def test_health_at_zero_is_a_death_and_a_full_health_end_a_timeout() -> None:
    dead = quiet(30)
    hit(dead, 29, 9.0, ZOMBIE)
    assert record(dead).end_type == "death"
    alive = record(quiet(30))
    assert (alive.end_type, alive.cause_of_death) == ("timeout", "")


def test_the_cause_is_the_largest_loss_of_the_last_20_steps() -> None:
    ev = quiet(100)
    hit(ev, 50, 5.0, ZOMBIE)  # the largest hit, but outside the window
    hit(ev, 85, 1.0, ARROW)
    for t in (90, 92, 99):
        drain(ev, t, DRINK)
    assert record(ev).cause_of_death == "dehydration"
    ev = quiet(100)
    hit(ev, 50, 4.0, ZOMBIE)
    hit(ev, 85, 2.0, ARROW)
    hit(ev, 90, 2.0, ARROW)
    drain(ev, 99, FOOD)
    assert record(ev).cause_of_death == "arrow"
    ev = quiet(100)
    hit(ev, 50, 6.0, ZOMBIE)
    for t in (95, 96, 99):
        drain(ev, t, FOOD)
    assert record(ev).cause_of_death == "starvation"


def test_a_400_step_gap_ended_by_an_unlock_is_one_escaped_stall() -> None:
    ev = quiet(600)
    ev["unlocked"][9, WOOD] = True  # step 10
    ev["unlocked"][409, Achievement.PLACE_TABLE.value] = True  # step 410
    ev["floor"][9:] = 1
    actions = np.full(600, Action.DO.value)
    actions[200:210] = Action.LEFT.value
    ev["forced"][205:210] = True  # asleep: the game played NOOP instead
    rec = record(ev, actions)
    assert len(rec.stalls) == 1
    s = rec.stalls[0]
    assert (s.start, s.end, s.length, s.floor) == (10, 410, 400, 1)
    assert (s.ending, s.ended_by) == ("unlock", "place_table")
    assert s.actions[Action.LEFT.value] == 5
    assert s.actions.sum() == 395
    assert s.intrinsics is None  # stats at the start need the replayed trace


def test_a_stall_from_the_start_or_to_the_end_is_kept_short_gaps_are_not() -> None:
    ev = quiet(1000)
    ev["unlocked"][349, WOOD] = True
    ev["unlocked"][399, Achievement.PLACE_TABLE.value] = True
    rec = record(ev)
    got = [(s.start, s.end, s.ending, s.ended_by) for s in rec.stalls]
    assert got == [(0, 350, "unlock", "collect_wood"), (400, 1000, "timeout", "")]
    assert rec.row()["n_stalls"] == 2
    assert rec.row()["max_stall"] == 600
    ev = quiet(599)
    ev["unlocked"][299, WOOD] = True  # exactly 300 steps, then 299
    assert [(s.start, s.end) for s in record(ev).stalls] == [(0, 300)]


def test_speedrun_splits_are_the_unlock_steps() -> None:
    ev = quiet(300)
    ev["unlocked"][4, WOOD] = True
    ev["unlocked"][49, Achievement.PLACE_TABLE.value] = True
    ev["unlocked"][199, Achievement.ENTER_DUNGEON.value] = True
    rec = record(ev)
    assert rec.splits["collect_wood"] == 5
    assert rec.splits["place_table"] == 50
    assert rec.splits["enter_dungeon"] == 200
    assert rec.splits["collect_diamond"] == -1
    assert rec.unlock_steps[WOOD] == 5
    assert rec.unlocks[0] == (5, "collect_wood")
    assert rec.tech_depth == 2
    assert rec.max_floor == 0


def test_fights_bridge_gaps_under_5_steps() -> None:
    ev = quiet(100)
    for a, b in ((10, 15), (18, 21), (40, 43), (47, 50)):  # gaps of 3 and 4 steps
        ev["hostiles_on_screen"][a:b] = 3
        ev["hostiles_close"][a:b] = 1
    ev["hostiles_on_screen"][19] = 4
    hit(ev, 12, 2.0, ZOMBIE)
    hit(ev, 19, 1.0, ZOMBIE)
    hit(ev, 44, 1.0, ARROW)  # inside the bridged gap
    ev["hostiles_on_screen"][60:63] = 2  # none within 2 tiles: no fight
    rec = record(ev)
    got = [
        (f.start, f.end, f.max_enemies, f.damage_taken, f.outcome) for f in rec.fights
    ]
    assert got == [(11, 21, 4, 3.0, "survived"), (41, 50, 3, 1.0, "survived")]
    ev = quiet(100)
    for a, b in ((10, 12), (17, 19)):  # a 5-step gap splits the fight
        ev["hostiles_on_screen"][a:b] = 2
        ev["hostiles_close"][a:b] = 1
    assert len(record(ev).fights) == 2


def test_a_death_shortly_after_the_last_fight_step_ends_the_fight_as_died() -> None:
    ev = quiet(100)
    ev["hostiles_on_screen"][90:97] = 2
    ev["hostiles_close"][90:97] = 1
    hit(ev, 99, 9.0, ZOMBIE)
    (fight,) = record(ev).fights
    assert (fight.start, fight.end, fight.outcome) == (91, 100, "died")


def classic_trace(inventory: list[list[int]], levels: list[list[int]]) -> Trace:
    n1 = len(inventory)
    return Trace(
        position=np.zeros((n1, 2), np.int32),
        floor=np.zeros(n1, np.int32),
        intrinsics=np.full((n1, 4), 9, np.float32),
        inventory=np.asarray(inventory),
        levels=np.asarray(levels),
        table_near=np.zeros(n1, bool),
    )


def test_gathered_and_used_are_the_positive_and_negative_inventory_changes() -> None:
    # wood, stone, coal, iron, diamond, sapling
    inv = [[0] * 6, [1, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0],
           [2, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0]]  # fmt: skip
    trace = classic_trace(inv, [[0, 0]] * 6)
    rec = record(quiet(5, 22), trace=trace, game="classic")
    row = rec.row()
    assert (row["gathered/wood"], row["used/wood"]) == (3, 1)
    assert (row["gathered/stone"], row["used/stone"]) == (1, 1)
    assert (row["gathered/sapling"], row["used/sapling"]) == (1, 1)
    assert row["use_ratio"] == pytest.approx(3 / 5)
    assert "gathered/ruby" not in row


def test_a_pickaxe_crafted_when_a_better_one_is_owned_is_a_wasted_craft() -> None:
    make_wood = ClassicAction.MAKE_WOOD_PICKAXE.value
    inv = [[3, 1, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]  # fmt: skip
    levels = [[0, 0], [1, 0], [2, 0], [2, 0], [2, 0]]  # pickaxe, sword
    actions = [make_wood, ClassicAction.MAKE_STONE_PICKAXE.value, make_wood, make_wood]
    trace = classic_trace(inv, levels)
    rec = record(quiet(4, 22), actions, trace=trace, game="classic")
    # wood, then stone pickaxe raise the level; the next wood pickaxe does not;
    # the last one has no wood, so nothing is consumed
    assert rec.wasted_crafts == 1


def test_a_table_placed_next_to_another_table_is_a_wasted_craft() -> None:
    place = ClassicAction.PLACE_TABLE.value
    inv = [[2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    trace = classic_trace(inv, [[0, 0]] * 3)._replace(
        table_near=np.array([1, 1, 1], bool)
    )
    rec = record(quiet(2, 22), [place, place], trace=trace, game="classic")
    assert rec.wasted_crafts == 2
    assert record(quiet(2, 22), game="classic").wasted_crafts is None


def test_wood_to_diamond_is_the_steps_between_their_first_unlocks() -> None:
    ev = quiet(300)
    ev["unlocked"][4, WOOD] = True
    ev["unlocked"][104, DIAMOND] = True
    assert record(ev).wood_to_diamond == 100
    ev["unlocked"][104, DIAMOND] = False
    assert record(ev).wood_to_diamond == -1


def test_return_score_actions_and_advanced_uses_come_from_the_stream() -> None:
    ev = quiet(4)
    ev["unlocked"][1, WOOD] = True
    hit(ev, 2, 2.0, ZOMBIE)
    ev["forced"][3] = True
    ev["effective"][1] = True
    actions = [Action.NOOP.value, Action.CAST_FIREBALL.value, Action.DO.value, 5]
    rec = record(ev, actions)
    assert rec.ret == pytest.approx(1.0 - 0.2)
    assert rec.score_pct == pytest.approx(0.8 / 226 * 100)
    row = rec.row()
    assert row["action_count/noop"] == 1
    assert row["action_count/do"] == 1
    assert row["noop_share"] == pytest.approx(1 / 3)
    assert row["wasted_share"] == pytest.approx(1 / 3)  # DO did nothing
    assert row["forced_share"] == pytest.approx(1 / 4)
    assert row["action_entropy"] == pytest.approx(np.log(3))
    assert row["adv/cast_fireball"] == 1
    assert row["steps_per_achievement"] == 4


def test_the_row_has_the_contract_columns_and_survives_json() -> None:
    ev = quiet(400)
    ev["unlocked"][349, WOOD] = True
    row = record(ev).row()
    for key in ("length", "return", "score_pct", "end_type", "cause_of_death",
                "max_floor", "tech_depth", "n_achievements", "ach/collect_wood",
                "unlock_step/collect_wood", "stalls", "n_stalls", "max_stall",
                "fights", "kills/zombie", "kills/passive", "damage_dealt",
                "damage_taken", "close_calls", "close_calls_recovered",
                "low_share/food", "zero_share/energy", "action_count/noop",
                "action_entropy", "wasted_share", "noop_share", "forced_share",
                "gathered/wood", "used/ruby", "use_ratio", "steps_per_achievement",
                "wasted_crafts", "wood_to_diamond", "adv/read_book",
                "split/make_diamond_pickaxe", "split/enter_graveyard"):  # fmt: skip
        assert key in row, key
    assert np.isnan(row["gathered/wood"])  # no trace: left to the replay
    assert row["ach/collect_wood"] is True
    assert isinstance(row["unlock_step/collect_wood"], np.int32)
    assert np.isnan(row["survival/level/food"])
    stall = json.loads(json.dumps(row["stalls"]))[0]
    assert {"start", "end", "length", "floor", "ending", "ended_by"} <= set(stall)


def test_close_calls_need_50_more_steps_and_recover_at_health_5() -> None:
    ev = quiet(160)
    hit(ev, 10, 7.0, ZOMBIE)  # health 2: a close call, recovered below
    ev["health_delta"][30] = 3.0
    hit(ev, 100, 3.0, ZOMBIE)  # health 2 again, never back to 5
    ev["health_delta"][120] += 1.0
    hit(ev, 125, 1.0, ZOMBIE)  # a new fall while open joins it; 34 steps remain
    rec = record(ev)
    assert (rec.close_calls, rec.close_calls_recovered) == (2, 1)


def _online(ev: dict[str, np.ndarray], actions: np.ndarray) -> Any:
    """Totals of the training accumulator over one episode ending at its last step."""
    n = len(actions)
    batched = StepEvents(**{k: jnp.asarray(v)[:, None] for k, v in ev.items()})
    done = jnp.arange(n)[:, None] == n - 1

    def step(acc: Any, x: Any) -> Any:
        e, a, d = x
        return episode_step(acc, jnp.zeros(1), d, e, a)

    acc = init_acc(1, ev["unlocked"].shape[1])
    _, totals = jax.lax.scan(step, acc, (batched, jnp.asarray(actions)[:, None], done))
    return np.asarray(totals.sums[-1, 0])


def test_the_record_agrees_with_the_training_accumulator() -> None:
    rng = np.random.default_rng(0)
    n = 900
    ev = quiet(n)
    hits = rng.choice(np.arange(0, 300, 2), 40, replace=False)
    for t in hits:  # healed in the next step, so health stays high
        loss = float(rng.integers(1, 3))
        hit(ev, int(t), loss, int(rng.choice([ZOMBIE, ARROW])))
        ev["health_delta"][t + 1] += loss
    for t in rng.choice(np.arange(301, 380, 2), 8, replace=False):
        drain(ev, int(t), int(rng.integers(0, 3)))
        ev["health_delta"][t + 1] += 1
    hit(ev, 400, 7.0, ZOMBIE)  # a close call, healed later
    ev["health_delta"][520] += 7
    hit(ev, n - 1, 9.0, ZOMBIE)
    for a, t in enumerate((5, 360, 700)):
        ev["unlocked"][t, a] = True
    ev["stat_low"] = rng.random((n, 3)) < 0.2
    ev["forced"] = rng.random(n) < 0.1
    ev["effective"] = rng.random(n) < 0.5
    ev["kills"][rng.choice(n, 5), ZOMBIE] = 1
    actions = rng.integers(0, 43, n)
    rec, sums = record(ev, actions), _online(ev, actions)

    def online(name: str) -> np.ndarray:
        return sums[ONLINE_LAYOUT[name]]

    assert rec.end_type == "death"
    cause = int(np.flatnonzero(online("cause"))[0])
    assert rec.cause_of_death == CAUSES[cause]
    assert len(rec.stalls) == online("stalls")[0] == 2
    assert rec.close_calls == online("close_calls")[0] == 1
    np.testing.assert_array_equal(rec.kills, online("kills"))
    assert rec.damage_taken == pytest.approx(float(online("damage_taken")[0]))
    np.testing.assert_allclose(rec.low_share * n, online("low"))
    np.testing.assert_array_equal(rec.action_counts, online("actions"))
    assert rec.wasted_share * rec.action_counts.sum() == pytest.approx(
        online("wasted")[0]
    )


def test_stored_records_slice_each_seed_to_its_length() -> None:
    ev = quiet(10)
    ev["unlocked"][2, WOOD] = True
    arrays = {f"events/{k}": np.stack([v, v]) for k, v in ev.items()}
    actions = np.full((2, 10), -1, np.int8)
    actions[0, :10], actions[1, :4] = 0, 1
    arrays |= {
        "actions": actions,
        "length": np.array([10, 4]),
        "seeds": np.array([7, 8]),
    }
    a, b = stored_records(arrays, "craftax")
    assert (a.length, b.length) == (10, 4)
    assert b.row()["action_count/left"] == 4
    assert a.unlock_steps[WOOD] == b.unlock_steps[WOOD] == 3


def _rows(state: Any) -> Any:
    """A one-row trajectory of ``state``."""
    return jax.tree.map(lambda x: np.asarray(x)[None], state)


def test_trace_of_reads_craftax_states(craftax_start: Any) -> None:
    s = craftax_start
    table = s.replace(
        map=s.map.at[0, 24, 25].set(BlockType.CRAFTING_TABLE.value),
        inventory=s.inventory.replace(wood=jnp.asarray(4), pickaxe=jnp.asarray(2)),
    )
    states = jax.tree.map(lambda a, b: np.concatenate([a, b]), _rows(s), _rows(table))
    tr = trace_of(states, "craftax")
    assert tr.position.tolist() == [[24, 24], [24, 24]]
    assert tr.floor.tolist() == [0, 0]
    assert tr.table_near.tolist() == [False, True]
    assert tr.inventory.shape == (2, 8)
    assert tr.inventory[1, 0] == 4
    assert tr.levels[1, 0] == 2
    assert tr.intrinsics[0].tolist() == [9, 9, 9, 9]


def test_trace_of_gives_classic_tool_levels(classic_start: Any) -> None:
    s = classic_start
    inv = s.inventory.replace(wood_pickaxe=jnp.asarray(2), stone_pickaxe=jnp.asarray(1))
    tr = trace_of(_rows(s.replace(inventory=inv)), "classic")
    assert tr.inventory.shape == (1, 6)
    assert tr.levels.tolist() == [[2, 0]]
    assert tr.floor.tolist() == [0]


@pytest.mark.parametrize("game", ["craftax", "classic"])
def test_trace_of_runs_under_jit_with_jnp_as_with_numpy(
    game: str, craftax_start: Any, classic_start: Any
) -> None:
    """A replay scan builds its trace rows in jit; they equal the numpy trace."""
    s = craftax_start if game == "craftax" else classic_start
    moved = s.replace(player_position=jnp.asarray([0, 1], jnp.int32))  # map edge
    states = jax.tree.map(lambda a, b: np.concatenate([a, b]), _rows(s), _rows(moved))
    want = trace_of(states, game)
    got = jax.jit(lambda x: trace_of(x, game, xp=jnp))(states)
    for w, g in zip(want, got, strict=True):
        np.testing.assert_array_equal(np.asarray(g), w)  # dtypes: jax has no int64


def test_survival_timing_counts_refills_their_gaps_and_the_stat_before() -> None:
    ev = quiet(100)
    ev["ate"][[10, 40, 90]] = True
    ev["drank"][50] = True
    ev["slept"][[20, 30]] = True
    intrinsics = np.full((101, 4), 9.0, np.float32)
    intrinsics[[10, 40, 90], 1] = [3, 5, 1]  # food before each meal
    intrinsics[50, 2] = 2
    intrinsics[[20, 30], 3] = [1, 4]
    trace = Trace(
        position=np.zeros((101, 2), np.int32),
        floor=np.zeros(101, np.int32),
        intrinsics=intrinsics,
        inventory=np.zeros((101, 8), np.int32),
        levels=np.zeros((101, 9), np.int32),
        table_near=np.zeros(101, bool),
    )
    row = record(ev, trace=trace).row()
    assert [row[f"survival/count/{s}"] for s in ("food", "drink", "energy")] == [
        3,
        1,
        2,
    ]
    assert row["survival/gap/food"] == 40  # (30 + 50) / 2
    assert np.isnan(row["survival/gap/drink"])
    assert row["survival/gap/energy"] == 10
    assert row["survival/level/food"] == 3
    assert row["survival/level/drink"] == 2
    assert row["survival/level/energy"] == 2.5
