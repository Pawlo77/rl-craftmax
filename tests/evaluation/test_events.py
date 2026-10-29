"""Tests for the per-step events on hand-edited Craftax and Craftax-Classic states."""

import functools
import os
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax.constants import Achievement, Action, BlockType
from craftax.craftax.craftax_state import EnvParams
from craftax.craftax_classic.constants import Action as ClassicAction
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.events import (
    DESPAWN_DISTANCE,
    LAVA,
    NO_SOURCE,
    StepEvents,
    despawn_distance,
    step_events,
)
from wut_murl_craftax.evaluation.milestones import CREATURE_OFFSET, CREATURES, FLOORS

ZOMBIE = CREATURE_OFFSET["melee"]
SKELETON = CREATURE_OFFSET["ranged"]
ARROW = CREATURE_OFFSET["projectile"]
ME = (24, 24)
"""Craftax start position; the player faces up, towards (23, 24)."""


class Env:
    """A no-auto-reset env with jitted step and events, reset with key 0."""

    def __init__(self, name: str, game: str) -> None:
        self.env = make_craftax_env_from_name(name, auto_reset=False)
        self.params = self.env.default_params
        self.game = game
        self.reset = jax.jit(self.env.reset)
        self.step = jax.jit(self.env.step)
        self.events = jax.jit(step_events, static_argnums=0)

    def start(self) -> Any:
        return self.reset(jax.random.PRNGKey(0), self.params)[1]

    def play(self, s: Any, action: int) -> tuple[StepEvents, bool]:
        _, s2, _, done, _ = self.step(jax.random.PRNGKey(1), s, action, self.params)
        return self.events(self.game, s, action, s2), bool(done)


@pytest.fixture(scope="module")
def craftax() -> Env:
    return Env("Craftax-Symbolic-v1", "craftax")


@pytest.fixture(scope="module")
def classic() -> Env:
    return Env("Craftax-Classic-Symbolic-v1", "classic")


def _off(m: Any) -> Any:
    return m.replace(mask=jnp.zeros_like(m.mask))


def quiet(s: Any) -> Any:
    """No creatures or projectiles anywhere, so nothing moves into the test."""
    return s.replace(
        melee_mobs=_off(s.melee_mobs),
        passive_mobs=_off(s.passive_mobs),
        ranged_mobs=_off(s.ranged_mobs),
        mob_projectiles=_off(s.mob_projectiles),
        player_projectiles=_off(s.player_projectiles),
        mob_map=jnp.zeros_like(s.mob_map),
    )


def put(s: Any, group: str, pos: tuple[int, int], **fields: Any) -> Any:
    """Slot 0 of ``group`` on floor 0, alive at ``pos``; mobs also enter ``mob_map``."""
    m = getattr(s, group)
    m = m.replace(
        position=m.position.at[0, 0].set(jnp.array(pos)),
        mask=m.mask.at[0, 0].set(True),
        **{k: getattr(m, k).at[0, 0].set(v) for k, v in fields.items()},
    )
    if group.endswith("_mobs"):
        s = s.replace(mob_map=s.mob_map.at[0, *pos].set(True))
    return s.replace(**{group: m})


def block(s: Any, pos: tuple[int, int], kind: BlockType) -> Any:
    return s.replace(map=s.map.at[0, *pos].set(kind.value))


def walled(s: Any, pos: tuple[int, int]) -> Any:
    """Stone on the four sides of ``pos`` (except the player's tile)."""
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        side = (pos[0] + dr, pos[1] + dc)
        if side != ME:
            s = block(s, side, BlockType.STONE)
    return s


def test_place_table_unlocks_one_bit(craftax: Env) -> None:
    s = block(quiet(craftax.start()), (23, 24), BlockType.GRASS)
    s = s.replace(inventory=s.inventory.replace(wood=2))
    ev, _ = craftax.play(s, Action.PLACE_TABLE.value)
    assert ev.unlocked.shape == (67,)
    assert np.flatnonzero(ev.unlocked).tolist() == [Achievement.PLACE_TABLE.value]
    assert ev.effective


def test_unlock_on_the_terminal_step_is_kept(craftax: Env) -> None:
    s = block(quiet(craftax.start()), (23, 24), BlockType.GRASS)
    s = s.replace(
        inventory=s.inventory.replace(wood=2),
        timestep=craftax.params.max_timesteps - 1,
    )
    ev, done = craftax.play(s, Action.PLACE_TABLE.value)
    assert done
    assert ev.unlocked[Achievement.PLACE_TABLE.value]


def test_descend_on_an_open_ladder_reaches_floor_1(craftax: Env) -> None:
    s = quiet(craftax.start())
    s = s.replace(player_position=s.down_ladders[0])
    ev, _ = craftax.play(s, Action.DESCEND.value)
    assert int(ev.floor) == 1
    assert ev.unlocked[Achievement.ENTER_DUNGEON.value]


def test_starvation_is_needs_drain_not_combat(craftax: Env) -> None:
    s = quiet(craftax.start()).replace(player_food=0, player_recover=-15.0)
    ev, _ = craftax.play(s, Action.NOOP.value)
    assert float(ev.health_loss) == 1
    assert ev.drain
    assert float(ev.combat_damage) == 0
    assert int(ev.damage_source) == NO_SOURCE


def _zombie_next_to_me(s: Any) -> Any:
    return put(s, "melee_mobs", (24, 25), health=5.0, attack_cooldown=0, type_id=0)


def test_zombie_hit_is_melee_damage_from_a_zombie(craftax: Env) -> None:
    ev, _ = craftax.play(_zombie_next_to_me(quiet(craftax.start())), Action.NOOP.value)
    assert float(ev.health_loss) == 2
    assert not ev.drain
    assert float(ev.combat_damage) == 2
    assert int(ev.damage_source) == ZOMBIE
    assert CREATURES[ZOMBIE][2] == "zombie"
    assert int(ev.hostiles_on_screen) == int(ev.hostiles_close) == 1


def test_drain_and_hit_in_one_step_split_one_point_to_drain(craftax: Env) -> None:
    s = quiet(craftax.start()).replace(player_food=0, player_recover=-15.0)
    ev, _ = craftax.play(_zombie_next_to_me(s), Action.NOOP.value)
    assert float(ev.health_loss) == 3
    assert ev.drain
    assert float(ev.combat_damage) == 2
    assert int(ev.damage_source) == ZOMBIE


def test_enemy_arrow_hit_is_projectile_damage(craftax: Env) -> None:
    s = put(quiet(craftax.start()), "mob_projectiles", (24, 25), type_id=0)
    s = s.replace(
        mob_projectile_directions=s.mob_projectile_directions.at[0, 0].set(
            jnp.array([0, -1])
        )
    )
    ev, _ = craftax.play(s, Action.NOOP.value)
    assert float(ev.combat_damage) == 2
    assert int(ev.damage_source) == ARROW


def test_point_blank_shot_of_a_cornered_skeleton(craftax: Env) -> None:
    s = walled(quiet(craftax.start()), (24, 25))
    s = put(s, "ranged_mobs", (24, 25), health=3.0, attack_cooldown=0, type_id=0)
    ev, _ = craftax.play(s, Action.NOOP.value)
    assert float(ev.combat_damage) == 2
    assert int(ev.damage_source) == ARROW


def test_skeleton_killed_by_an_arrow(craftax: Env) -> None:
    at = (24, 30)
    s = walled(quiet(craftax.start()), at)
    s = put(s, "ranged_mobs", at, health=1.0, attack_cooldown=3, type_id=0)
    s = put(s, "player_projectiles", at, type_id=0)
    s = s.replace(
        player_projectile_directions=s.player_projectile_directions.at[0, 0].set(
            jnp.array([0, 1])
        )
    )
    ev, _ = craftax.play(s, Action.NOOP.value)
    assert ev.kills.dtype == jnp.int8
    assert np.flatnonzero(ev.kills).tolist() == [SKELETON]
    assert int(ev.kills[SKELETON]) == 1
    assert int(ev.passive_kills) == 0
    assert float(ev.projectile_damage) == float(ev.damage_dealt) == 1
    assert ev.unlocked[Achievement.DEFEAT_SKELETON.value]


@pytest.mark.parametrize("refill_at", [(30, 30), (22, 24)])
def test_slot_refilled_in_the_kill_step_still_counts_the_kill(
    craftax: Env, refill_at: tuple[int, int]
) -> None:
    s = put(quiet(craftax.start()), "melee_mobs", (23, 24), health=1.0, type_id=0)
    m = s.melee_mobs  # a spawn refills slot 0 with a fresh zombie, far or next door
    s2 = s.replace(
        melee_mobs=m.replace(
            position=m.position.at[0, 0].set(jnp.array(refill_at)),
            health=m.health.at[0, 0].set(5.0),
        )
    )
    ev = craftax.events("craftax", s, Action.DO.value, s2)
    assert int(ev.kills[ZOMBIE]) == 1
    assert float(ev.damage_dealt) == 1
    assert float(ev.projectile_damage) == 0


def test_despawn_is_not_a_kill(craftax: Env) -> None:
    s = put(quiet(craftax.start()), "melee_mobs", (24, 38), health=5.0, type_id=0)
    ev, _ = craftax.play(s, Action.NOOP.value)  # 14 tiles away: despawns
    assert int(ev.kills.sum()) == 0
    assert float(ev.damage_dealt) == 0


def _vanish_on_floor(craftax: Env, group: str, floor: int, dist: int) -> StepEvents:
    """Events of a slot-0 mob ``dist`` tiles away that is gone in ``s2``."""
    s = quiet(craftax.start()).replace(player_level=jnp.int32(floor))
    m = getattr(s, group)
    alive = m.replace(
        position=m.position.at[floor, 0].set(jnp.array((24, 24 + dist))),
        mask=m.mask.at[floor, 0].set(True),
        health=m.health.at[floor, 0].set(1.0),
    )
    s = s.replace(**{group: alive})
    s2 = s.replace(**{group: alive.replace(mask=alive.mask.at[floor, 0].set(False))})
    return craftax.events("craftax", s, Action.NOOP.value, s2)


def test_the_boss_floor_keeps_melee_and_ranged_mobs_but_not_passive_ones(
    craftax: Env,
) -> None:
    far, boss = 14, len(FLOORS) - 1
    assert int(_vanish_on_floor(craftax, "melee_mobs", 0, far).kills.sum()) == 0
    assert int(_vanish_on_floor(craftax, "melee_mobs", boss, far).kills.sum()) == 1
    assert int(_vanish_on_floor(craftax, "ranged_mobs", boss, far).kills.sum()) == 1
    cow = _vanish_on_floor(craftax, "passive_mobs", boss, far)
    assert int(cow.passive_kills) == 0  # a passive mob despawns on the boss floor too
    assert float(cow.damage_dealt) == 0
    assert int(_vanish_on_floor(craftax, "passive_mobs", boss, 3).passive_kills) == 1


def test_the_despawn_distance_is_an_argument(craftax: Env) -> None:
    s = put(quiet(craftax.start()), "melee_mobs", (24, 36), health=1.0, type_id=0)
    m = s.melee_mobs
    s2 = s.replace(melee_mobs=m.replace(mask=m.mask.at[0, 0].set(False)))
    act = Action.NOOP.value
    assert int(craftax.events("craftax", s, act, s2).kills.sum()) == 1  # 12 < 14
    far = jax.jit(step_events, static_argnames=("game", "despawn"))
    for dist, kills in ((12, 0), (13, 1)):
        ev = far("craftax", s, act, s2, despawn=dist)
        assert int(ev.kills.sum()) == kills


def test_despawn_distance_of_the_params() -> None:
    p = EnvParams()
    assert despawn_distance(None) == DESPAWN_DISTANCE == p.mob_despawn_distance
    assert despawn_distance(p.replace(mob_despawn_distance=9)) == 9


def _cow_in_front(s: Any) -> Any:
    s = put(s, "passive_mobs", (23, 24), health=1.0, type_id=0)
    return s.replace(player_food=3)


def _water_in_front(s: Any) -> Any:
    return block(s, (23, 24), BlockType.WATER).replace(player_drink=3)


def _tired(s: Any) -> Any:
    return s.replace(player_energy=3)


@pytest.mark.parametrize(
    ("edit", "action", "flag"),
    [
        (_cow_in_front, Action.DO, "ate"),
        (_water_in_front, Action.DO, "drank"),
        (_tired, Action.SLEEP, "slept"),
    ],
)
def test_eat_drink_and_sleep_start(
    craftax: Env, edit: Any, action: Action, flag: str
) -> None:
    ev, _ = craftax.play(edit(quiet(craftax.start())), action.value)
    got = {f: bool(getattr(ev, f)) for f in ("ate", "drank", "slept")}
    assert got == {f: f == flag for f in got}
    if flag == "ate":
        assert int(ev.passive_kills) == 1
        assert int(ev.kills.sum()) == 0
        assert float(ev.projectile_damage) == 0
        assert float(ev.damage_dealt) == 1


@pytest.mark.parametrize(("direction", "effective"), [(3, False), (1, True)])
def test_walking_into_stone_is_wasted_unless_it_turns(
    craftax: Env, direction: int, effective: bool
) -> None:
    s = block(quiet(craftax.start()), (23, 24), BlockType.STONE)
    ev, _ = craftax.play(s.replace(player_direction=direction), Action.UP.value)
    assert bool(ev.effective) is effective
    assert not ev.forced


def _into_stone(s: Any) -> Any:
    """Facing stone above, so ``UP`` changes nothing the player did."""
    return block(s, (23, 24), BlockType.STONE)


def test_passive_stat_decay_does_not_make_an_action_effective(craftax: Env) -> None:
    s = _into_stone(quiet(craftax.start()))
    s = s.replace(player_hunger=25.0, player_thirst=20.0, player_fatigue=30.0)
    ev, _ = craftax.play(s, Action.UP.value)  # food, drink and energy all fall by 1
    assert bool(ev.stat_low.any()) is False
    assert not ev.effective


def test_a_projectile_in_flight_does_not_make_an_action_effective(
    craftax: Env,
) -> None:
    s = put(_into_stone(quiet(craftax.start())), "player_projectiles", (24, 26))
    s = s.replace(
        player_projectile_directions=s.player_projectile_directions.at[0, 0].set(
            jnp.array([0, 1])
        )
    )
    ev, _ = craftax.play(s, Action.UP.value)
    assert not ev.effective


def test_casting_a_fireball_is_effective(craftax: Env) -> None:
    s = block(quiet(craftax.start()), (23, 24), BlockType.GRASS)
    s = s.replace(learned_spells=s.learned_spells.at[0].set(True))
    ev, _ = craftax.play(s, Action.CAST_FIREBALL.value)
    assert ev.effective


def test_any_action_while_sleeping_is_a_forced_noop(craftax: Env) -> None:
    s = quiet(craftax.start()).replace(is_sleeping=True, player_energy=3)
    ev, _ = craftax.play(s, Action.DO.value)
    assert ev.forced


def test_bad_potion_is_neither_drain_nor_combat(craftax: Env) -> None:
    s = quiet(craftax.start())
    colour = int(np.flatnonzero(np.asarray(s.potion_mapping) == 1)[0])
    s = s.replace(
        inventory=s.inventory.replace(potions=s.inventory.potions.at[colour].set(1))
    )
    ev, _ = craftax.play(s, Action.DRINK_POTION_RED.value + colour)
    assert float(ev.health_loss) == 3
    assert float(ev.combat_damage) == 0
    assert int(ev.damage_source) == NO_SOURCE


def test_classic_lava_under_the_player_is_the_damage_source(classic: Env) -> None:
    s = classic.start()
    s = s.replace(
        zombies=_off(s.zombies),
        skeletons=_off(s.skeletons),
        arrows=_off(s.arrows),
        mob_map=jnp.zeros_like(s.mob_map),
    )
    r, c = (int(x) for x in s.player_position)
    s = s.replace(map=s.map.at[r, c + 1].set(BlockType.LAVA.value))
    ev, done = classic.play(s, ClassicAction.RIGHT.value)
    assert done
    assert ev.unlocked.shape == (22,)
    assert int(ev.floor) == 0
    assert float(ev.health_loss) == 9
    assert int(ev.damage_source) == LAVA


def test_vmap_equals_a_loop_over_eight_envs(craftax: Env) -> None:
    n = 8
    keys = jax.random.split(jax.random.PRNGKey(3), n)
    _, s = jax.vmap(craftax.env.reset, (0, None))(keys, craftax.params)
    vstep = jax.jit(jax.vmap(craftax.env.step, (0, 0, 0, None)))
    actions = [
        jax.random.randint(jax.random.fold_in(keys[0], t), (n,), 0, 43)
        for t in range(40)
    ]
    for a in actions[:-1]:  # random play moves mobs, stats and the map
        s = vstep(keys, s, a, craftax.params)[1]
    a = actions[-1]
    s2 = vstep(keys, s, a, craftax.params)[1]
    batched = jax.jit(jax.vmap(functools.partial(step_events, "craftax")))(s, a, s2)
    for i in range(n):
        row = functools.partial(jax.tree.map, lambda x, i=i: x[i])
        one = craftax.events("craftax", row(s), a[i], row(s2))
        for got, want in zip(
            jax.tree.leaves(batched), jax.tree.leaves(one), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(got)[i], np.asarray(want))


def test_importing_events_keeps_the_cpu_split_open() -> None:
    code = (
        "import jax\n"
        "from wut_murl_craftax.evaluation import events\n"
        "from wut_murl_craftax.orchestration.runtime import configure_runtime\n"
        "configure_runtime(host_devices=3)\n"
        "print(len(jax.devices('cpu')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "XLA_FLAGS": ""},
    )
    assert out.stdout.split()[-1] == "3"
