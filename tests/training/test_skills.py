"""Tests for the skill library on scripted Craftax and Craftax-Classic states.

Every skill function gets only decoded observations (rendered by the env), the action
and the reward; ``EnvState`` is read here to build the states and the expected values.
"""

import inspect
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax.constants import Action, BlockType
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.events import step_events
from wut_murl_craftax.training import skills
from wut_murl_craftax.training.progress_key import fields, unwrap
from wut_murl_craftax.training.skills import (
    NUM_SKILLS,
    REASONS,
    SKILL_IDS,
    SKILLS,
    ObsView,
    achievement_reward,
    available,
    decode_obs,
    key_fields,
    low_stats,
    primitive_action,
    skill_step,
)

ME, UP = (24, 24), (23, 24)
"""Craftax start position and the tile it faces (direction up)."""
GROUPS = ("melee_mobs", "passive_mobs", "ranged_mobs")
CRAFTAX_GROUPS = (*GROUPS, "mob_projectiles", "player_projectiles")
CLASSIC_GROUPS = ("zombies", "cows", "skeletons", "arrows")
CRAFTAX_ONLY = (
    "collect_gem", "make_diamond_pickaxe", "make_armour", "make_torches",
    "descend", "open_chest", "enchant", "read_book", "level_up",
)  # fmt: skip


class Game:
    """A no-auto-reset env with jitted reset, step and observation."""

    def __init__(self, name: str) -> None:
        self.env = make_craftax_env_from_name(name, auto_reset=False)
        self.params = self.env.default_params
        self.obs = jax.jit(self.env.get_obs)
        self._step = jax.jit(self.env.step)
        self.groups = CLASSIC_GROUPS if "Classic" in name else CRAFTAX_GROUPS
        self.state = self.env.reset(jax.random.key(0), self.params)[1]

    def start(self) -> Any:
        """The reset state with every creature slot dead and far off screen."""
        s = self.state
        return s.replace(**{g: _gone(getattr(s, g)) for g in self.groups})

    def view(self, s: Any) -> ObsView:
        return decode_obs(self.obs(s))

    def play(self, s: Any, action: int) -> tuple[Any, float, bool]:
        _, s2, r, done, _ = self._step(jax.random.key(1), s, action, self.params)
        return s2, float(r), bool(done)


def _gone(m: Any) -> Any:
    # a dead slot on screen would clear the channel of a live one of its type
    return m.replace(
        mask=jnp.zeros_like(m.mask), position=jnp.full_like(m.position, -1000)
    )


@pytest.fixture(scope="module")
def craftax() -> Game:
    return Game("Craftax-Symbolic-v1")


@pytest.fixture(scope="module")
def classic() -> Game:
    return Game("Craftax-Classic-Symbolic-v1")


def inv(s: Any, **items: Any) -> Any:
    return s.replace(inventory=s.inventory.replace(**items))


def block(s: Any, pos: tuple[int, int], kind: BlockType, floor: int = 0) -> Any:
    return s.replace(map=s.map.at[floor, *pos].set(kind.value))


def mob(
    s: Any, group: str, pos: tuple[int, int], type_id: int = 0, health: float = 5.0
) -> Any:
    """Slot 0 of ``group`` alive at ``pos`` on floor 0."""
    m = getattr(s, group)
    m = m.replace(
        position=m.position.at[0, 0].set(jnp.array(pos)),
        mask=m.mask.at[0, 0].set(True),
        health=m.health.at[0, 0].set(health),
        type_id=m.type_id.at[0, 0].set(type_id),
    )
    return s.replace(**{group: m})


def step(
    skill: str,
    a: ObsView,
    b: ObsView,
    action: int = Action.NOOP.value,
    r: float = 0.0,
    done: bool = False,
    steps: int = 0,
    mult: float = 1.0,
    granularity: str = "recipe",
) -> tuple[bool, bool, str]:
    """``skill_step`` of one named skill as Python values (success, ended, reason)."""
    out = skill_step(
        SKILL_IDS[skill], a, action, r, b, done, steps, mult, granularity=granularity
    )
    success, ended, reason = out
    return bool(success), bool(ended), REASONS[int(reason)]


def can(skill: str, v: ObsView, granularity: str = "recipe") -> bool:
    return bool(available(v, granularity)[SKILL_IDS[skill]])


# --- table and constants ---


def test_skill_table() -> None:
    names = [s.name for s in SKILLS]
    assert NUM_SKILLS == len(SKILLS) == 25
    assert names[:6] == [
        "collect_wood", "collect_stone", "collect_coal", "collect_iron",
        "collect_diamond", "collect_gem",
    ]  # fmt: skip
    assert names[-2:] == ["read_book", "level_up"]
    assert [s.name for s in SKILLS if s.kind == "primitive"] == names[-2:]
    t_max = {s.name: s.t_max for s in SKILLS}
    assert t_max["place_table"] == 100
    assert t_max["collect_diamond"] == 500
    assert t_max["sleep"] == 400
    assert t_max["descend"] == 1000
    assert t_max["read_book"] == t_max["level_up"] == 1
    assert {n: i for i, n in enumerate(names)} == SKILL_IDS


def test_ids_are_the_craftax_enums() -> None:
    for name in ("TREE", "PATH", "CRAFTING_TABLE", "FURNACE", "CHEST", "NECROMANCER"):
        assert getattr(skills, name) == BlockType[name].value
    assert BlockType.NECROMANCER_VULNERABLE.value == skills.NECROMANCER_VULNERABLE
    for name in ("NOOP", "DO", "PLACE_TABLE", "PLACE_FURNACE", "READ_BOOK"):
        assert getattr(skills, name) == Action[name].value
    assert Action.LEVEL_UP_DEXTERITY.value == skills.LEVEL_UP_DEXTERITY
    assert Action.LEVEL_UP_INTELLIGENCE.value == skills.LEVEL_UP_INTELLIGENCE


# --- decoder and key ---


def _random_craftax(base: Any, rng: np.random.Generator) -> Any:
    """A Craftax state with inventory, stats, floor, view and creatures drawn."""

    def ints(hi: int, n: int | None = None, lo: int = 0) -> Any:
        return jnp.asarray(rng.integers(lo, hi + 1, n), jnp.int32)

    floor = int(rng.integers(0, 9))
    p = rng.integers(2, 46, 2)
    s = inv(
        base, **{k: ints(99) for k in ("wood", "stone", "coal", "iron", "diamond")},
        sapphire=ints(99), ruby=ints(99), sapling=ints(99), torches=ints(99),
        arrows=ints(99), books=ints(99), pickaxe=ints(4), sword=ints(4), bow=ints(1),
        armour=ints(2, 4), potions=ints(99, 6),
    )  # fmt: skip
    r0, c0 = p - 1
    s = s.replace(
        player_level=jnp.int32(floor),
        player_position=jnp.asarray(p, jnp.int32),
        player_direction=ints(4, lo=1),
        player_health=jnp.float32(rng.uniform(0, 13)),
        player_food=ints(13), player_drink=ints(13), player_energy=ints(13),
        player_mana=ints(21), player_xp=ints(20), player_dexterity=ints(5, lo=1),
        player_strength=ints(5, lo=1), player_intelligence=ints(5, lo=1),
        is_sleeping=jnp.bool_(rng.integers(2)),
        learned_spells=jnp.asarray(rng.integers(0, 2, 2), bool),
        sword_enchantment=ints(2), bow_enchantment=ints(2),
        armour_enchantments=ints(2, 4), monsters_killed=ints(12, 9),
        map=s.map.at[floor, r0 : r0 + 3, c0 : c0 + 3].set(ints(36, (3, 3))),
        item_map=s.item_map.at[floor, r0 : r0 + 3, c0 : c0 + 3].set(ints(4, (3, 3))),
        light_map=s.light_map.at[floor, r0 - 1 : r0 + 4, c0 - 1 : c0 + 4].set(
            jnp.asarray(rng.choice([0.0, 0.04, 0.5, 1.0], (5, 5)), jnp.float32)
        ),
    )  # fmt: skip
    for slot in range(4):  # up to four creatures, one slot each, any class
        g = CRAFTAX_GROUPS[rng.integers(5)]
        m = getattr(s, g)
        at = jnp.asarray(p + rng.integers(-2, 3, 2), jnp.int32)
        m = m.replace(
            position=m.position.at[floor, slot].set(at),
            mask=m.mask.at[floor, slot].set(True),
            type_id=m.type_id.at[floor, slot].set(
                rng.integers(0, 3 if g == "passive_mobs" else 8)
            ),
        )
        s = s.replace(**{g: m})
    return s


def _expected_tiles(s: Any, groups: tuple[str, ...], floor: Any) -> tuple[Any, ...]:
    """Blocks and items of the 3 x 3 around the player, creatures of the 5 x 5."""
    p = np.asarray(s.player_position)
    r0, c0 = p - 1
    pick = (lambda x: x[floor]) if floor is not None else (lambda x: x)
    blocks = np.asarray(pick(s.map))[r0 : r0 + 3, c0 : c0 + 3]
    items = np.zeros((3, 3), int)
    if floor is not None:
        items = np.asarray(s.item_map[floor])[r0 : r0 + 3, c0 : c0 + 3]
    mobs = np.zeros((5, 5, 40), bool)
    for cls, g in enumerate(groups):
        m = getattr(s, g)
        pos, mask = np.asarray(pick(m.position)), np.asarray(pick(m.mask))
        types = np.asarray(pick(m.type_id)) if floor is not None else 0 * mask
        for (r, c), alive, t in zip(pos, mask, types, strict=True):
            if alive and abs(r - p[0]) <= 2 and abs(c - p[1]) <= 2:
                mobs[r - r0 + 1, c - c0 + 1, cls * 8 + t] = True
    if floor is not None:  # dark tiles are blank in the observation
        lit = np.asarray(s.light_map[floor])[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4] > 0.05
        blocks, items = blocks * lit[1:4, 1:4], items * lit[1:4, 1:4]
        mobs &= lit[..., None]
    return blocks, items, mobs


def _check_tiles(v: ObsView, s: Any, groups: tuple[str, ...], floor: Any) -> None:
    blocks, items, mobs = _expected_tiles(s, groups, floor)
    np.testing.assert_array_equal(v.blocks, blocks)
    np.testing.assert_array_equal(v.items, items)
    np.testing.assert_array_equal(v.creatures, mobs)
    dr, dc = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][int(s.player_direction)]
    assert int(v.facing_block) == blocks[1 + dr, 1 + dc]
    assert int(v.facing_item) == items[1 + dr, 1 + dc]
    np.testing.assert_array_equal(v.facing_creature, mobs[2 + dr, 2 + dc])


@pytest.fixture(scope="module")
def random_craftax(craftax: Game) -> list[Any]:
    """100 random Craftax states."""
    rng = np.random.default_rng(0)
    return [_random_craftax(craftax.start(), rng) for _ in range(100)]


def test_decode_obs_equals_the_state(craftax: Game, random_craftax: list[Any]) -> None:
    for s in random_craftax:
        v, i, floor = craftax.view(s), s.inventory, int(s.player_level)
        want = [
            i.wood, i.stone, i.coal, i.iron, i.diamond, i.sapphire, i.ruby,
            i.sapling, i.arrows, i.torches, i.books, jnp.sum(i.potions),
        ]  # fmt: skip
        np.testing.assert_array_equal(v.materials, want)
        np.testing.assert_array_equal(v.potions, i.potions)
        assert v.materials.dtype == jnp.int32
        assert (int(v.pickaxe), int(v.sword), int(v.bow)) == (i.pickaxe, i.sword, i.bow)
        np.testing.assert_array_equal(v.armour, i.armour)
        assert int(v.ench_sword) == s.sword_enchantment
        assert int(v.ench_bow) == s.bow_enchantment
        np.testing.assert_array_equal(v.ench_armour, s.armour_enchantments)
        assert float(v.health) == pytest.approx(float(s.player_health), abs=1e-5)
        stats = ("food", "drink", "energy", "mana", "xp", "dexterity", "strength")
        for name in (*stats, "intelligence"):
            assert int(getattr(v, name)) == int(getattr(s, f"player_{name}")), name
        assert int(v.floor) == floor
        assert bool(v.ladder_open) == (int(s.monsters_killed[floor]) >= 8)
        assert bool(v.sleeping) == bool(s.is_sleeping)
        np.testing.assert_array_equal(v.spells, s.learned_spells)
        assert int(v.direction) == int(s.player_direction)
        _check_tiles(v, s, CRAFTAX_GROUPS, floor)


@pytest.mark.parametrize("granularity", ["recipe", "log2"])
def test_key_fields_equal_the_progress_fields(
    craftax: Game, random_craftax: list[Any], granularity: str
) -> None:
    """The top-level key holds nothing the observation lacks."""
    for s in random_craftax:
        got = key_fields(craftax.view(s), granularity)
        assert got.dtype == jnp.int32
        np.testing.assert_array_equal(got, fields(unwrap(s), granularity))


def _random_classic(base: Any, rng: np.random.Generator) -> Any:
    def ints(hi: int, lo: int = 0) -> Any:
        return jnp.int32(rng.integers(lo, hi + 1))

    names = ("wood", "stone", "coal", "iron", "diamond", "sapling")
    tools = ("pickaxe", "sword")
    tool_names = [f"{t}_{k}" for k in tools for t in ("wood", "stone", "iron")]
    p = rng.integers(2, 62, 2)
    s = inv(base, **{n: ints(9) for n in (*names, *tool_names)})
    r0, c0 = p - 1
    s = s.replace(
        player_position=jnp.asarray(p, jnp.int32),
        player_direction=ints(4, 1),
        player_health=ints(9), player_food=ints(9), player_drink=ints(9),
        player_energy=ints(9), is_sleeping=jnp.bool_(rng.integers(2)),
        map=s.map.at[r0 : r0 + 3, c0 : c0 + 3].set(
            jnp.asarray(rng.integers(0, 17, (3, 3)), jnp.int32)
        ),
    )  # fmt: skip
    for slot, g in enumerate(rng.choice(CLASSIC_GROUPS, 2)):
        m = getattr(s, g)
        at = jnp.asarray(p + rng.integers(-2, 3, 2), jnp.int32)
        m = m.replace(
            position=m.position.at[slot].set(at), mask=m.mask.at[slot].set(True)
        )
        s = s.replace(**{g: m})
    return s


def test_classic_decoder_and_key(classic: Game) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = _random_classic(classic.start(), rng)
        v, i = classic.view(s), s.inventory
        want = [i.wood, i.stone, i.coal, i.iron, i.diamond, 0, 0, i.sapling]
        np.testing.assert_array_equal(v.materials, [*want, 0, 0, 0, 0])
        tools = [i.wood_pickaxe, i.stone_pickaxe, i.iron_pickaxe]
        tools += [i.wood_sword, i.stone_sword, i.iron_sword]
        np.testing.assert_array_equal(v.tools, tools)
        assert float(v.health) == int(s.player_health)
        for name in ("food", "drink", "energy"):
            assert int(getattr(v, name)) == int(getattr(s, f"player_{name}"))
        assert int(v.floor) == 0
        assert bool(v.sleeping) == bool(s.is_sleeping)
        _check_tiles(v, s, CLASSIC_GROUPS, None)
        for g in ("recipe", "log2"):
            np.testing.assert_array_equal(key_fields(v, g), fields(s, g))


def test_decode_obs_refuses_other_sizes() -> None:
    with pytest.raises(ValueError, match="observation"):
        decode_obs(jnp.zeros(100))


# --- reward and success tests ---


def test_achievement_reward_removes_the_health_part(craftax: Game) -> None:
    s = craftax.start()
    hit = craftax.view(s.replace(player_health=jnp.float32(7.0)))
    full = craftax.view(s)
    eight = craftax.view(s.replace(player_health=jnp.float32(8.0)))
    assert float(achievement_reward(0.8, full, hit)) == pytest.approx(1.0, abs=1e-5)
    assert float(achievement_reward(0.1, eight, full)) == pytest.approx(0.0, abs=1e-5)


def test_collect_wood_needs_the_next_bucket(craftax: Game) -> None:
    s = block(craftax.start(), UP, BlockType.TREE)
    s2, r, _ = craftax.play(s, Action.DO.value)
    assert int(s2.inventory.wood) == 1
    a, b = craftax.view(s), craftax.view(s2)
    assert step("collect_wood", a, b, Action.DO.value, r)[:2] == (True, True)
    one, two = craftax.view(inv(s, wood=1)), craftax.view(inv(s, wood=2))
    assert not step("collect_wood", one, two, Action.DO.value, granularity="log2")[0]
    assert step("collect_wood", one, two, Action.DO.value)[0]
    assert can("collect_wood", a)
    assert not can("collect_wood", craftax.view(inv(s, wood=4)))
    assert can("collect_wood", craftax.view(inv(s, wood=4)), "log2")


def test_make_wood_pickaxe_needs_a_table_nearby(craftax: Game) -> None:
    s = inv(craftax.start(), wood=1)
    near = block(s, (23, 23), BlockType.CRAFTING_TABLE)
    for start, made in ((near, True), (s, False)):
        s2, r, _ = craftax.play(start, Action.MAKE_WOOD_PICKAXE.value)
        a, b = craftax.view(start), craftax.view(s2)
        assert can("make_wood_pickaxe", a)
        got = step(
            "make_wood_pickaxe", a, b, Action.MAKE_WOOD_PICKAXE.value, r, steps=5
        )
        assert got == ((True, True, "success") if made else (False, False, "running"))


def test_place_furnace_needs_a_table_in_the_next_view(craftax: Game) -> None:
    s = block(inv(craftax.start(), stone=1), UP, BlockType.GRASS)
    a = craftax.view(s)
    placed = block(inv(s, stone=0), UP, BlockType.FURNACE)
    with_table = block(placed, (25, 25), BlockType.CRAFTING_TABLE)
    far_table = block(placed, (26, 26), BlockType.CRAFTING_TABLE)
    act = Action.PLACE_FURNACE.value
    assert step("place_furnace", a, craftax.view(with_table), act)[0]
    assert not step("place_furnace", a, craftax.view(far_table), act)[0]
    s2, r, _ = craftax.play(block(s, (25, 25), BlockType.CRAFTING_TABLE), act)
    a2 = craftax.view(block(s, (25, 25), BlockType.CRAFTING_TABLE))
    assert step("place_furnace", a2, craftax.view(s2), act, r)[0]


def test_place_table_counts_only_a_placement(craftax: Game) -> None:
    s = block(inv(craftax.start(), wood=2), UP, BlockType.GRASS)
    s2, r, _ = craftax.play(s, Action.PLACE_TABLE.value)
    a, b = craftax.view(s), craftax.view(s2)
    assert step("place_table", a, b, Action.PLACE_TABLE.value, r)[0]
    assert not step("place_table", a, b, Action.NOOP.value, r)[0]


def test_descend_on_an_open_ladder(craftax: Game) -> None:
    s = craftax.start()
    assert can("descend", craftax.view(s))  # floor 0 starts open
    down = s.replace(player_level=jnp.int32(1), player_position=s.down_ladders[1])
    closed = down.replace(monsters_killed=down.monsters_killed.at[1].set(7))
    opened = down.replace(monsters_killed=down.monsters_killed.at[1].set(8))
    assert not can("descend", craftax.view(closed))
    a = craftax.view(opened)
    assert can("descend", a)
    s2, r, _ = craftax.play(opened, Action.DESCEND.value)
    assert int(s2.player_level) == 2
    assert step("descend", a, craftax.view(s2), Action.DESCEND.value, r)[0]


def test_drink_to_max_and_sleep_to_max_energy(craftax: Game) -> None:
    s = craftax.start()

    def v(**stats: Any) -> ObsView:
        return craftax.view(s.replace(**stats))

    assert step("drink", v(player_drink=8), v(player_drink=9))[0]
    assert step("drink", v(player_drink=7), v(player_drink=8))[:2] == (False, False)
    asleep = v(is_sleeping=True, player_energy=8)
    assert step("sleep", asleep, v(player_energy=9)) == (True, True, "success")
    woken = v(player_energy=5)
    assert step("sleep", v(is_sleeping=True, player_energy=5), woken) == (
        False, True, "failure",
    )  # fmt: skip


def test_fight_needs_a_hostile_gone_from_the_neighbourhood(craftax: Game) -> None:
    s = craftax.start()
    do = Action.DO.value
    zombie = mob(s, "melee_mobs", UP)
    cow = mob(s, "passive_mobs", UP)
    gone = craftax.view(s)
    assert step("fight", craftax.view(zombie), gone, do) == (True, True, "success")
    assert not step("fight", craftax.view(cow), gone, do)[0]
    assert not step("fight", craftax.view(zombie), gone, Action.NOOP.value)[0]
    near = craftax.view(mob(s, "melee_mobs", (24, 25)))
    assert not step("fight", craftax.view(zombie), near, do)[0]
    assert step("fight", craftax.view(zombie), near, do, r=1.0)[0]


def test_fight_looks_for_the_attacked_mob_in_the_5_by_5(craftax: Game) -> None:
    """A mob moves at most one tile a step, so one still within 2 tiles survived."""
    s, do = craftax.start(), Action.DO.value
    skeleton = craftax.view(mob(s, "ranged_mobs", UP))
    away = craftax.view(mob(s, "ranged_mobs", (22, 24)))  # stepped back, 2 tiles off
    assert not step("fight", skeleton, away, do)[0]
    assert step("fight", skeleton, craftax.view(s), do)[0]
    weak = mob(s, "ranged_mobs", UP, health=1.0)
    s2, r, _ = craftax.play(weak, do)
    assert not bool(s2.ranged_mobs.mask[0, 0])
    assert step("fight", craftax.view(weak), craftax.view(s2), do, r) == (
        True, True, "success",
    )  # fmt: skip


def test_classic_fight_and_tools(classic: Game) -> None:
    """Classic: zombies and skeletons are hostile, cows not; each tool is one bit."""
    s, do = classic.start(), Action.DO.value

    def at_front(group: str) -> ObsView:
        m = getattr(s, group)
        m = m.replace(
            position=m.position.at[0].set(jnp.array([31, 32])),
            mask=m.mask.at[0].set(True),
        )
        return classic.view(s.replace(**{group: m}))

    gone = classic.view(s)
    assert step("fight", at_front("zombies"), gone, do)[0]
    assert step("fight", at_front("skeletons"), gone, do)[0]
    assert not step("fight", at_front("cows"), gone, do)[0]
    iron = classic.view(inv(s, iron_pickaxe=1))
    both = classic.view(inv(s, iron_pickaxe=1, stone_pickaxe=1))
    assert step("make_stone_pickaxe", iron, both)[0]
    assert not step("make_iron_pickaxe", iron, both)[0]


def test_open_chest_and_enchant(craftax: Game) -> None:
    s = craftax.start()
    chest = craftax.view(block(s, UP, BlockType.CHEST))
    path = craftax.view(block(s, UP, BlockType.PATH))
    assert step("open_chest", chest, path, Action.DO.value)[0]
    assert not step("open_chest", chest, chest, Action.DO.value)[0]
    vault = s.replace(player_level=jnp.int32(4), player_mana=jnp.int32(9))
    vault = block(inv(vault, sword=1, ruby=1), UP, BlockType.ENCHANTMENT_TABLE_FIRE, 4)
    a = craftax.view(vault)
    assert can("enchant", a)
    assert not can("enchant", craftax.view(inv(vault, ruby=0)))
    fire = vault.replace(sword_enchantment=jnp.int32(1))
    b = craftax.view(fire)
    assert step("enchant", a, b, Action.ENCHANT_SWORD.value)[0]
    assert not can("enchant", b)  # every enchantable item already fire
    helmet = inv(fire, armour=fire.inventory.armour.at[0].set(1))
    assert can("enchant", craftax.view(helmet))
    all_fire = helmet.replace(armour_enchantments=jnp.ones(4, jnp.int32))
    assert not can("enchant", craftax.view(all_fire))


def test_unlock_any_reads_the_achievement_part(craftax: Game) -> None:
    s = craftax.start()
    a, healed = craftax.view(s.replace(player_health=jnp.float32(8.0))), craftax.view(s)
    assert step("unlock_any", a, a, r=1.0)[0]
    assert not step("unlock_any", a, healed, r=0.1)[0]


# --- common ends ---


def test_step_limit_scales_with_the_timeout_multiplier(craftax: Game) -> None:
    a = craftax.view(craftax.start())
    assert step("place_table", a, a, steps=98) == (False, False, "running")
    assert step("place_table", a, a, steps=99) == (False, True, "timeout")
    assert step("place_table", a, a, steps=99, mult=2.0)[1] is False
    assert step("place_table", a, a, steps=49, mult=0.5)[2] == "timeout"
    assert step("read_book", a, a, mult=2.0)[1]


def test_done_ends_with_episode_end_and_no_success(craftax: Game) -> None:
    s = block(craftax.start(), UP, BlockType.TREE)
    reset = craftax.view(inv(s, wood=3))  # the reset observation shows more wood
    got = step("collect_wood", craftax.view(s), reset, Action.DO.value, 1.0, done=True)
    assert got == (False, True, "episode_end")


def test_floor_change_and_low_stats_interrupt(craftax: Game) -> None:
    s = inv(craftax.start(), pickaxe=2)
    a = craftax.view(s)
    below = craftax.view(s.replace(player_level=jnp.int32(1)))
    assert step("collect_stone", a, below) == (False, True, "floor_change")
    four = craftax.view(s.replace(player_food=4))
    three = craftax.view(s.replace(player_food=3))
    assert step("collect_iron", four, three) == (False, True, "low_stat")
    assert step("eat", four, three) == (False, False, "running")
    assert step("collect_iron", three, three)[1] is False  # already low at the start
    np.testing.assert_array_equal(low_stats(three), [True, False, False, False])


# --- primitive options and masks ---


def test_primitive_options_have_fixed_actions(craftax: Game) -> None:
    s = craftax.start()

    def level_up(dex: int, st: int, it: int) -> int:
        v = craftax.view(
            s.replace(
                player_dexterity=jnp.int32(dex),
                player_strength=jnp.int32(st),
                player_intelligence=jnp.int32(it),
            )
        )
        return int(primitive_action(SKILL_IDS["level_up"], v))

    v = craftax.view(s)
    assert int(primitive_action(SKILL_IDS["read_book"], v)) == Action.READ_BOOK.value
    assert level_up(1, 1, 1) == Action.LEVEL_UP_DEXTERITY.value
    assert level_up(2, 1, 1) == Action.LEVEL_UP_STRENGTH.value
    assert level_up(2, 2, 1) == Action.LEVEL_UP_INTELLIGENCE.value
    assert level_up(3, 2, 2) == Action.LEVEL_UP_STRENGTH.value
    assert int(primitive_action(SKILL_IDS["collect_wood"], v)) == -1


def test_masks(craftax: Game, random_craftax: list[Any]) -> None:
    s = inv(craftax.start(), wood=1, stone=1, iron=1, coal=0, pickaxe=2)
    assert not can("make_iron_pickaxe", craftax.view(s))
    assert can("make_iron_pickaxe", craftax.view(inv(s, coal=1)))
    for st in random_craftax:
        m = available(craftax.view(st))
        assert m.shape == (25,)
        assert m.dtype == jnp.bool_
        assert m[SKILL_IDS["fight"]]
        assert m[SKILL_IDS["unlock_any"]]
    books = craftax.view(inv(s, books=1))
    assert can("read_book", books)
    assert not can("level_up", books)
    assert can("level_up", craftax.view(s.replace(player_xp=jnp.int32(1))))


def test_classic_mask_has_sixteen_skills(classic: Game) -> None:
    s = classic.start().replace(player_food=5, player_drink=5, player_energy=5)
    rich = inv(s, wood=2, stone=1, coal=1, iron=1)
    tooled = inv(rich, wood_pickaxe=1, stone_pickaxe=1, iron_pickaxe=1)
    rng = np.random.default_rng(2)
    states = [rich, tooled] + [_random_classic(s, rng) for _ in range(30)]
    masks = np.stack([np.asarray(available(classic.view(x))) for x in states])
    ever = {SKILLS[k].name for k in np.flatnonzero(masks.any(0))}
    assert len(ever) == 16
    assert ever.isdisjoint(CRAFTAX_ONLY)
    assert len(CRAFTAX_ONLY) == 9
    assert not can("collect_stone", classic.view(inv(rich, stone_pickaxe=1)))
    assert can("collect_stone", classic.view(inv(rich, wood_pickaxe=1)))


# --- access and batching ---


def test_no_privileged_input() -> None:
    for fn in (available, skill_step, primitive_action, decode_obs, key_fields):
        params = inspect.signature(fn).parameters
        assert not {"state", "env_state", "events"} & set(params), fn.__name__
    assert list(inspect.signature(skills.test_agreement).parameters) == [
        "skill", "success", "events",
    ]  # fmt: skip


def test_importing_skills_leaves_craftax_unimported() -> None:
    """The runner may import the module before ``configure_runtime``."""
    code = (
        "import sys, wut_murl_craftax.training.skills\n"
        "print(any(m.startswith('craftax') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_agreement_with_step_events(craftax: Game) -> None:
    s = craftax.start()
    do = Action.DO.value
    weak = mob(s, "melee_mobs", UP)
    weak = weak.replace(melee_mobs=weak.melee_mobs.replace(
        health=weak.melee_mobs.health.at[0, 0].set(1.0)
    ))  # fmt: skip
    m = weak.melee_mobs  # one step up into a dark tile: unseen, not killed
    walked = weak.replace(
        melee_mobs=m.replace(position=m.position.at[0, 0, 0].set(22)),
        light_map=weak.light_map.at[0, 22, 24].set(0.0),
    )
    fight = SKILL_IDS["fight"]
    for s2 in (craftax.play(weak, do)[0], walked):
        ev = step_events("craftax", weak, do, s2)
        a, b = craftax.view(weak), craftax.view(s2)
        success = skill_step(fight, a, do, 0.0, b, False, 0, 1.0)[0]
        assert bool(success)
        checked, agree = skills.test_agreement(fight, success, ev)
        assert bool(checked)
        assert bool(agree) == (s2 is not walked)
    ev = step_events("craftax", s, Action.NOOP.value, s)
    checked, _ = skills.test_agreement(SKILL_IDS["drink"], jnp.bool_(False), ev)
    assert not bool(checked)


def _batch(views: list[ObsView]) -> ObsView:
    return jax.tree.map(lambda *x: jnp.stack(x), *views)


def test_vmap_matches_a_loop_and_jit_compiles_once(
    craftax: Game, random_craftax: list[Any]
) -> None:
    """All 25 skill ids batched on real (state, next state) pairs, five of them
    scripted successes."""
    acts = np.random.default_rng(3).integers(0, 17, NUM_SKILLS)
    pairs = [
        craftax.play(x, int(k)) for x, k in zip(random_craftax, acts, strict=False)
    ]
    a = [craftax.view(x) for x in random_craftax[:NUM_SKILLS]]
    b = [craftax.view(s2) for s2, _, _ in pairs]
    r = [rr for _, rr, _ in pairs]
    done = [d for _, _, d in pairs]
    s = craftax.start()
    tree = block(s, UP, BlockType.TREE)
    grass = block(inv(s, wood=2), UP, BlockType.GRASS)
    wins = {
        "collect_wood": (tree, Action.DO.value),
        "place_table": (grass, Action.PLACE_TABLE.value),
        "fight": (mob(s, "melee_mobs", UP, health=1.0), Action.DO.value),
        "drink": (s.replace(player_drink=8), Action.NOOP.value),
        "unlock_any": (grass, Action.PLACE_TABLE.value),
    }
    for name, (x, k) in wins.items():
        i, (x2, rr, d) = SKILL_IDS[name], craftax.play(x, k)
        if name == "drink":
            x2 = x.replace(player_drink=9)
        a[i], b[i], r[i], done[i], acts[i] = craftax.view(x), craftax.view(x2), rr, d, k
    ids = jnp.arange(NUM_SKILLS, dtype=jnp.int32)
    steps = jnp.arange(NUM_SKILLS) * 20

    def run(k: Any, va: Any, act: Any, rr: Any, vb: Any, d: Any, n: Any) -> Any:
        return skill_step(k, va, act, rr, vb, d, n, 1.0)

    args = (jnp.asarray(acts), jnp.asarray(r), _batch(b), jnp.asarray(done), steps)
    batched = jax.vmap(run)(ids, _batch(a), *args)
    for i in range(NUM_SKILLS):
        one = run(ids[i], a[i], acts[i], r[i], b[i], done[i], steps[i])
        for x, y in zip(batched, one, strict=True):
            assert bool(x[i] == y), (SKILLS[i].name, x[i], y)
    reasons = [REASONS[int(k)] for k in batched[2]]
    assert all(reasons[SKILL_IDS[n]] == "success" for n in wins), reasons
    assert reasons.count("floor_change") < NUM_SKILLS // 2, reasons
    np.testing.assert_array_equal(
        jax.vmap(available)(_batch(a)), np.stack([available(x) for x in a])
    )
    traces: list[int] = []

    def counted(k: Any, va: Any, vb: Any) -> Any:
        traces.append(1)
        return skill_step(k, va, 5, 0.0, vb, False, 0, 1.0)

    jitted: Callable[..., Any] = jax.jit(counted)
    for k in range(NUM_SKILLS):
        jitted(jnp.int32(k), a[0], b[0])
    assert len(traces) == 1
