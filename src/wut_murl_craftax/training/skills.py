"""Skill library of the skill methods: observation decoder, key, masks, skill ends.

`decode_obs` turns one symbolic observation (Craftax or Craftax-Classic) into an
`ObsView` of exact integers; `key_fields` rebuilds `progress_key.fields` from it,
`available` holds the start conditions and `skill_step` the success tests (one
``jax.lax.switch``) and the common ends. Acting reads only observations, the action
and the reward; `test_agreement` alone reads ``StepEvents``, for logging. Pure, for
``jit`` and ``vmap``. No Craftax import: its constants start the JAX backends, so the
block and action ids below are literals the tests check against its enums, and
achievement names are looked up only when `test_agreement` is traced (after
``configure_runtime``). Rules and skill table: [[Craftax skill library]].
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ..evaluation.events import StepEvents
from ..evaluation.milestones import Game, achievement_names
from .progress_key import CLASSIC_MATERIALS, MATERIALS, Granularity, bucket

type Kind = Literal["learned", "primitive"]
"""A skill run by the shared network, or a one-step option with a fixed action."""


class Skill(NamedTuple):
    """One option of the library."""

    name: str
    t_max: int
    """Step limit before the timeout multiplier."""
    kind: Kind


SKILLS: tuple[Skill, ...] = tuple(
    Skill(n, t, "primitive" if t == 1 else "learned")
    for n, t in (
        ("collect_wood", 300), ("collect_stone", 300), ("collect_coal", 300),
        ("collect_iron", 300), ("collect_diamond", 500), ("collect_gem", 500),
        ("place_table", 100), ("place_furnace", 100), ("make_wood_pickaxe", 200),
        ("make_stone_pickaxe", 200), ("make_iron_pickaxe", 200),
        ("make_diamond_pickaxe", 200), ("upgrade_sword", 200), ("make_armour", 200),
        ("make_torches", 200), ("drink", 200), ("eat", 300), ("sleep", 400),
        ("fight", 300), ("descend", 1000), ("open_chest", 500), ("enchant", 500),
        ("unlock_any", 500), ("read_book", 1), ("level_up", 1),
    )
)  # fmt: skip
"""The 25 options in table order; their index is the skill id."""
NUM_SKILLS = len(SKILLS)
"""Columns of the top-level table."""
SKILL_IDS: dict[str, int] = {s.name: i for i, s in enumerate(SKILLS)}
"""Skill name -> id."""
T_MAX = np.array([s.t_max for s in SKILLS], np.int32)
"""Step limit per skill id."""
PRIMITIVE = np.array([s.kind == "primitive" for s in SKILLS])
"""Whether a skill id is a one-step primitive option."""
CRAFTAX_ONLY = (
    "collect_gem", "make_diamond_pickaxe", "make_armour", "make_torches",
    "descend", "open_chest", "enchant", "read_book", "level_up",
)  # fmt: skip
"""Skills Craftax-Classic lacks (no gems, diamond tools, armour, floors, magic, XP)."""
REASONS = (
    "running", "success", "failure", "timeout", "episode_end", "floor_change",
    "low_stat",
)  # fmt: skip
"""Why a skill ended (index returned by `skill_step`); ``running``: it did not."""
RUNNING, SUCCESS, FAILURE, TIMEOUT, EPISODE_END, FLOOR_CHANGE, LOW_STAT = range(7)
"""Indices of `REASONS`."""
LOW_STAT_LEVEL = 3
"""Food, drink, energy or health at or below this sets a low-stat flag."""
HEALTH_EPS = 1e-3
"""Float slack of the health flag: Craftax health is a float read back as 10 x."""
UNLOCK_MIN = 0.5
"""Achievement part of the reward that counts as an unlock (smallest weight 1)."""
TREE, PATH, CRAFTING_TABLE, FURNACE, CHEST = 5, 7, 11, 12, 23
NECROMANCER, NECROMANCER_VULNERABLE = 32, 36
"""``BlockType`` values (Classic shares 0-16)."""
_BOSS = (NECROMANCER, NECROMANCER_VULNERABLE)
NOOP, DO, PLACE_TABLE, PLACE_FURNACE, READ_BOOK = 0, 5, 8, 9, 35
LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH, LEVEL_UP_INTELLIGENCE = 39, 40, 41
"""``Action`` values (Classic shares 0-16)."""
DIRECTIONS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], np.int32)
"""Row, column offset per ``player_direction`` (1 left, 2 right, 3 up, 4 down)."""
HOSTILE = np.isin(np.arange(40), [*range(8), *range(16, 24)])
"""Creature channels (``class * 8 + type``) of melee and ranged mobs."""
CLASSIC_CREATURES = np.array([0, 8, 16, 24])
"""Craftax channels of the Classic zombie, cow, skeleton and arrow flags."""
CRAFTAX_OBS, CLASSIC_OBS = 8268, 1345
"""Symbolic observation sizes."""
_LOW_STAT_EXEMPT = np.array([SKILL_IDS[n] for n in ("drink", "eat", "sleep")])
_UNLOCKS: dict[str, tuple[str, ...]] = {
    **{f"collect_{m}": (f"collect_{m}",) for m in ("wood", "stone", "coal", "iron")},
    "collect_diamond": ("collect_diamond",),
    "collect_gem": ("collect_sapphire", "collect_ruby"),
    "place_table": ("place_table",), "place_furnace": ("place_furnace",),
    **{f"make_{t}_pickaxe": (f"make_{t}_pickaxe",) for t in ("wood", "stone", "iron")},
    "make_diamond_pickaxe": ("make_diamond_pickaxe",),
    "upgrade_sword": ("make_wood_sword", "make_stone_sword", "make_iron_sword",
                      "make_diamond_sword"),
    "make_armour": ("make_iron_armour", "make_diamond_armour"),
    "make_torches": ("make_torch",), "sleep": ("wake_up",),
    "fight": ("defeat_", "damage_necromancer"), "descend": ("enter_",),
    "open_chest": ("open_chest",), "enchant": ("enchant_",), "read_book": ("learn_",),
}  # fmt: skip
"""Achievement-name prefixes whose first unlock means the skill succeeded."""
_CRAFTAX_ORDER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])
"""Observation slots of the first ten `MATERIALS` (it stores torches, arrows)."""
_FULL_EVENT = np.array([SKILL_IDS[n] for n in ("fight", "eat", "unlock_any")])


class ObsView(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """What one observation shows, as exact integers; per env (``vmap`` for batches).

    Classic leaves the Craftax-only fields at 0 (``False``) and keeps its six tools
    in ``tools``; tiles are the 3 x 3 around the player, the player in the centre,
    blank (block 0, item 0, no creature) where the tile is too dark to see.
    """

    materials: jax.Array
    """`progress_key.MATERIALS` counts, int32[12]; potions summed over colours."""
    potions: jax.Array
    """Potions per colour, int32[6]."""
    tools: jax.Array
    """Classic wood, stone, iron pickaxe and sword counts, int32[6] (Craftax 0)."""
    pickaxe: jax.Array
    sword: jax.Array
    bow: jax.Array
    armour: jax.Array
    """Helmet, chestplate, leggings, boots: 0 none, 1 iron, 2 diamond; int32[4]."""
    ench_sword: jax.Array
    ench_bow: jax.Array
    ench_armour: jax.Array
    """Enchantments: 0 none, 1 fire, 2 ice; armour int32[4]."""
    health: jax.Array
    """Health, float32 (Craftax health is fractional)."""
    food: jax.Array
    drink: jax.Array
    energy: jax.Array
    mana: jax.Array
    xp: jax.Array
    dexterity: jax.Array
    strength: jax.Array
    intelligence: jax.Array
    floor: jax.Array
    ladder_open: jax.Array
    """8 kills on this floor (floor 0 starts open), bool."""
    sleeping: jax.Array
    spells: jax.Array
    """Fireball, iceball learned, bool[2]."""
    direction: jax.Array
    """Facing direction 1-4 (`DIRECTIONS`)."""
    blocks: jax.Array
    items: jax.Array
    """``BlockType`` and ``ItemType`` of the 3 x 3 tiles, int32[3, 3]."""
    creatures: jax.Array
    """Creature channels of the 3 x 3 tiles, bool[3, 3, 40]."""
    classic: bool = struct.field(pytree_node=False, default=False)  # type: ignore[no-untyped-call]  # flax API

    def _front(self) -> tuple[jax.Array, jax.Array]:
        d = jnp.asarray(DIRECTIONS)[self.direction]
        return 1 + d[0], 1 + d[1]

    @property
    def facing_block(self) -> jax.Array:
        """Block of the faced tile."""
        return self.blocks[self._front()]

    @property
    def facing_item(self) -> jax.Array:
        """Item of the faced tile."""
        return self.items[self._front()]

    @property
    def facing_creature(self) -> jax.Array:
        """Creature channels of the faced tile, bool[40]."""
        return self.creatures[self._front()]


def _int(x: jax.Array) -> jax.Array:
    return jnp.round(x).astype(jnp.int32)


def _count(x: jax.Array) -> jax.Array:
    """Counts encoded as ``sqrt(n) / 10``."""
    return _int(100 * x * x)


def _decode_craftax(obs: jax.Array) -> ObsView:
    tiles = obs[:8217].reshape(9, 11, 83)[3:6, 4:7]
    inv, stats, special = obs[8217:8233], obs[8239:8248], obs[8260:8268]
    counts, potions = _count(inv[:10]), _count(obs[8233:8239])
    books = _int(2 * inv[10])
    tail = jnp.stack([books, potions.sum()])
    materials = jnp.concatenate([counts[_CRAFTAX_ORDER], tail])
    ints = _int(10 * stats[1:])
    return ObsView(
        materials=materials,
        potions=potions,
        tools=jnp.zeros(6, jnp.int32),
        pickaxe=_int(4 * inv[11]),
        sword=_int(4 * inv[12]),
        bow=_int(inv[15]),
        armour=_int(2 * obs[8252:8256]),
        ench_sword=_int(inv[13]),
        ench_bow=_int(inv[14]),
        ench_armour=_int(obs[8256:8260]),
        health=10 * stats[0],
        food=ints[0], drink=ints[1], energy=ints[2], mana=ints[3], xp=ints[4],
        dexterity=ints[5], strength=ints[6], intelligence=ints[7],
        floor=_int(10 * special[5]),
        ladder_open=special[6] > 0.5,
        sleeping=special[1] > 0.5,
        spells=special[3:5] > 0.5,
        direction=jnp.argmax(obs[8248:8252]).astype(jnp.int32) + 1,
        blocks=jnp.argmax(tiles[..., :37], -1).astype(jnp.int32),
        items=jnp.argmax(tiles[..., 37:42], -1).astype(jnp.int32),
        creatures=tiles[..., 42:82] > 0.5,
    )  # fmt: skip


def _decode_classic(obs: jax.Array) -> ObsView:
    tiles = obs[:1323].reshape(7, 9, 21)[2:5, 3:6]
    inv, stats = _int(10 * obs[1323:1335]), _int(10 * obs[1335:1339])
    where = np.array([MATERIALS.index(n) for n in CLASSIC_MATERIALS])
    zero, zeros4 = jnp.int32(0), jnp.zeros(4, jnp.int32)
    creatures = (
        jnp
        .zeros((3, 3, 40), bool)
        .at[..., CLASSIC_CREATURES]
        .set(tiles[..., 17:21] > 0.5)
    )
    return ObsView(
        materials=jnp.zeros(len(MATERIALS), jnp.int32).at[where].set(inv[:6]),
        potions=jnp.zeros(6, jnp.int32),
        tools=inv[6:],
        pickaxe=zero, sword=zero, bow=zero, armour=zeros4,
        ench_sword=zero, ench_bow=zero, ench_armour=zeros4,
        health=stats[0].astype(jnp.float32),
        food=stats[1], drink=stats[2], energy=stats[3],
        mana=zero, xp=zero, dexterity=zero, strength=zero, intelligence=zero,
        floor=zero, ladder_open=jnp.bool_(False), sleeping=obs[1344] > 0.5,
        spells=jnp.zeros(2, bool),
        direction=jnp.argmax(obs[1339:1343]).astype(jnp.int32) + 1,
        blocks=jnp.argmax(tiles[..., :17], -1).astype(jnp.int32),
        items=jnp.zeros((3, 3), jnp.int32),
        creatures=creatures,
        classic=True,
    )  # fmt: skip


def decode_obs(obs: jax.Array) -> ObsView:
    """View of one symbolic observation; the game is told by its size.

    Raises:
        ValueError: On a size that is neither `CRAFTAX_OBS` nor `CLASSIC_OBS`.
    """
    match obs.shape[-1]:
        case 8268:
            return _decode_craftax(obs)
        case 1345:
            return _decode_classic(obs)
        case n:
            msg = (
                f"unknown observation size {n}; expected {CRAFTAX_OBS} or {CLASSIC_OBS}"
            )
            raise ValueError(msg)


def _n(v: ObsView, name: str) -> jax.Array:
    return v.materials[MATERIALS.index(name)]


def key_fields(view: ObsView, granularity: Granularity = "recipe") -> jax.Array:
    """`progress_key.fields` of the observed state, int32 (Classic `CLASSIC_FIELDS`)."""

    def mats(names: tuple[str, ...]) -> jax.Array:
        return jnp.stack([bucket(n, _n(view, n), granularity) for n in names])

    if view.classic:
        return jnp.concatenate([mats(CLASSIC_MATERIALS), jnp.minimum(view.tools, 1)])
    enchanted = jnp.stack([
        view.ench_sword > 0,
        view.ench_bow > 0,
        jnp.any(view.ench_armour > 0),
    ]) & (granularity != "tiers")
    head = jnp.stack([
        view.floor, view.ladder_open, view.pickaxe, view.sword,
        jnp.minimum(view.bow, 1), jnp.sum(view.armour),
    ])  # fmt: skip
    parts = [head, view.spells, enchanted, mats(MATERIALS)]
    return jnp.concatenate([p.astype(jnp.int32) for p in parts])


def achievement_reward(
    r: jax.Array | float, view: ObsView, view2: ObsView
) -> jax.Array:
    """Unlock part of the step reward: ``r`` minus 0.1 x the observed health change."""
    return jnp.asarray(r, jnp.float32) - 0.1 * (view2.health - view.health)


def _max_stats(v: ObsView) -> tuple[jax.Array | int, jax.Array | int]:
    """Max health and max food, drink, energy (Craftax: from the attributes)."""
    return (9, 9) if v.classic else (8 + v.strength, 7 + 2 * v.dexterity)


def low_stats(view: ObsView) -> jax.Array:
    """Low-stat flags of food, drink, energy, health, bool[4]."""
    s = jnp.stack([view.food, view.drink, view.energy]) <= LOW_STAT_LEVEL
    return jnp.append(s, view.health <= LOW_STAT_LEVEL + HEALTH_EPS)


def _has(v: ObsView, tool: int) -> jax.Array:
    """Tool 0-5 (wood, stone, iron pickaxe; wood, stone, iron sword) held."""
    if v.classic:  # one inventory slot per tool, no tiers
        return v.tools[tool] > 0
    return (v.pickaxe if tool < 3 else v.sword) >= tool % 3 + 1


def _sword_rank(v: ObsView) -> jax.Array:
    return jnp.sum(v.tools[3:] > 0) if v.classic else v.sword


def _on(v: ObsView, floors: tuple[int, ...]) -> jax.Array:
    return jnp.isin(v.floor, jnp.array(floors))


def _below_cap(v: ObsView, name: str, g: Granularity) -> jax.Array:
    """The material's bucket can still rise (inventory caps: Classic 9, Craftax 99)."""
    return bucket(name, _n(v, name), g) < bucket(name, 9 if v.classic else 99, g)


def available(view: ObsView, granularity: Granularity = "recipe") -> jax.Array:
    """Start condition per skill id, bool[25]; ``fight`` and ``unlock_any`` always.

    A collect skill needs its material's ``granularity`` bucket below the top one.
    """
    v, g = view, granularity
    wood, stone, coal, iron = (_n(v, n) >= 1 for n in ("wood", "stone", "coal", "iron"))
    diamond, top = _n(v, "diamond"), _max_stats(v)[1]
    sword_tiers = (
        (wood & ~_has(v, 3))
        | (wood & stone & ~_has(v, 4))
        | (wood & stone & coal & iron & ~_has(v, 5))
        | (wood & (diamond >= 2) & (v.sword < 4) & (not v.classic))
    )
    gem = ((_n(v, "sapphire") >= 1) & (v.floor == 3)) | (
        (_n(v, "ruby") >= 1) & (v.floor == 4)
    )
    armour = ((_n(v, "iron") >= 3) & (_n(v, "coal") >= 3) & jnp.any(v.armour == 0)) | (
        (diamond >= 3) & jnp.any(v.armour < 2)
    )
    gear = (v.sword > 0) | (v.bow > 0) | (jnp.sum(v.armour) > 0)
    attrs = jnp.stack([v.dexterity, v.strength, v.intelligence])

    def collect(name: str, tool: jax.Array, floors: tuple[int, ...]) -> jax.Array:
        return tool & _on(v, floors) & _below_cap(v, name, g)

    gems = _below_cap(v, "sapphire", g) | _below_cap(v, "ruby", g)
    cond = {
        "collect_wood": collect("wood", jnp.bool_(True), (0, 6, 7)),
        "collect_stone": collect("stone", _has(v, 0), (0, 2, 5, 6, 7)),
        "collect_coal": collect("coal", _has(v, 0), (0, 2, 5, 6)),
        "collect_iron": collect("iron", _has(v, 1), (0, 2, 5)),
        "collect_diamond": collect("diamond", _has(v, 2), (0, 2, 5, 7)),
        "collect_gem": (v.pickaxe >= 4) & _on(v, (2, 5, 6, 7)) & gems,
        "place_table": _n(v, "wood") >= 2,
        "place_furnace": stone,
        "make_wood_pickaxe": wood & ~_has(v, 0),
        "make_stone_pickaxe": wood & stone & ~_has(v, 1),
        "make_iron_pickaxe": wood & stone & iron & coal & ~_has(v, 2),
        "make_diamond_pickaxe": wood & (diamond >= 3) & (v.pickaxe < 4),
        "upgrade_sword": sword_tiers,
        "make_armour": armour,
        "make_torches": wood & coal & (_n(v, "torches") < 99),
        "drink": (v.drink < top) & _on(v, (0, 1, 2, 3, 4, 5, 7)),
        "eat": (v.food < top) & _on(v, (0, 1, 2, 3, 4, 5, 6)),
        "sleep": v.energy < top,
        "fight": True,
        "descend": v.ladder_open & (v.floor < 8),
        "open_chest": _on(v, (1, 3, 4)),
        "enchant": _on(v, (3, 4)) & (v.mana >= 9) & gem & gear,
        "unlock_any": True,
        "read_book": (_n(v, "books") >= 1) & ~jnp.all(v.spells),
        "level_up": (v.xp >= 1) & (jnp.min(attrs) < 5),
    }
    if v.classic:
        cond |= dict.fromkeys(CRAFTAX_ONLY, False)
    return jnp.stack([jnp.asarray(cond[s.name], jnp.bool_) for s in SKILLS])


class _Step(NamedTuple):
    """One env step as the success tests see it."""

    a: ObsView
    act: jax.Array
    ach: jax.Array
    """Achievement part of the reward."""
    b: ObsView


type _Test = Callable[[_Step], jax.Array]


def _success_tests(g: Granularity) -> list[_Test]:
    """Success test per skill id."""

    def rises(*names: str) -> _Test:
        def test(s: _Step) -> jax.Array:
            up = [bucket(n, _n(s.b, n), g) > bucket(n, _n(s.a, n), g) for n in names]
            return jnp.any(jnp.stack(up))

        return test

    def more(read: Callable[[ObsView], jax.Array]) -> _Test:
        return lambda s: read(s.b) > read(s.a)

    def made(tool: int) -> _Test:
        return lambda s: _has(s.b, tool) & ~_has(s.a, tool)

    def placed(action: int, kind: int) -> _Test:
        return lambda s: (
            (s.act == action) & (s.a.facing_block != kind) & (s.b.facing_block == kind)
        )

    def furnace(s: _Step) -> jax.Array:
        return placed(PLACE_FURNACE, FURNACE)(s) & jnp.any(s.b.blocks == CRAFTING_TABLE)

    def fight(s: _Step) -> jax.Array:
        hostile = s.a.facing_creature & HOSTILE
        gone = ~jnp.any(s.b.creatures, axis=(0, 1))
        boss = jnp.isin(s.a.facing_block, jnp.array(_BOSS))
        attack = (s.act == DO) & ~s.a.sleeping & (jnp.any(hostile) | boss)
        return attack & (jnp.any(hostile & gone) | (s.ach > UNLOCK_MIN))

    def enchanted(v: ObsView) -> jax.Array:
        return jnp.concatenate([jnp.stack([v.ench_sword, v.ench_bow]), v.ench_armour])

    tests: dict[str, _Test] = {
        **{f"collect_{m}": rises(m) for m in ("wood", "stone", "coal", "iron")},
        "collect_diamond": rises("diamond"),
        "collect_gem": rises("sapphire", "ruby"),
        "place_table": placed(PLACE_TABLE, CRAFTING_TABLE),
        "place_furnace": furnace,
        "make_wood_pickaxe": made(0),
        "make_stone_pickaxe": made(1),
        "make_iron_pickaxe": made(2),
        "make_diamond_pickaxe": lambda s: (s.a.pickaxe < 4) & (s.b.pickaxe >= 4),
        "upgrade_sword": more(_sword_rank),
        "make_armour": more(lambda v: jnp.sum(v.armour)),
        "make_torches": more(lambda v: _n(v, "torches")),
        "drink": lambda s: (s.b.drink > s.a.drink) & (s.b.drink >= _max_stats(s.b)[1]),
        "eat": more(lambda v: v.food),
        "sleep": lambda s: (
            s.a.sleeping & ~s.b.sleeping & (s.b.energy >= _max_stats(s.b)[1])
        ),
        "fight": fight,
        "descend": more(lambda v: v.floor),
        "open_chest": lambda s: (
            (s.act == DO) & (s.a.facing_block == CHEST) & (s.b.facing_block == PATH)
        ),
        "enchant": lambda s: jnp.any(enchanted(s.a) != enchanted(s.b)),
        "unlock_any": lambda s: s.ach > UNLOCK_MIN,
        "read_book": more(lambda v: jnp.sum(v.spells)),
        "level_up": more(lambda v: v.dexterity + v.strength + v.intelligence),
    }
    return [tests[s.name] for s in SKILLS]


def skill_step(
    skill: jax.Array | int,
    view: ObsView,
    action: jax.Array | int,
    r: jax.Array | float,
    view2: ObsView,
    done: jax.Array | bool,
    steps_in_skill: jax.Array | int,
    timeout_mult: jax.Array | float,
    *,
    granularity: Granularity = "recipe",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Success, end flag and `REASONS` index of the step ``view -(action)-> view2``.

    ``steps_in_skill`` counts the skill's steps before this one; the step limit is
    ``round(t_max x timeout_mult)`` (primitive options: 1). The first match wins:
    episode end (``done``: ``view2`` is a reset, so no test, no success), success,
    failure (``sleep`` woken below max energy), floor change, a low-stat flag
    switching on (not during ``drink``, ``eat``, ``sleep``), step limit. Collect
    skills succeed when the material's ``granularity`` bucket rises.
    """
    skill = jnp.asarray(skill, jnp.int32)
    ach = achievement_reward(r, view, view2)
    branches = [
        lambda s, t=t: jnp.asarray(t(s), jnp.bool_) for t in _success_tests(granularity)
    ]
    step = _Step(view, jnp.asarray(action), ach, view2)
    passed = jax.lax.switch(skill, branches, step)
    woken = view.sleeping & ~view2.sleeping & (skill == SKILL_IDS["sleep"])
    new_low = jnp.any(low_stats(view2) & ~low_stats(view))
    exempt = jnp.isin(skill, jnp.asarray(_LOW_STAT_EXEMPT))
    limit = jnp.maximum(1, jnp.round(jnp.asarray(T_MAX)[skill] * timeout_mult))
    limit = jnp.where(jnp.asarray(PRIMITIVE)[skill], 1, limit)
    done = jnp.asarray(done, jnp.bool_)
    ends = [done, passed, woken, view2.floor != view.floor, new_low & ~exempt]
    ends.append(steps_in_skill + 1 >= limit)
    reasons = [EPISODE_END, SUCCESS, FAILURE, FLOOR_CHANGE, LOW_STAT, TIMEOUT]
    reason = jnp.select(ends, reasons, RUNNING).astype(jnp.int32)
    return passed & ~done, reason != RUNNING, reason


def primitive_action(skill: jax.Array | int, view: ObsView) -> jax.Array:
    """Fixed action of a primitive option, -1 for a learned skill.

    ``level_up`` raises the lowest attribute; ties go dexterity, strength,
    intelligence (the first minimum).
    """
    attrs = jnp.stack([view.dexterity, view.strength, view.intelligence])
    level_up = LEVEL_UP_DEXTERITY + jnp.argmin(attrs)
    picks = [skill == SKILL_IDS["read_book"], skill == SKILL_IDS["level_up"]]
    return jnp.select(picks, [READ_BOOK, level_up], -1).astype(jnp.int32)


def test_agreement(
    skill: jax.Array | int, success: jax.Array, events: StepEvents
) -> tuple[jax.Array, jax.Array]:
    """Whether the events judge this skill end, and whether they agree with ``success``.

    Logging only. ``fight`` (a melee or ranged kill, or a kill unlock), ``eat`` (food
    rose) and ``unlock_any`` (any unlock) are judged at every end; the others only
    when one of their achievements (``_UNLOCKS``) is first unlocked this step, which
    means success. ``drink`` and ``level_up`` have no event counterpart.
    """
    game: Game = "classic" if events.unlocked.shape[-1] == 22 else "craftax"
    names = achievement_names(game)
    targets = np.array([
        [any(n.startswith(p) for p in _UNLOCKS.get(s.name, ())) for n in names]
        for s in SKILLS
    ])
    hit = jnp.any(events.unlocked & jnp.asarray(targets)[skill])
    kill = (jnp.sum(events.kills) > 0) | hit
    full = [skill == SKILL_IDS[n] for n in ("fight", "eat", "unlock_any")]
    judged = jnp.select(full, [kill, events.ate, jnp.any(events.unlocked)], hit)
    checked = jnp.isin(skill, jnp.asarray(_FULL_EVENT)) | hit
    return checked, jnp.asarray(success, jnp.bool_) == judged


test_agreement.__test__ = False  # type: ignore[attr-defined]  # pytest would collect it where imported
