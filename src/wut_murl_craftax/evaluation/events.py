"""Per-step events of Craftax and Craftax-Classic: a fixed-size ``StepEvents`` per step.

``step_events(game, s, action, s2)`` compares the state before an action with the
(pre-reset) state after it; it is pure, so it runs under ``jit`` and ``vmap``.
Definitions: "Shared step events" of the evaluation plan. Hits and kills are read from
the traces ``craftax_step`` leaves (attack cooldowns, projectile paths, the recover
counter, mob slots); where the state cannot tell causes apart, ``_damage_source``,
``_losses`` and ``step_events`` state the rule. Craftax is imported on first use: its
constants build JAX arrays, which would start the backends before ``configure_runtime``
splits the CPU.
"""

import importlib
from types import ModuleType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from .milestones import CREATURE_OFFSET, CREATURES, FLOORS, Game

NO_SOURCE = -1
"""``damage_source`` of a step with neither a hit nor unexplained combat damage."""
LAVA = len(CREATURES)
"""``damage_source`` of lava under the player (Classic; Craftax lava is unwalkable)."""
UNKNOWN = LAVA + 1
"""``damage_source`` of combat damage without a hit found in the state."""
DAMAGE_SOURCES: tuple[str, ...] = (*(n for _, _, n in CREATURES), "lava", "unknown")
"""Name per ``damage_source`` id >= 0: ``CREATURES`` order, then lava and unknown."""
MELEE_COOLDOWN, RANGED_COOLDOWN = 5, 4
"""Cooldown an attack sets in both games; otherwise it only counts down, so a slot
holding it in ``s2`` attacked in this step."""
DESPAWN_DISTANCE = 14
"""``EnvParams.mob_despawn_distance`` default of both games."""


@struct.dataclass
class StepEvents:
    """Events of one env step; under ``vmap`` every field gains the env axis."""

    unlocked: jax.Array
    """Achievements first unlocked this step, bool (67; Classic 22), by ``.value``
    (names: ``milestones.achievement_names``)."""
    floor: jax.Array
    """``s2.player_level``, int8 (Classic 0)."""
    health_delta: jax.Array
    """Health change, float32; reward = unlocked tier weights + 0.1 x this."""
    health_loss: jax.Array
    """``max(0, -health_delta)``, float16."""
    drain: jax.Array
    """Needs drain: the recover counter crossed its lower threshold (1 health), bool."""
    combat_damage: jax.Array
    """Health loss not from drain or a harmful potion, float16."""
    damage_source: jax.Array
    """``CREATURES`` index of the hit (enemy projectiles by projectile type), ``LAVA``,
    ``UNKNOWN`` or ``NO_SOURCE``, int8."""
    kills: jax.Array
    """Melee and ranged kills per ``CREATURES`` index, int8; passive slots stay 0."""
    passive_kills: jax.Array
    """Cows, bats and snails killed, int8."""
    damage_dealt: jax.Array
    """Mob health removed on the player's floor (overkill excluded), float16."""
    projectile_damage: jax.Array
    """Part of ``damage_dealt`` not on the ``DO`` target, float16."""
    ate: jax.Array
    drank: jax.Array
    slept: jax.Array
    """Food rose, drink rose, sleep started; bool."""
    stat_low: jax.Array
    stat_zero: jax.Array
    """Food, drink, energy ``<= 2`` and ``== 0`` in ``s2``, bool[3]."""
    forced: jax.Array
    """The game replaced the action with ``NOOP`` (sleeping or resting), bool."""
    effective: jax.Array
    """The action changed the player or the world (passive decay and projectiles in
    flight excluded), bool; wasted = not effective, not ``NOOP``, not forced."""
    hostiles_on_screen: jax.Array
    hostiles_close: jax.Array
    """Melee, ranged mobs and enemy projectiles in the view and within 2 tiles, int8."""


class _Group(NamedTuple):
    """One creature class on the player's floor, slots on the first axis."""

    pos: jax.Array
    mask: jax.Array
    health: jax.Array
    cooldown: jax.Array
    type_id: jax.Array
    heading: jax.Array
    """Direction of a projectile; zeros for mobs."""


class _View(NamedTuple):
    """The fields both games share, read on floor ``lv`` of ``s2``."""

    level: jax.Array
    position: jax.Array
    direction: jax.Array
    health: jax.Array
    food: jax.Array
    drink: jax.Array
    energy: jax.Array
    recover: jax.Array
    mana: jax.Array
    sleeping: jax.Array
    resting: jax.Array
    bolts: jax.Array
    """Player projectile slots alive."""
    melee: _Group
    passive: _Group
    ranged: _Group
    shots: _Group
    """Enemy projectiles."""
    tile: jax.Array
    """Block under the player."""
    digest: tuple[jax.Array, ...]
    """Game-specific fields whose change makes an action effective."""


class _Rules(NamedTuple):
    """Per-game constants, built inside the trace."""

    consts: ModuleType
    melee_damage: jax.Array
    """Base damage (sum of the damage vector) per melee ``type_id``."""
    shot_damage: jax.Array
    """Base damage per projectile ``type_id``."""
    shot_of: jax.Array
    """Projectile ``type_id`` fired per ranged ``type_id``."""
    hit_in_place: bool
    """A projectile already on the player's tile hits (Craftax only)."""
    half_view: jax.Array


def _rules(game: Game) -> _Rules:
    """Constants of ``game``; Classic zombies and arrows deal 2 (7 to a sleeper)."""
    c = importlib.import_module(
        "craftax.craftax_classic.constants"
        if game == "classic"
        else "craftax.craftax.constants"
    )
    half = jnp.asarray(c.OBS_DIM) // 2
    if game == "classic":
        two = jnp.array([2.0])
        return _Rules(c, two, two, jnp.array([0]), False, half)
    dmg = c.MOB_TYPE_DAMAGE_MAPPING.sum(-1)  # (type, class); class 1 melee, 3 shot
    return _Rules(
        c,
        dmg[:, 1],
        dmg[:, 3],
        c.RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING,
        True,
        half,
    )


def _group(m: Any, lv: Any = None, heading: jax.Array | None = None) -> _Group:
    """Slots of ``m`` on floor ``lv`` (``None``: Classic, no floor axis or types)."""

    def pick(x: jax.Array) -> jax.Array:
        return x if lv is None else x[lv]

    pos, mask = pick(m.position), pick(m.mask)
    type_id = jnp.zeros_like(mask, jnp.int32) if lv is None else pick(m.type_id)
    return _Group(
        pos,
        mask,
        pick(m.health).astype(jnp.float32),
        pick(m.attack_cooldown),
        type_id,
        jnp.zeros_like(pos) if heading is None else pick(heading),
    )


def _craftax_view(s: Any, lv: jax.Array) -> _View:
    """Craftax state on floor ``lv``."""
    r, c = s.player_position
    digest = (
        s.map[lv],
        s.item_map[lv],
        *jax.tree.leaves(s.inventory),
        s.learned_spells,
        s.sword_enchantment,
        s.bow_enchantment,
        s.armour_enchantments,
        s.player_xp,
        s.player_dexterity,
        s.player_strength,
        s.player_intelligence,
    )
    return _View(
        s.player_level,
        s.player_position,
        s.player_direction,
        jnp.asarray(s.player_health, jnp.float32),
        s.player_food,
        s.player_drink,
        s.player_energy,
        s.player_recover,
        s.player_mana,
        s.is_sleeping,
        s.is_resting,
        s.player_projectiles.mask[lv],
        _group(s.melee_mobs, lv),
        _group(s.passive_mobs, lv),
        _group(s.ranged_mobs, lv),
        _group(s.mob_projectiles, lv, s.mob_projectile_directions),
        s.map[lv, r, c],
        digest,
    )


def _classic_view(s: Any) -> _View:
    """Classic state on the Craftax slots, type 0 on floor 0.

    Zombies are melee, cows passive, skeletons ranged, arrows enemy projectiles.
    """
    r, c = s.player_position
    return _View(
        jnp.int32(0),
        s.player_position,
        s.player_direction,
        jnp.asarray(s.player_health, jnp.float32),
        s.player_food,
        s.player_drink,
        s.player_energy,
        s.player_recover,
        jnp.int32(0),
        s.is_sleeping,
        jnp.bool_(False),
        jnp.zeros(0, bool),
        _group(s.zombies),
        _group(s.cows),
        _group(s.skeletons),
        _group(s.arrows, heading=s.arrow_directions),
        s.map[r, c],
        (s.map, *jax.tree.leaves(s.inventory)),
    )


def _damage_source(rules: _Rules, a: _View, b: _View, combat: jax.Array) -> jax.Array:
    """Source of this step's hits, from the game's own traces.

    Melee: a live slot whose cooldown reads ``MELEE_COOLDOWN`` in ``s2``. Enemy
    projectile: a live projectile on or one step before the player (players move
    before mobs). Point-blank shot: an adjacent ranged mob that attacked while a
    projectile slot was free; its new projectile hits in the same step. Several hits:
    the largest base damage of the type, ties to melee. Lava under the player (Classic,
    health set to 0) wins over hits.
    """
    p = b.position
    on = (a.shots.pos == p).all(-1) & rules.hit_in_place
    shot = a.shots.mask & (on | (a.shots.pos + a.shots.heading == p).all(-1))
    near = jnp.abs(a.ranged.pos - p).sum(-1) == 1
    free = a.shots.mask.sum() < a.shots.mask.shape[0]
    blank = a.ranged.mask & (b.ranged.cooldown == RANGED_COOLDOWN) & near & free
    melee = a.melee.mask & (b.melee.cooldown == MELEE_COOLDOWN)
    fired = rules.shot_of[a.ranged.type_id]
    hit = jnp.concatenate([melee, shot, blank])
    shot_id = CREATURE_OFFSET["projectile"]
    src = jnp.concatenate([
        CREATURE_OFFSET["melee"] + a.melee.type_id,
        shot_id + a.shots.type_id,
        shot_id + fired,
    ])
    dmg = jnp.concatenate([
        rules.melee_damage[a.melee.type_id],
        rules.shot_damage[a.shots.type_id],
        rules.shot_damage[fired],
    ])
    best = src[jnp.argmax(jnp.where(hit, dmg, -1.0))]
    rest = jnp.where(hit.any(), best, jnp.where(combat > 0, UNKNOWN, NO_SOURCE))
    lava = b.tile == rules.consts.BlockType.LAVA.value
    return jnp.where(lava, LAVA, rest).astype(jnp.int8)


def _losses(
    a: _Group, b: _Group, p: jax.Array, boss: jax.Array, target: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Killed slots, health removed and the part off the ``DO`` target.

    A live slot that is dead in ``s2``, or was refilled by a spawn in the same step
    (jumped more than one tile, or gained health: live mobs never heal), was killed
    unless it was at despawn distance from the player (never on the boss floor); a
    killed slot loses its remaining health.
    """
    refilled = (jnp.abs(b.pos - a.pos).sum(-1) > 1) | (b.health > a.health)
    stays = (jnp.abs(a.pos - p).sum(-1) < DESPAWN_DISTANCE) | boss
    killed = a.mask & stays & (~b.mask | refilled)
    lost = jnp.where(killed, a.health, jnp.maximum(a.health - b.health, 0.0))
    dealt = jnp.where(a.mask, lost, 0.0)
    return killed, dealt, jnp.where(target, 0.0, dealt)


def _effective(a: _View, b: _View) -> jax.Array:
    """Whether the action changed the player or the world around it.

    Changes: position, direction, floor, sleeping, resting, the digest, the health of
    the mobs alive in ``s``; rises of food, drink, energy; a fall of mana; a player
    projectile created. Passive decay, regeneration and projectiles already in flight
    do not count, so the wasted share is not biased low.
    """
    shared = (a.position, a.direction, a.level, a.sleeping, a.resting)
    mine = (b.position, b.direction, b.level, b.sleeping, b.resting)
    pairs = zip((*shared, *a.digest), (*mine, *b.digest), strict=True)
    changed = [jnp.any(x != y) for x, y in pairs]
    changed += [b.food > a.food, b.drink > a.drink, b.energy > a.energy]
    changed += [b.mana < a.mana, jnp.any(~a.bolts & b.bolts)]
    for ga, gb in ((a.melee, b.melee), (a.passive, b.passive), (a.ranged, b.ranged)):
        changed.append(jnp.any(ga.mask & (ga.health != gb.health)))
    return jnp.stack(changed).any()


def _potion_harm(c: ModuleType, s: Any, act: jax.Array) -> jax.Array:
    """3 when a potion with effect 1 (health -3 in ``drink_potion``) is drunk."""
    colour = act - c.Action.DRINK_POTION_RED.value
    k = jnp.clip(colour, 0, 5)
    drunk = (colour >= 0) & (colour < 6) & (s.inventory.potions[k] > 0)
    return jnp.where(drunk & (s.potion_mapping[k] == 1), 3.0, 0.0)


def _hostiles(rules: _Rules, b: _View) -> tuple[jax.Array, jax.Array]:
    """Live melee, ranged mobs and enemy projectiles in the view and within 2 tiles."""
    groups = (b.melee, b.ranged, b.shots)
    alive = jnp.concatenate([g.mask for g in groups])
    d = jnp.abs(jnp.concatenate([g.pos for g in groups]) - b.position)
    on = alive & (d <= rules.half_view).all(-1)
    close = alive & (d.max(-1) <= 2)
    return on.sum().astype(jnp.int8), close.sum().astype(jnp.int8)


def step_events(game: Game, s: Any, action: jax.Array | int, s2: Any) -> StepEvents:
    """Events of the step ``s -(action)-> s2`` of ``game`` (raw ``EnvState``s).

    ``s2`` is the state before any auto-reset, so unlocks on a done step are kept.
    Combat damage = health loss - drain (1) - harmful potion (3), clipped at 0; a hit
    hidden by a heal in the same step keeps its source with 0 damage.
    """
    rules = _rules(game)
    c = rules.consts
    if game == "classic":
        a, b = _classic_view(s), _classic_view(s2)
    else:
        a, b = _craftax_view(s, s2.player_level), _craftax_view(s2, s2.player_level)
    forced = a.sleeping | a.resting
    act = jnp.where(forced, c.Action.NOOP.value, action)
    # the counter moves by at most 1 a step and resets to 0 only at > 25 or < -15
    drain = (a.recover < -14) & (b.recover == 0)
    delta = b.health - a.health
    loss = jnp.maximum(-delta, 0.0)
    harm = _potion_harm(c, s, act) if game == "craftax" else 0.0
    combat = jnp.maximum(loss - drain - harm, 0.0)
    front, do = a.position + c.DIRECTIONS[a.direction], c.Action.DO.value
    boss = b.level == len(FLOORS) - 1
    out = [
        _losses(ga, gb, b.position, boss, (ga.pos == front).all(-1) & (act == do))
        for ga, gb in ((a.melee, b.melee), (a.passive, b.passive), (a.ranged, b.ranged))
    ]
    (km, dm, pm), (kp, dp, pp), (kr, dr, pr) = out
    kills = (
        jnp
        .zeros(len(CREATURES), jnp.int32)
        .at[CREATURE_OFFSET["melee"] + a.melee.type_id]
        .add(km)
        .at[CREATURE_OFFSET["ranged"] + a.ranged.type_id]
        .add(kr)
    )
    stats = jnp.stack([b.food, b.drink, b.energy])
    on_screen, close = _hostiles(rules, b)
    f16 = jnp.float16
    return StepEvents(
        unlocked=s2.achievements & ~s.achievements,
        floor=jnp.asarray(b.level, jnp.int8),
        health_delta=delta,
        health_loss=loss.astype(f16),
        drain=drain,
        combat_damage=combat.astype(f16),
        damage_source=_damage_source(rules, a, b, combat),
        kills=kills.astype(jnp.int8),
        passive_kills=kp.sum().astype(jnp.int8),
        damage_dealt=(dm.sum() + dp.sum() + dr.sum()).astype(f16),
        projectile_damage=(pm.sum() + pp.sum() + pr.sum()).astype(f16),
        ate=b.food > a.food,
        drank=b.drink > a.drink,
        slept=~a.sleeping & b.sleeping,
        stat_low=stats <= 2,
        stat_zero=stats == 0,
        forced=forced,
        effective=_effective(a, b),
        hostiles_on_screen=on_screen,
        hostiles_close=close,
    )
