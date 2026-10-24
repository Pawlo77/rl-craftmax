"""Trainer registry: every method's `Trainer`, keyed by registry name, and `Policy`.

A hook method registers its base's trainer with a `HookMethod`, for example
``make_ppo_rnn_trainer(PPORNNConfig(), method=HookMethod(CountBonusConfig, hooks))``:
each run config's method keys build its hooks, the other keys configure the base.
`combine` composes two hook methods on one control; the ``combo`` entry, the pair that
``configs/methods/combo.yaml`` names, is built on first use; a saved run rebuilds its
own pair (`run_trainer`, never combo.yaml). The runner imports this
module after ``configure_runtime``, so it and the trainers may import Craftax (which
starts the JAX backends) at module top.
"""

import dataclasses
import functools
import itertools
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import jax

from ..orchestration.queue import COMBINE_KEY, METHODS_DIR, load_method
from ..orchestration.runner import Trainer
from . import count_bonus, sil
from .dyna_q import DynaTrainer
from .hooks import (
    HookFields,
    HookMethod,
    HookSpec,
    ResolvedHooks,
    SplitState,
    State,
    Traj,
    build_hooks,
    resolve,
)
from .ppo import PPOTrainer
from .ppo_rnn import PPORNNConfig
from .ppo_rnn import make_trainer as make_ppo_rnn_trainer
from .pqn_rnn import PQNRNNConfig
from .pqn_rnn import make_trainer as make_pqn_rnn_trainer
from .smdp import SMDPTrainer

HOOKS_ONLY = (
    "combine takes hook methods only; a pair with a skill method (smdp-skills, "
    "dyna-q) needs its own path, decided at Phase 3 Task 1 Step 1"
)
_SOLE = ("extra_steps", "env_wrapper_fn")
"""Fields at most one method of a pair sets: two extra losses or wrappers do not
compose."""

type _Each = tuple[tuple[int, Callable[..., Any]], ...]
"""A pair's set hooks of one name, each with its method's index in the state."""


class Policy(Protocol):
    """How a trained agent acts; evaluation drives every method through it."""

    def init_carry(self, n: int) -> Any:
        """Recurrent state for ``n`` envs; an empty pytree for feed-forward nets."""
        ...

    def step(
        self,
        params: Any,
        carry: Any,
        obs: jax.Array,
        key: jax.Array,
        greedy: bool,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, Any]:
        """Actions for a batch of observations and the next carry.

        ``greedy`` takes the most likely action instead of sampling; jit-safe as a
        static argument or a traced bool. ``reward`` and ``done`` belong to the env
        step that produced ``obs`` (zeros at an episode start); evaluation passes
        them, policies that do not need them ignore them.
        """
        ...


type _Actor = tuple[int, ...] | None
"""Methods whose parts the rollout reads, when the pair state is a `SplitState`."""


def _put(state: Any, i: int, part: Any) -> Any:
    """``state`` with method ``i``'s part replaced: a tuple, or a dict by index."""
    return {**state, i: part} if isinstance(state, dict) else _put_t(state, i, part)


def _put_t(state: tuple[Any, ...], i: int, part: Any) -> tuple[Any, ...]:
    return (*state[:i], part, *state[i + 1 :])


def _join(state: Any) -> tuple[Any, ...]:
    """Both parts in method order, from a tuple or a `SplitState` of index dicts."""
    if not isinstance(state, SplitState):
        return cast(tuple[Any, ...], state)
    parts = state.actor | state.learner
    return tuple(parts[i] for i in range(len(parts)))


def _split(actor: _Actor, full: tuple[Any, ...]) -> Any:
    if actor is None:
        return full
    return SplitState(
        actor={i: x for i, x in enumerate(full) if i in actor},
        learner={i: x for i, x in enumerate(full) if i not in actor},
    )


def _nbytes(tree: Any) -> int:
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(tree))


def _init(
    fns: tuple[Callable[..., State], ...], actor: _Actor, key: jax.Array, spec: HookSpec
) -> Any:
    """Both parts; a memory limit leaves each method room for the other's state."""
    keys, limit, specs = jax.random.split(key), spec.memory_limit_bytes, [spec] * 2
    if limit is not None:
        free = dataclasses.replace(spec, memory_limit_bytes=None)
        sizes = [
            _nbytes(jax.eval_shape(lambda k, f=f: f(k, free), k))
            for f, k in zip(fns, keys, strict=True)
        ]
        specs = [
            dataclasses.replace(spec, memory_limit_bytes=limit - sum(sizes) + n)
            for n in sizes
        ]
    parts = tuple(f(k, s) for f, k, s in zip(fns, keys, specs, strict=True))
    return _split(actor, parts)


def _reward(fns: _Each, state: Any, *step: Any) -> tuple[jax.Array, Any]:
    """Each set reward on its method's part (of the actor dict when split); they add."""
    total = None
    for i, fn in fns:
        r, part = fn(state[i], *step)
        state, total = _put(state, i, part), r if total is None else total + r
    return cast(jax.Array, total), state


def _observe(fns: _Each, actor: _Actor, state: Any, traj: Traj) -> Any:
    full = _join(state)
    for i, fn in fns:
        full = _put_t(full, i, fn(full[i], traj))
    return _split(actor, full)


def _extra_loss(
    fns: _Each, actor: _Actor, params: Any, state: Any, key: jax.Array
) -> Any:
    ((i, fn),) = fns  # extra_steps is in _SOLE: one extra loss at most
    full = _join(state)
    loss, (part, metrics) = fn(params, full[i], key)
    return loss, (_split(actor, _put_t(full, i, part)), metrics)


def _after_extra(
    fns: _Each, actor: _Actor, state: Any, metrics: dict[str, jax.Array]
) -> Any:
    ((i, fn),) = fns
    full = _join(state)
    return _split(actor, _put_t(full, i, fn(full[i], metrics)))


def _metrics(fns: _Each, state: Any) -> dict[str, Any]:
    """Both methods' metrics, each under its own prefix; a name logged twice fails."""
    out: dict[str, Any] = {}
    full = _join(state)
    for i, fn in fns:
        new = fn(full[i])
        if clash := sorted(out.keys() & new.keys()):
            msg = f"both methods of the pair log {clash}"
            raise ValueError(msg)
        out |= new
    return out


def _actor_parts(hooks: list[ResolvedHooks], base: Any) -> _Actor:
    """The methods the rollout reads, when the others can stay on the learner.

    Only on a base that places a `SplitState` (``splits_method_state``), with a learner
    hook set and methods on both sides; else None, one state placed as a whole.
    """
    actor = tuple(i for i, h in enumerate(hooks) if "intrinsic_reward_fn" in h.active)
    learner = any({"observe", "extra_loss_fn"} & set(h.active) for h in hooks)
    split = getattr(base, "splits_method_state", False) and learner
    return actor if split and 0 < len(actor) < len(hooks) else None


def _pair_hooks(pair: tuple[HookMethod[Any], ...], cfg: Any, base: Any) -> HookFields:
    """Both methods' hooks on one state, a tuple or split (`_actor_parts`).

    A split state holds each part in a dict by method index: ``{0: state_a}``.

    Raises:
        ValueError: If both set a field of `_SOLE`.
    """
    built = [
        build_hooks(m, m.config(**{k: getattr(cfg, k) for k in m.keys}), base)
        for m in pair
    ]
    if clash := [n for n in _SOLE if all(getattr(h, n) for h in built)]:
        msg = f"both methods of the pair set {clash}; {_SOLE} do not compose"
        raise ValueError(msg)
    hooks = [resolve(h) for h in built]  # `active` drops a 0-step extra loss
    actor = _actor_parts(hooks, base)

    def lift(name: str, compose: Callable[..., Any], *args: Any) -> Any:
        fns = tuple(
            (i, getattr(h, name)) for i, h in enumerate(hooks) if name in h.active
        )
        return functools.partial(compose, fns, *args) if fns else None

    wrapper = [h.env_wrapper_fn for h in built if h.env_wrapper_fn is not None]
    init = tuple(h.init_state_fn for h in hooks)
    return HookFields(
        init_state_fn=functools.partial(_init, init, actor),
        intrinsic_reward_fn=lift("intrinsic_reward_fn", _reward),
        observe=lift("observe", _observe, actor),
        extra_loss_fn=lift("extra_loss_fn", _extra_loss, actor),
        extra_steps=sum(h.extra_steps for h in hooks),
        after_extra_fn=lift("after_extra_fn", _after_extra, actor),
        env_wrapper_fn=wrapper[0] if wrapper else None,
        metrics_fn=lift("metrics_fn", _metrics),
    )


def _pair_artifacts(
    pair: tuple[HookMethod[Any], ...], run_dir: Path, update: int, state: Any
) -> None:
    for method, part in zip(pair, _join(state), strict=True):
        if method.artifacts is not None:
            method.artifacts(run_dir, update, part)


def combine(a: HookMethod[Any], b: HookMethod[Any]) -> HookMethod[Any]:
    """One hook method of two on the same control; its state is ``(state_a, state_b)``.

    Its keys are both methods' keys. Intrinsic rewards add up; ``observe`` runs a's,
    then b's; the extra loss (with ``extra_steps`` and ``after_extra_fn``) and the env
    wrapper are those of the method that sets them, on its own state; metrics merge,
    each under its method's prefix. A field of `_SOLE` set by both fails when the hooks
    are built, a metric name logged by both when the metrics are first traced.

    Raises:
        TypeError: If either is not a `HookMethod` (`HOOKS_ONLY`).
        ValueError: If both have a key.
    """
    if not (isinstance(a, HookMethod) and isinstance(b, HookMethod)):
        raise TypeError(HOOKS_ONLY)
    if clash := sorted(a.keys & b.keys):
        msg = f"both methods of the pair have the keys {clash}"
        raise ValueError(msg)
    fields = [
        (f.name, f.type, dataclasses.field(default=f.default))
        if f.default_factory is dataclasses.MISSING
        else (f.name, f.type, dataclasses.field(default_factory=f.default_factory))
        for m in (a, b)
        for f in dataclasses.fields(cast(Any, m.config))
    ]

    def check(cfg: Any) -> None:
        """Each method's own validation of its keys."""
        for m in (a, b):
            m.config(**{k: getattr(cfg, k) for k in m.keys})

    config = dataclasses.make_dataclass(
        "PairConfig",
        fields,
        frozen=True,
        kw_only=True,
        namespace={"__post_init__": check},
    )
    files = a.artifacts is not None or b.artifacts is not None
    return HookMethod(
        config,
        functools.partial(_pair_hooks, (a, b)),
        functools.partial(_pair_artifacts, (a, b)) if files else None,
    )


def pair_trainer(
    entries: Mapping[str, Trainer], spec: Mapping[str, Any], where: str = "combo.yaml"
) -> Trainer:
    """`combine` of the pair ``spec`` (``{base, methods}``) names, on its control.

    Raises:
        TypeError: If a named entry is not a hook method (`HOOKS_ONLY`).
        ValueError: If ``spec`` names an unknown entry, or a method is not built on
            the named control.
    """
    if unknown := sorted({spec["base"], *spec["methods"]} - entries.keys()):
        msg = f"{where} names unknown registry entries {unknown}"
        raise ValueError(msg)
    base = entries[spec["base"]]
    pair: list[HookMethod[Any]] = []
    for name in spec["methods"]:
        method = getattr(entries[name], "method", None)
        if not isinstance(method, HookMethod):
            msg = f"{where}: {name} is not a hook method: {HOOKS_ONLY}"
            raise TypeError(msg)
        if type(entries[name]) is not type(base):
            msg = f"{where}: {name} is not built on {spec['base']}"
            raise ValueError(msg)
        pair.append(method)
    return cast(Trainer, dataclasses.replace(cast(Any, base), method=combine(*pair)))


def combo_trainer(
    entries: Mapping[str, Trainer], methods_dir: Path = METHODS_DIR
) -> Trainer:
    """The ``combo`` entry: `pair_trainer` of the pair ``combo.yaml`` names.

    The pair comes from ``methods_dir`` (the registry's: the shipped configs), so a
    queue run with another ``--methods-dir`` must name the same pair there.

    Raises:
        TypeError: If a named entry is not a hook method (`HOOKS_ONLY`).
        ValueError: If combo.yaml names no pair or an unknown entry, or a method is
            not built on the named control.
    """
    spec = load_method("combo", methods_dir).combine
    if spec is None:
        msg = "combo.yaml has no `combine` block naming the pair"
        raise ValueError(msg)
    return pair_trainer(entries, spec)


def _fits(entries: Mapping[str, Trainer], spec: dict[str, Any], config: Any) -> bool:
    """Does the pair's trainer accept every key of ``config``?"""
    try:
        cast(Any, pair_trainer(entries, spec)).config(config)
    except (TypeError, ValueError):
        return False
    return True


def combo_pair(
    config: Mapping[str, Any], entries: Mapping[str, Trainer]
) -> dict[str, Any]:
    """The ``{base, methods}`` of a saved combo run config (ruling R80a).

    Its `COMBINE_KEY` block when recorded; else the one pair of hook entries on one
    control that each set a key of ``config`` and whose trainer accepts all of them.
    An inferred pair is in registry-name order: the order of the method states, which
    acting and evaluation never read.

    Raises:
        ValueError: If no pair or more than one fits.
    """
    if COMBINE_KEY in config:
        return dict(config[COMBINE_KEY])
    named = {n: t for n, t in entries.items() if n != "combo"}
    hooked = {
        n: m
        for n, t in named.items()
        if isinstance(m := getattr(t, "method", None), HookMethod)
    }
    controls = [n for n in named if n not in hooked]
    fits = [
        spec
        for a, b in itertools.combinations(sorted(hooked), 2)
        if config.keys() & hooked[a].keys and config.keys() & hooked[b].keys
        for spec in ({"base": c, "methods": [a, b]} for c in controls)
        if _fits(named, spec, config)
    ]
    if len(fits) != 1:
        msg = (
            f"{len(fits)} registry pairs fit the combo config {sorted(config)}: "
            f"{fits}; record its `{COMBINE_KEY}` block"
        )
        raise ValueError(msg)
    return fits[0]


def run_trainer(
    config: Mapping[str, Any], methods: Mapping[str, Trainer] | None = None
) -> Trainer:
    """The trainer of a saved run's ``config.json``: ``methods[config["method"]]``.

    A ``combo`` run gets the pair it trained with (`combo_pair`), never the live
    combo.yaml; ``methods`` defaults to `METHODS`.
    """
    methods = METHODS if methods is None else methods
    if config["method"] != "combo":
        return methods[config["method"]]
    entries = {n: methods[n] for n in methods if n != "combo"}
    return pair_trainer(entries, combo_pair(config, entries), "config.json")


_ENTRIES: dict[str, Trainer] = {
    "ppo": PPOTrainer(),
    "ppo-rnn": make_ppo_rnn_trainer(PPORNNConfig()),
    "pqn-rnn": make_pqn_rnn_trainer(PQNRNNConfig()),
    "count-bonus-ppo-rnn": make_ppo_rnn_trainer(
        PPORNNConfig(), method=count_bonus.METHOD
    ),
    "count-bonus-pqn-rnn": make_pqn_rnn_trainer(
        PQNRNNConfig(), method=count_bonus.METHOD
    ),
    "sil": make_ppo_rnn_trainer(PPORNNConfig(), method=sil.METHOD),
    "smdp-skills": SMDPTrainer(),
    "dyna-q": DynaTrainer(),
}


class _Registry(Mapping[str, Trainer]):
    """The entries plus ``combo``, built on its first lookup.

    A combo.yaml that cannot build fails the combo runs only, never this import.
    """

    def __getitem__(self, name: str) -> Trainer:
        return _combo() if name == "combo" else _ENTRIES[name]

    def __contains__(self, name: object) -> bool:
        return name == "combo" or name in _ENTRIES

    def __iter__(self) -> Iterator[str]:
        return iter([*_ENTRIES, "combo"])

    def __len__(self) -> int:
        return len(_ENTRIES) + 1


@functools.cache
def _combo() -> Trainer:
    return combo_trainer(_ENTRIES)


METHODS: Mapping[str, Trainer] = _Registry()
"""Registry name (`RunSpec.registry_name`, ``config["method"]``) -> trainer; each
method plan adds its entry."""
