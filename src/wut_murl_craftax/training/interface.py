"""Trainer registry: every method's `Trainer`, keyed by registry name, and `Policy`.

A hook method registers its base's trainer with a `HookMethod`, for example
``make_ppo_rnn_trainer(PPORNNConfig(), method=HookMethod(CountBonusConfig, hooks))``:
each run config's method keys build its hooks, the other keys configure the base.
`combine` composes two hook methods on one control; the ``combo`` entry is the pair
that ``configs/methods/combo.yaml`` names. The runner imports this module after
``configure_runtime``, so it and the trainers may import Craftax (which starts the JAX
backends) at module top.
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import jax

from ..orchestration.queue import METHODS_DIR, load_method
from ..orchestration.runner import Trainer
from . import count_bonus, sil
from .dyna_q import DynaTrainer
from .hooks import HookFields, HookMethod, HookSpec, State, Traj, _built, resolve
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


def _put(state: tuple[Any, ...], i: int, part: Any) -> tuple[Any, ...]:
    return (*state[:i], part, *state[i + 1 :])


def _init(fns: tuple[Callable[..., State], ...], key: jax.Array, spec: HookSpec) -> Any:
    return tuple(f(k, spec) for f, k in zip(fns, jax.random.split(key), strict=True))


def _reward(fns: _Each, state: Any, *step: Any) -> tuple[jax.Array, Any]:
    """Each set reward on its method's part of the state; the bonuses add up."""
    total = None
    for i, fn in fns:
        r, part = fn(state[i], *step)
        state, total = _put(state, i, part), r if total is None else total + r
    return cast(jax.Array, total), state


def _observe(fns: _Each, state: Any, traj: Traj) -> Any:
    for i, fn in fns:
        state = _put(state, i, fn(state[i], traj))
    return state


def _extra_loss(fns: _Each, params: Any, state: Any, key: jax.Array) -> Any:
    ((i, fn),) = fns  # extra_steps is in _SOLE: one extra loss at most
    loss, (part, metrics) = fn(params, state[i], key)
    return loss, (_put(state, i, part), metrics)


def _after_extra(fns: _Each, state: Any, metrics: dict[str, jax.Array]) -> Any:
    ((i, fn),) = fns
    return _put(state, i, fn(state[i], metrics))


def _metrics(fns: _Each, state: Any) -> dict[str, Any]:
    """Both methods' metrics, each under its own prefix; a name logged twice fails."""
    out: dict[str, Any] = {}
    for i, fn in fns:
        new = fn(state[i])
        if clash := sorted(out.keys() & new.keys()):
            msg = f"both methods of the pair log {clash}"
            raise ValueError(msg)
        out |= new
    return out


def _pair_hooks(pair: tuple[HookMethod[Any], ...], cfg: Any, base: Any) -> HookFields:
    """Both methods' hooks on one state ``(state_a, state_b)``.

    Raises:
        ValueError: If both set a field of `_SOLE`.
    """
    built = [
        _built(m, m.config(**{k: getattr(cfg, k) for k in m.keys}), base) for m in pair
    ]
    if clash := [n for n in _SOLE if all(getattr(h, n) for h in built)]:
        msg = f"both methods of the pair set {clash}; {_SOLE} do not compose"
        raise ValueError(msg)
    hooks = [resolve(h) for h in built]  # `active` drops a 0-step extra loss

    def lift(name: str, compose: Callable[..., Any]) -> Any:
        fns = tuple(
            (i, getattr(h, name)) for i, h in enumerate(hooks) if name in h.active
        )
        return functools.partial(compose, fns) if fns else None

    wrapper = [h.env_wrapper_fn for h in built if h.env_wrapper_fn is not None]
    return HookFields(
        init_state_fn=functools.partial(_init, tuple(h.init_state_fn for h in hooks)),
        intrinsic_reward_fn=lift("intrinsic_reward_fn", _reward),
        observe=lift("observe", _observe),
        extra_loss_fn=lift("extra_loss_fn", _extra_loss),
        extra_steps=sum(h.extra_steps for h in hooks),
        after_extra_fn=lift("after_extra_fn", _after_extra),
        env_wrapper_fn=wrapper[0] if wrapper else None,
        metrics_fn=lift("metrics_fn", _metrics),
    )


def _pair_artifacts(
    pair: tuple[HookMethod[Any], ...], run_dir: Path, update: int, state: Any
) -> None:
    for method, part in zip(pair, state, strict=True):
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


def combo_trainer(
    entries: Mapping[str, Trainer], methods_dir: Path = METHODS_DIR
) -> Trainer:
    """The ``combo`` entry: `combine` of the pair ``combo.yaml`` names, on its control.

    Raises:
        TypeError: If a named entry is not a hook method (`HOOKS_ONLY`).
        ValueError: If combo.yaml names no pair, or a method is not built on the
            named control.
    """
    spec = load_method("combo", methods_dir).combine
    if spec is None:
        msg = "combo.yaml has no `combine` block naming the pair"
        raise ValueError(msg)
    base = entries[spec["base"]]
    pair: list[HookMethod[Any]] = []
    for name in spec["methods"]:
        method = getattr(entries[name], "method", None)
        if not isinstance(method, HookMethod):
            msg = f"combo.yaml: {name} is not a hook method: {HOOKS_ONLY}"
            raise TypeError(msg)
        if type(entries[name]) is not type(base):
            msg = f"combo.yaml: {name} is not built on {spec['base']}"
            raise ValueError(msg)
        pair.append(method)
    return cast(Trainer, dataclasses.replace(cast(Any, base), method=combine(*pair)))


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
METHODS: Mapping[str, Trainer] = _ENTRIES | {"combo": combo_trainer(_ENTRIES)}
"""Registry name (`RunSpec.registry_name`, ``config["method"]``) -> trainer; each
method plan adds its entry."""
