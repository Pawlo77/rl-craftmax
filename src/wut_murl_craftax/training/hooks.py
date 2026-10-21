"""Shared extension points of the recurrent controls: the hooks later methods set.

`HookFields` is the base of ``PPORNNConfig`` and ``PQNRNNConfig``; a method sets hooks
in its factory. `None` hooks are skipped at trace time (`ResolvedHooks.active`), so with
none the trainer graph is the control's. Hooks close over the method's config and the
network (``build_network``); `config_dict` keeps them out of YAML and the run-name hash.
Env states give ``achievements`` and inventory; step outcomes come from ``traj.events``
(`StepEvents` of the pre-reset next state; on a done ``next_env_state`` already starts
the next episode). ``r_int`` is added before GAE, extrinsic rewards stay for the score.
``env_wrapper_fn`` wraps the no-auto-reset env inside the log and optimistic-reset
wrappers, so a reward or ``done`` change acts before the reset. No Craftax import here.
"""

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ..evaluation.events import StepEvents

type State = Any
"""Method state: any pytree (arrays, nested dicts, typed PRNG keys)."""

HOOK_NAMES = (
    "init_state_fn",
    "intrinsic_reward_fn",
    "observe",
    "extra_loss_fn",
    "env_wrapper_fn",
    "metrics_fn",
)
"""The callable hooks, in call order; also the names `ResolvedHooks.active` reports."""
OLD_METRIC_PREFIX = "hook/"
"""Prefix of the first draft of the metric names; now refused."""


@dataclass(frozen=True, slots=True)
class HookSpec:
    """What the run looks like, handed to ``init_state_fn``."""

    n_envs: int
    obs_dim: int
    n_actions: int
    n_achievements: int
    total_steps: int


@dataclass(frozen=True, kw_only=True)
class HookFields:
    """Hook fields of a trainer config; the config classes inherit, never redeclare.

    Keyword-only, so subclass fields stay positional; no slots, so a config can also
    inherit a slotted trainer config (``PPORNNConfig(PPOConfig, HookFields)``).
    """

    init_state_fn: Callable[[jax.Array, HookSpec], State] | None = None
    """``(key, spec) -> state``, once in init; ``spec`` is `HookSpec` (Classic, 8 envs:
    8, 1345, 17, 22, budget). The state is one pytree in the train state, saved by
    ``save_state`` and moved to the actor for the rollout and back."""
    intrinsic_reward_fn: (
        Callable[
            [State, Any, Any, jax.Array, jax.Array, jax.Array], tuple[jax.Array, State]
        ]
        | None
    ) = None
    """``(state, env_state, next_env_state, obs_next, done, t) -> (r_int, state)`` every
    env step in the rollout scan (actor); ``t`` is the int32 env-step counter. Never
    compare the two env states across a done, read ``traj.events`` instead."""
    observe: Callable[[State, "Traj"], State] | None = None
    """``(state, traj) -> state``, once per rollout before GAE (learner device)."""
    extra_loss_fn: (
        Callable[
            [Any, State, jax.Array],
            tuple[jax.Array, tuple[State, dict[str, jax.Array]]],
        ]
        | None
    ) = None
    """``(params, state, key) -> (loss, (state, metrics))``, called per extra step."""
    extra_steps: int = 0
    """K optimizer steps on ``extra_loss_fn`` after the PPO epochs, same Adam state; the
    learning-rate schedule is indexed by update, so K does not shift it. At 0 the extra
    loss is not active, even when set."""
    env_wrapper_fn: Callable[[Any, Any], tuple[Any, Any]] | None = None
    """``(env, env_params) -> (env, env_params)``, once when the env is built."""
    metrics_fn: Callable[[State], dict[str, Any]] | None = None
    """``(state) -> dict``, once per update; names are ``count/..``, ``sil/..``,
    ``smdp/..``, ``dyna/..``, logged unchanged (`check_metric_names`; the old ``hook/``
    prefix is refused, at trace time on the first update)."""


@struct.dataclass
class Traj:
    """One rollout, shapes ``(num_steps, envs, ...)``, handed to ``observe``.

    ``hidden`` is the recurrent state the policy acted from at each step, float16 (a
    float32 value is cast on construction; PPO-RNN: the GRU state, 512; PQN-RNN: the
    LSTM ``(c, h)`` concatenated, 1024). ``events`` holds the `StepEvents` of each step.
    ``extra`` holds each base's own arrays (PPO-RNN: ``r_int``, ``values``, ``logp``;
    PQN-RNN: ``r_int``, ``q_vals``, ``last_actions``). ``rewards`` are extrinsic.
    """

    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    hidden: jax.Array
    events: StepEvents
    extra: dict[str, jax.Array]

    def __post_init__(self) -> None:
        """Cast a float32 ``hidden`` to float16; placeholders pass through."""
        h = self.hidden
        if isinstance(h, jax.Array | np.ndarray) and h.dtype == jnp.float32:
            object.__setattr__(self, "hidden", h.astype(jnp.float16))


@dataclass(frozen=True, slots=True)
class ResolvedHooks:
    """Hooks with no-op defaults filled in, plus which ones the method really set."""

    init_state_fn: Callable[[jax.Array, HookSpec], State]
    intrinsic_reward_fn: Callable[..., tuple[jax.Array, State]]
    observe: Callable[[State, Traj], State]
    extra_loss_fn: Callable[..., tuple[jax.Array, tuple[State, dict[str, jax.Array]]]]
    extra_steps: int
    env_wrapper_fn: Callable[[Any, Any], tuple[Any, Any]]
    metrics_fn: Callable[[State], dict[str, Any]]
    active: tuple[str, ...]
    """Names of the hooks the method set, in `HOOK_NAMES` order; trainers skip the
    others at trace time."""


def check_metric_names(names: Iterable[str]) -> None:
    """Require ``<group>/<name>`` metric names; the old ``hook/`` prefix is refused.

    Raises:
        ValueError: On a name without a ``/`` prefix, with an empty part or with
            the old ``hook/`` prefix.
    """
    for name in names:
        group, _, rest = name.partition("/")
        if name.startswith(OLD_METRIC_PREFIX):
            msg = f"metric name {name!r}: the {OLD_METRIC_PREFIX!r} prefix is retired"
            raise ValueError(msg)
        if not group or not rest or "" in rest.split("/"):
            msg = f"metric name {name!r} must look like 'count/size' (group/name)"
            raise ValueError(msg)


def _no_init(_key: jax.Array, _spec: HookSpec) -> State:
    return {}


def _zero_reward(
    state: State,
    _env_state: Any,
    _next_env_state: Any,
    _obs_next: Any,
    done: jax.Array,
    _t: Any,
) -> tuple[jax.Array, State]:
    return jnp.zeros(jnp.shape(done), jnp.float32), state


def _keep_state(state: State, _traj: Traj) -> State:
    return state


def _no_extra_loss(
    _params: Any, state: State, _key: jax.Array
) -> tuple[jax.Array, tuple[State, dict[str, jax.Array]]]:
    return jnp.zeros((), jnp.float32), (state, {})


def _keep_env(env: Any, env_params: Any) -> tuple[Any, Any]:
    return env, env_params


def _no_metrics(_state: State) -> dict[str, Any]:
    return {}


def _checked(
    metrics_fn: Callable[[State], Mapping[str, Any]],
) -> Callable[[State], dict[str, Any]]:
    def call(state: State) -> dict[str, Any]:
        out = dict(metrics_fn(state))
        check_metric_names(out)
        return out

    return call


def resolve(hooks: HookFields) -> ResolvedHooks:
    """Fill the no-op defaults of ``hooks`` and report the hooks the method set.

    ``extra_loss_fn`` counts as set only with ``extra_steps > 0``.

    Raises:
        ValueError: If ``extra_steps`` is negative, or positive with no
            ``extra_loss_fn``.
    """
    if hooks.extra_steps < 0:
        msg = f"extra_steps must be >= 0, got {hooks.extra_steps}"
        raise ValueError(msg)
    if hooks.extra_steps and hooks.extra_loss_fn is None:
        msg = "extra_steps > 0 needs an extra_loss_fn"
        raise ValueError(msg)

    def pick[F](hook: F | None, default: F) -> F:
        return default if hook is None else hook  # `or` would drop a falsy callable

    metrics = hooks.metrics_fn
    unused = () if hooks.extra_steps else ("extra_loss_fn",)
    return ResolvedHooks(
        init_state_fn=pick(hooks.init_state_fn, _no_init),
        intrinsic_reward_fn=pick(hooks.intrinsic_reward_fn, _zero_reward),
        observe=pick(hooks.observe, _keep_state),
        extra_loss_fn=pick(hooks.extra_loss_fn, _no_extra_loss),
        extra_steps=hooks.extra_steps,
        env_wrapper_fn=pick(hooks.env_wrapper_fn, _keep_env),
        metrics_fn=_checked(metrics) if metrics is not None else _no_metrics,
        active=tuple(
            n for n in HOOK_NAMES if getattr(hooks, n) is not None and n not in unused
        ),
    )


def config_dict(cfg: HookFields) -> dict[str, Any]:
    """Fields of a trainer config for YAML and the run-name hash: callables dropped.

    ``extra_steps`` and the trainer's own fields stay; the hooks are carried by the
    method name, not by the config.
    """
    names = (f.name for f in dataclasses.fields(cfg) if f.name not in HOOK_NAMES)
    return {n: getattr(cfg, n) for n in names}
