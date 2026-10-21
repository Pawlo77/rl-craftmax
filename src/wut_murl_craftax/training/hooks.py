"""Shared extension points of the recurrent controls: the hooks later methods set.

`HookFields` is the base of ``PPORNNConfig`` and ``PQNRNNConfig``. A method is a
`HookMethod`: its config class (its own YAML keys) and a builder of its hooks from it
and the resolved base config, for each run config (`with_method`). `None` hooks are
skipped at trace time (`ResolvedHooks.active`): with none, the graph is the control's.
Env states give ``achievements`` and inventory; step outcomes come from ``traj.events``
(`StepEvents` of the pre-reset next state; on a done ``next_env_state`` already starts
the next episode). ``r_int`` is added before GAE, extrinsic rewards stay for the score.
``env_wrapper_fn`` wraps the bare env inside the log wrapper in training and evaluation
(``ppo.make_env``), so it acts before resets. No Craftax import here.
"""

import dataclasses
import functools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

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
    "after_extra_fn",
    "env_wrapper_fn",
    "metrics_fn",
)
"""The callable hooks, in call order; also the names `ResolvedHooks.active` reports."""
OLD_METRIC_PREFIX = "hook/"
"""Prefix of the first draft of the metric names; now refused."""
MEMORY_LIMIT_KEY = "memory_limit_bytes"
"""Runtime key the runner adds to each run config beside ``steps``: the machine
profile's limit. Never hashed or saved; trainer configs ignore it and pass it on as
`HookSpec.memory_limit_bytes`."""


@dataclass(frozen=True, slots=True)
class HookSpec:
    """What the run looks like, handed to ``init_state_fn``."""

    n_envs: int
    obs_dim: int
    n_actions: int
    n_achievements: int
    total_steps: int
    memory_limit_bytes: int | None = None
    """Memory the method's buffers may take on this machine (the profile's); None
    outside the runner (evaluation, tests): nothing to check."""


@dataclass(frozen=True, kw_only=True)
class HookFields:
    """Hook fields of a trainer config; the config classes inherit, never redeclare.

    Keyword-only, so subclass fields stay positional; no slots, so a config can also
    inherit a slotted trainer config (``PPORNNConfig(PPOConfig, HookFields)``).
    """

    init_state_fn: Callable[[jax.Array, HookSpec], State] | None = None
    """``(key, spec) -> state``, once in init; ``spec`` is `HookSpec` (Classic, 8 envs:
    8, 1345, 17, 22, budget). The state is one pytree in the train state, saved by
    ``save_state``. It lives on the actor; with ``observe`` or ``extra_loss_fn`` it
    moves to the learner and back each update, or stays on the learner for good when
    ``intrinsic_reward_fn`` is unset (PPO-RNN; the rollout never reads it)."""
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
    loss is not active, even when set. PPO-RNN skips a step whose gradient is all zero
    (nothing to learn): the params and Adam's moments and count stay, while the step
    count and the schedule's count advance as for an applied step, so the learning-rate
    anneal stays aligned with the control's.
    PPO-RNN adds ``<group>/grad_norm`` (mean gradient norm of the K steps) and
    ``<group>/kl_after`` (mean KL of the policy before and after them on the PPO batch)
    to the extra loss's metrics, ``<group>`` being the one group of its names."""
    after_extra_fn: Callable[[State, dict[str, jax.Array]], State] | None = None
    """``(state, metrics) -> state``, once after the extra steps with their metrics
    (averaged, plus the two above); PPO-RNN only, active with the extra loss."""
    env_wrapper_fn: Callable[[Any, Any], tuple[Any, Any]] | None = None
    """``(env, env_params) -> (env, env_params)``, once when the training or the
    evaluation env is built. The wrapped env must proxy ``name`` and keep the Craftax
    state reachable through ``.env_state`` (`unwrap_state`): step events and episode
    metrics read it; ``ppo.make_env`` refuses a wrapper that breaks either."""
    metrics_fn: Callable[[State], dict[str, Any]] | None = None
    """``(state) -> dict``, once per update where the state is (after the learner
    hooks, else on the actor after the rollout); names are ``count/..``, ``sil/..``,
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
    after_extra_fn: Callable[[State, dict[str, jax.Array]], State]
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


def _keep_after_extra(state: State, _metrics: dict[str, jax.Array]) -> State:
    return state


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

    ``extra_loss_fn`` and ``after_extra_fn`` count as set only with ``extra_steps > 0``.

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
    unused = () if hooks.extra_steps else ("extra_loss_fn", "after_extra_fn")
    return ResolvedHooks(
        init_state_fn=pick(hooks.init_state_fn, _no_init),
        intrinsic_reward_fn=pick(hooks.intrinsic_reward_fn, _zero_reward),
        observe=pick(hooks.observe, _keep_state),
        extra_loss_fn=pick(hooks.extra_loss_fn, _no_extra_loss),
        extra_steps=hooks.extra_steps,
        after_extra_fn=pick(hooks.after_extra_fn, _keep_after_extra),
        env_wrapper_fn=pick(hooks.env_wrapper_fn, _keep_env),
        metrics_fn=_checked(metrics) if metrics is not None else _no_metrics,
        active=tuple(
            n for n in HOOK_NAMES if getattr(hooks, n) is not None and n not in unused
        ),
    )


@dataclass(frozen=True, slots=True)
class HookMethod[C]:
    """A method made of hooks on a base trainer, configured by its own run-config keys.

    ``config`` is a frozen dataclass whose fields are the method's YAML keys (YAML
    lists arrive as tuples). ``hooks(method_cfg, base_cfg)`` builds its `HookFields`
    from one such config and the resolved base trainer config (e.g. SIL's network
    and ``gamma``). It must be pure Python and create no arrays: it may run inside
    ``jax.eval_shape`` (evaluation), so arrays belong in ``init_state_fn``.
    """

    config: type[C]
    hooks: Callable[[C, Any], HookFields]

    @property
    def keys(self) -> frozenset[str]:
        """The method's run-config keys: the fields of ``config``."""
        return frozenset(f.name for f in dataclasses.fields(cast(Any, self.config)))


def _hashable(key: str, value: Any) -> Any:
    """A YAML value as a hashable one: lists become tuples, mappings are refused."""
    if isinstance(value, list | tuple):
        return tuple(_hashable(key, v) for v in value)
    if isinstance(value, Mapping):
        msg = f"method key {key!r} holds a mapping; method keys take scalars or lists"
        raise TypeError(msg)
    return value


def _closed_over(fn: Any) -> list[Any]:
    """Values a hook closes over: its closure cells and partial arguments."""
    if isinstance(fn, functools.partial):
        return [*fn.args, *fn.keywords.values(), *_closed_over(fn.func)]
    return [c.cell_contents for c in getattr(fn, "__closure__", None) or ()]


@functools.cache
def _built(method: HookMethod[Any], cfg: Any, base: Any) -> HookFields:
    """One `HookFields` per method and base config, so equal run configs compile once.

    Equal run configs give equal trainer configs: the per-config setup caches hit.

    Raises:
        ValueError: If a hook closes over an array (a tracer under ``eval_shape``).
    """
    hooks = method.hooks(cfg, base)
    for name in HOOK_NAMES:
        if any(isinstance(v, jax.Array) for v in _closed_over(getattr(hooks, name))):
            msg = f"{name} closes over an array; create arrays in init_state_fn"
            raise ValueError(msg)
    return hooks


def with_method[T: HookFields](
    cfg: T,
    method: HookMethod[Any] | None,
    config: Mapping[str, Any],
    from_config: Callable[[Mapping[str, Any], T], T],
) -> T:
    """The trainer config of a run config: base keys onto ``cfg``, then method hooks.

    ``from_config`` applies the base keys; ``method`` builds the hooks from its own
    keys and that base config.

    The method's hooks (and ``extra_steps``) replace the base's; ``from_config``
    refuses unknown keys, with the method's keys added to its error. Without a method
    every key goes to ``from_config``.
    """
    if method is None:
        return from_config(config, cfg)
    mine = {k: _hashable(k, v) for k, v in config.items() if k in method.keys}
    try:
        base = from_config({k: v for k, v in config.items() if k not in mine}, cfg)
    except ValueError as error:
        error.add_note(f"method keys: {sorted(method.keys)}")
        raise
    hooks = _built(method, method.config(**mine), base)
    names = (f.name for f in dataclasses.fields(HookFields))
    return dataclasses.replace(base, **{n: getattr(hooks, n) for n in names})


def check_method_keys(method: HookMethod[Any] | None, base: type[Any]) -> None:
    """Refuse method keys that are also ``base`` config fields or ``method``.

    Raises:
        ValueError: Naming the shadowed keys.
    """
    if method is None:
        return
    taken = {f.name for f in dataclasses.fields(base)} | {"method"}
    if clash := sorted(method.keys & taken):
        msg = f"method keys {clash} are also {base.__name__} keys"
        raise ValueError(msg)


def config_dict(cfg: HookFields) -> dict[str, Any]:
    """Fields of a trainer config for YAML and the run-name hash: callables dropped.

    ``extra_steps`` and the trainer's own fields stay; the hooks are carried by the
    method name, not by the config.
    """
    names = (f.name for f in dataclasses.fields(cfg) if f.name not in HOOK_NAMES)
    return {n: getattr(cfg, n) for n in names}
