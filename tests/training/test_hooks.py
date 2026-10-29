"""Tests for the shared method hooks: defaults, resolve, config identity, Traj."""

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.events import StepEvents, step_events
from wut_murl_craftax.evaluation.milestones import achievement_names
from wut_murl_craftax.orchestration.queue import config_hash
from wut_murl_craftax.training import hooks
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.hooks import (
    HOOK_NAMES,
    HookFields,
    HookMethod,
    HookSpec,
    Traj,
    check_method_keys,
    check_metric_names,
    config_dict,
    resolve,
    with_method,
)

CLASSIC = "Craftax-Classic-Symbolic-v1"
T, N = 3, 8


@dataclasses.dataclass(frozen=True, slots=True)
class _Cfg(HookFields):
    """Stand-in for a trainer config: its own fields after the hook fields."""

    lr: float = 3e-4
    envs: int = 8


def _all_hooks() -> dict[str, Any]:
    return {
        "init_state_fn": lambda key, spec: {"n": jnp.zeros(spec.n_envs)},
        "intrinsic_reward_fn": lambda s, e, e2, o, d, t: (jnp.ones_like(d, float), s),
        "observe": lambda s, traj: s,
        "extra_loss_fn": lambda p, s, k: (jnp.float32(0), (s, {})),
        "after_extra_fn": lambda s, m: s,
        "env_wrapper_fn": lambda env, params: (env, params),
        "metrics_fn": lambda s: {"count/size": 1.0},
    }


def test_hook_fields_default_to_none_and_no_extra_steps() -> None:
    hf = HookFields()
    assert all(getattr(hf, name) is None for name in HOOK_NAMES)
    assert hf.extra_steps == 0
    assert set(HOOK_NAMES) | {"extra_steps"} == {f.name for f in dataclasses.fields(hf)}


def test_extra_steps_is_declared_in_hook_fields_only() -> None:
    assert "extra_steps" in HookFields.__dataclass_fields__
    own = {f.name for f in dataclasses.fields(_Cfg)} - set(
        HookFields.__dataclass_fields__
    )
    assert own == {"lr", "envs"}
    assert "extra_steps" not in _Cfg.__annotations__


def test_hook_fields_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        HookFields().extra_steps = 1  # type: ignore[misc]


def test_config_dict_drops_callables_and_keeps_numbers() -> None:
    cfg = _Cfg(extra_steps=2, **_all_hooks())
    assert config_dict(cfg) == {"extra_steps": 2, "lr": 3e-4, "envs": 8}


def test_callables_stay_out_of_the_yaml_round_trip() -> None:
    cfg = _Cfg(extra_steps=2, **_all_hooks())
    back = yaml.safe_load(yaml.safe_dump(config_dict(cfg)))
    assert back == config_dict(cfg)
    assert _Cfg(**back).extra_steps == 2


def test_callables_stay_out_of_the_run_name_hash() -> None:
    plain, hooked = _Cfg(), _Cfg(**_all_hooks())
    assert config_hash(config_dict(hooked)) == config_hash(config_dict(plain))
    # the hash function itself drops callables, so a raw dict with them agrees too
    raw = {**config_dict(plain), "intrinsic_reward_fn": _all_hooks()["observe"]}
    assert config_hash(raw) == config_hash(config_dict(plain))
    # a number in the same family still changes the hash
    assert config_hash(config_dict(_Cfg(extra_steps=1))) != config_hash(
        config_dict(plain)
    )


def test_resolve_fills_noop_defaults() -> None:
    r = resolve(HookFields())
    assert r.active == ()
    assert r.extra_steps == 0
    state = r.init_state_fn(jax.random.key(0), HookSpec(8, 1345, 17, 22, 1000))
    assert jax.tree.leaves(state) == []
    done = jnp.array([True, False, True])
    r_int, s2 = r.intrinsic_reward_fn(state, None, None, None, done, jnp.int32(5))
    assert r_int.shape == done.shape
    assert r_int.dtype == jnp.float32
    assert not r_int.any()
    assert s2 is state
    assert r.observe(state, "traj") is state
    env = object()
    assert r.env_wrapper_fn(env, "p") == (env, "p")
    loss, (s3, metrics) = r.extra_loss_fn({}, state, jax.random.key(1))
    assert float(loss) == 0.0
    assert s3 is state
    assert metrics == {}
    assert r.after_extra_fn(state, {"sil/kl_after": 0.0}) is state
    assert r.metrics_fn(state) == {}


def test_resolve_reports_active_hook_names() -> None:
    hooks_ = _all_hooks()
    r = resolve(
        _Cfg(
            intrinsic_reward_fn=hooks_["intrinsic_reward_fn"], observe=hooks_["observe"]
        )
    )
    assert r.active == ("intrinsic_reward_fn", "observe")
    assert resolve(_Cfg(extra_steps=1, **hooks_)).active == HOOK_NAMES


def test_an_extra_loss_without_extra_steps_is_not_active() -> None:
    """``extra_steps`` gates the extra loss, so the trainer never traces it at 0."""
    loss, after = _all_hooks()["extra_loss_fn"], _all_hooks()["after_extra_fn"]
    assert resolve(HookFields(extra_loss_fn=loss, after_extra_fn=after)).active == ()
    assert resolve(
        HookFields(extra_loss_fn=loss, after_extra_fn=after, extra_steps=1)
    ).active == ("extra_loss_fn", "after_extra_fn")


def test_resolve_keeps_a_falsy_callable_hook() -> None:
    class Falsy:
        def __call__(self, state: Any, traj: Any) -> Any:
            return state

        def __len__(self) -> int:
            return 0

    observe = Falsy()
    r = resolve(HookFields(observe=observe))
    assert r.observe is observe
    assert r.active == ("observe",)


def test_hook_fields_are_keyword_only() -> None:
    """Hooks never bind positionally; a subclass may add fields without defaults."""
    with pytest.raises(TypeError):
        HookFields(None)  # type: ignore[misc]

    @dataclasses.dataclass(frozen=True, slots=True)
    class Required(HookFields):
        lr: float

    cfg = Required(0.1, extra_steps=2)
    assert (cfg.lr, cfg.extra_steps) == (0.1, 2)


def test_resolve_keeps_user_hooks() -> None:
    hooks_ = _all_hooks()
    r = resolve(_Cfg(**hooks_))
    assert r.observe is hooks_["observe"]
    assert r.init_state_fn is hooks_["init_state_fn"]
    assert r.intrinsic_reward_fn is hooks_["intrinsic_reward_fn"]
    assert r.extra_loss_fn is hooks_["extra_loss_fn"]
    assert r.env_wrapper_fn is hooks_["env_wrapper_fn"]


def test_extra_steps_need_an_extra_loss_and_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="extra_loss_fn"):
        resolve(HookFields(extra_steps=2))
    with pytest.raises(ValueError, match="extra_steps"):
        resolve(HookFields(extra_steps=-1))
    resolve(HookFields(extra_steps=2, extra_loss_fn=_all_hooks()["extra_loss_fn"]))


def test_hook_spec_matches_craftax_classic_with_eight_envs() -> None:
    env = make_craftax_env_from_name(CLASSIC, auto_reset=False)
    params = env.default_params
    spec = HookSpec(
        n_envs=8,
        obs_dim=int(np.prod(env.observation_space(params).shape)),
        n_actions=env.action_space(params).n,
        n_achievements=len(achievement_names("classic")),
        total_steps=200_000,
    )
    assert spec == HookSpec(8, 1345, 17, 22, 200_000)


def test_init_state_fn_receives_the_spec() -> None:
    seen: list[HookSpec] = []

    def init(key: jax.Array, spec: HookSpec) -> dict[str, jax.Array]:
        seen.append(spec)
        return {"count": jnp.zeros(spec.n_achievements)}

    spec = HookSpec(8, 1345, 17, 22, 200_000)
    state = resolve(HookFields(init_state_fn=init)).init_state_fn(
        jax.random.key(0), spec
    )
    assert seen == [spec]
    assert (spec.n_envs, spec.obs_dim, spec.n_actions, spec.n_achievements) == (
        8,
        1345,
        17,
        22,
    )
    assert state["count"].shape == (22,)


def test_metric_names_reach_the_update_unchanged() -> None:
    names = {"count/size": 3.0, "sil/fill": 0.5, "smdp/len": 2.0, "dyna/err": 0.1}
    r = resolve(HookFields(metrics_fn=lambda s: names))
    assert r.metrics_fn({}) == names


@pytest.mark.parametrize("bad", ["x", "hook/x", "hook/count/x", "/x", "count/", "a//b"])
def test_bad_metric_names_raise_at_the_first_update(bad: str) -> None:
    r = resolve(HookFields(metrics_fn=lambda s: {"count/ok": 1.0, bad: 2.0}))
    with pytest.raises(ValueError, match="metric name"):
        r.metrics_fn({})


def test_check_metric_names_directly() -> None:
    check_metric_names(["count/a", "sil/b/c"])
    check_metric_names([])
    with pytest.raises(ValueError, match="'x'"):
        check_metric_names(["count/a", "x"])
    with pytest.raises(ValueError, match="hook/"):
        check_metric_names(["hook/a"])
    with pytest.raises(ValueError, match="1"):
        check_metric_names([1])  # type: ignore[list-item]  # a non-str key


def test_metrics_check_runs_when_traced_under_jit() -> None:
    r = resolve(HookFields(metrics_fn=lambda s: {"x": s}))
    with pytest.raises(ValueError, match="metric name"):
        jax.jit(r.metrics_fn)(jnp.float32(1))


def _traj(hidden: jax.Array, **over: Any) -> Traj:
    ev = over.pop("events", {"unlocked": jnp.zeros((T, N, 22), bool)})
    kw: dict[str, Any] = {
        "obs": jnp.zeros((T, N, 1345)),
        "actions": jnp.zeros((T, N), jnp.int32),
        "rewards": jnp.zeros((T, N)),
        "dones": jnp.zeros((T, N), bool),
        "hidden": hidden,
        "events": ev,
        "extra": {"r_int": jnp.zeros((T, N))},
    }
    return Traj(**(kw | over))


def test_traj_casts_float32_hidden_to_float16() -> None:
    traj = _traj(jnp.ones((T, N, 512), jnp.float32))
    assert traj.hidden.dtype == jnp.float16
    assert _traj(jnp.ones((T, N, 512), jnp.float16)).hidden.dtype == jnp.float16
    assert set(Traj.__dataclass_fields__) == {
        "obs", "actions", "rewards", "dones", "hidden", "events", "extra",
    }  # fmt: skip


def test_traj_survives_scan_vmap_and_tree_utils() -> None:
    def body(c: jax.Array, _: None) -> tuple[jax.Array, Traj]:
        return c + 1, _traj(jnp.zeros((T, N, 4), jnp.float32) + c)

    _, stacked = jax.lax.scan(body, jnp.float32(0), None, length=2)
    assert stacked.hidden.shape == (2, T, N, 4)
    assert stacked.hidden.dtype == jnp.float16
    # a tree map over placeholders (here ints, as vmap's in_axes) must not trip the cast
    axes = jax.tree.map(lambda _: 0, stacked)
    assert axes.hidden == 0


def test_traj_holds_real_step_events_per_step() -> None:
    env = make_craftax_env_from_name(CLASSIC, auto_reset=False)
    params = env.default_params
    _, s = env.reset_env(jax.random.key(0), params)
    _, s2, *_ = env.step_env(jax.random.key(1), s, 0, params)
    ev = step_events("classic", s, 0, s2)
    stacked = jax.tree.map(lambda x: jnp.stack([x] * T)[:, None].repeat(N, 1), ev)
    traj = _traj(
        jnp.zeros((T, N, 512)), events=stacked, extra={"values": jnp.zeros((T, N))}
    )
    assert traj.events.unlocked.shape == (T, N, 22)
    assert isinstance(traj.events, StepEvents)
    assert set(traj.extra) == {"values"}


def test_method_state_survives_save_and_load(tmp_path: Path) -> None:
    def init(key: jax.Array, spec: HookSpec) -> dict[str, Any]:
        return {
            "table": jax.random.normal(key, (spec.n_achievements, 4)),
            "buf": {"obs": jnp.arange(6.0).reshape(2, 3), "n": jnp.int32(5)},
            "key": jax.random.key(7),
        }

    r = resolve(HookFields(init_state_fn=init))
    state = r.init_state_fn(jax.random.key(0), HookSpec(8, 1345, 17, 22, 1000))
    save_state(tmp_path, {"method": state, "update": jnp.int32(3)})
    like = {
        "method": jax.tree.map(jnp.zeros_like, state),
        "update": jnp.int32(0),
    }
    got = load_state(tmp_path, like)
    np.testing.assert_array_equal(got["method"]["table"], state["table"])
    np.testing.assert_array_equal(got["method"]["buf"]["obs"], state["buf"]["obs"])
    assert int(got["method"]["buf"]["n"]) == 5
    np.testing.assert_array_equal(
        jax.random.key_data(got["method"]["key"]), jax.random.key_data(state["key"])
    )
    assert int(got["update"]) == 3


def _r_int(cfg: _Cfg) -> float:
    fn = cfg.intrinsic_reward_fn
    assert fn is not None
    return float(fn({}, None, None, None, jnp.zeros(2, bool), 0)[0][0])


def _build(
    config: dict[str, Any], method: HookMethod | None, base: _Cfg | None = None
) -> _Cfg:
    """`with_method` on `_Cfg`, whose ``from_config`` takes its own field names."""

    def from_config(keys: Any, cfg: _Cfg) -> _Cfg:
        if unknown := set(keys) - {"method", "lr", "envs"}:
            msg = f"unknown keys: {sorted(unknown)}"
            raise ValueError(msg)
        return dataclasses.replace(
            cfg, **{k: v for k, v in keys.items() if k != "method"}
        )

    return with_method(base or _Cfg(), method, config, from_config)


def test_a_method_builds_its_hooks_from_its_keys_and_the_base(
    knob_method: HookMethod,
) -> None:
    """Its keys and the resolved base config build the hooks; the rest is the base."""
    cfg = _build({"method": "knob", "knob_reward": 0.5, "lr": 2.0}, knob_method)
    assert (_r_int(cfg), cfg.lr) == (1.0, 2.0)
    assert _r_int(_build({"knob_reward": 0.5, "lr": 0.5}, knob_method)) == 0.25
    assert _r_int(_build({"lr": 1.0}, knob_method)) == 0.0  # its default knob


def test_equal_keys_give_equal_configs(knob_method: HookMethod) -> None:
    """One built hook set per method and base config: trainer setups cache on it."""
    a = _build({"knob_reward": 0.5}, knob_method)
    b = _build({"knob_reward": 0.5}, knob_method)
    assert a == b
    assert hash(a) == hash(b)
    assert a != _build({"knob_reward": 0.25}, knob_method)
    assert a != _build({"knob_reward": 0.5, "lr": 1.0}, knob_method)


def test_the_method_hooks_replace_the_factory_hooks(knob_method: HookMethod) -> None:
    base = _Cfg(**_all_hooks(), extra_steps=2, lr=1.0)
    cfg = _build({"knob_reward": 0.5}, knob_method, base)
    assert _r_int(cfg) == 0.5
    assert resolve(cfg).active == ("intrinsic_reward_fn",)
    assert (cfg.extra_steps, cfg.lr) == (0, base.lr)


def test_without_a_method_the_keys_go_to_the_base() -> None:
    assert _build({"lr": 2.0, "method": "x"}, None, _Cfg(lr=1.0)) == _Cfg(lr=2.0)


def test_an_unknown_key_names_the_method_keys(knob_method: HookMethod) -> None:
    with pytest.raises(ValueError, match="knob_rewrd") as info:
        _build({"knob_rewrd": 0.5}, knob_method)
    assert "method keys: ['knob_reward']" in "".join(info.value.__notes__)


@dataclasses.dataclass(frozen=True, slots=True)
class _Listed:
    tiers: tuple[int, ...] = (1, 2)


def _listed_hooks(cfg: _Listed, base: Any) -> HookFields:
    return HookFields(extra_steps=len(cfg.tiers))


def test_yaml_lists_become_tuples_and_mappings_are_refused() -> None:
    listed = HookMethod(_Listed, _listed_hooks)
    assert _build({"tiers": [1, [2, 3], 4]}, listed).extra_steps == 3
    with pytest.raises(TypeError, match=r"'tiers'.*mapping"):
        _build({"tiers": {"a": 1}}, listed)


def _array_hooks(cfg: _Listed, base: Any) -> HookFields:
    table = jnp.zeros(3)

    def observe(s: Any, traj: Any) -> Any:
        return s + table.sum()

    return HookFields(observe=observe)


def test_a_builder_that_creates_arrays_is_refused() -> None:
    """Builders may run inside ``jax.eval_shape``: an array there is a tracer."""
    with pytest.raises(ValueError, match=r"observe.*array"):
        _build({}, HookMethod(_Listed, _array_hooks))


@dataclasses.dataclass(frozen=True, slots=True)
class _Clash:
    lr: float = 0.1


def test_method_keys_that_shadow_base_keys_are_refused() -> None:
    clash = HookMethod(_Clash, lambda cfg, base: HookFields())
    with pytest.raises(ValueError, match="'lr'"):
        check_method_keys(clash, _Cfg)
    check_method_keys(None, _Cfg)


def test_module_has_no_craftax_import_at_the_top() -> None:
    src = Path(hooks.__file__).read_text()
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("craftax" in ln and "wut_murl" not in ln for ln in top)
