"""Tests for the count table and bonus: counts, masking, decay, SimHash, metrics."""

import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax.craftax_state import StaticEnvParams
from craftax.craftax_classic.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.orchestration.queue import RunSpec, config_hash, load_method
from wut_murl_craftax.training import count_bonus
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.count_bonus import (
    CountBonusConfig,
    init,
    intrinsic_reward,
    metrics,
    simhash_key,
    top_keys,
    write_top_keys,
)
from wut_murl_craftax.training.hooks import HookSpec
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.progress_key import CLASSIC_FIELDS, fields, hash_fields
from wut_murl_craftax.training.records import LOSS_COLUMNS
from wut_murl_craftax.training.wrappers import LogEnvState

CLASSIC = "Craftax-Classic-Symbolic-v1"
BETA = 0.1
SPEC = HookSpec(
    n_envs=2, obs_dim=1345, n_actions=17, n_achievements=22, total_steps=1000
)
KNOBS = {
    "beta": BETA,
    "granularity": "recipe",
    "decay": "constant",
    "log2_k": 20,
    "key": "progress",
    "max_occupancy": 0.25,
}
"""Every method knob (the config has no defaults): smoke config A."""
SIMHASH = {"key": "simhash", "log2_k": 24, "max_occupancy": 0.1}
"""The SimHash ablation's overrides, as count-bonus.yaml documents them."""
SIZES = {"batch": 4, "game": "classic"}
"""2 envs x 2 steps per update (a metrics window), Craftax-Classic fields."""
CLASSIC_METRICS = {
    "count/distinct_keys",
    "count/new_keys",
    "count/occupancy",
    "count/collision_est",
    "count/beta_t",
    "count/r_int_mean",
    "count/r_int_share",
    "count/r_int_by_floor/0",
    "count/max_key_floor",
    "count/table_full",
    "count/negative_counts",
}
"""Every metric of a Craftax-Classic run (one floor), the two abort guards included."""


def _cfg(**over: Any) -> CountBonusConfig:
    return CountBonusConfig(**(KNOBS | over))


@pytest.fixture(scope="module")
def env() -> Any:
    """Craftax-Classic env without auto-reset."""
    return make_craftax_env_from_name(CLASSIC, auto_reset=False)


@pytest.fixture(scope="module")
def start(env: Any) -> Any:
    """Craftax-Classic start state (key a)."""
    return env.reset(jax.random.key(0), env.default_params)[1]


@pytest.fixture(scope="module")
def picked(start: Any) -> Any:
    """The start state with a wood pickaxe (key b)."""
    return start.replace(inventory=start.inventory.replace(wood_pickaxe=1))


def _batch(*states: Any, returns: tuple[float, ...] | None = None) -> LogEnvState:
    """Stack per-env inner states under a log state, as the trainers see them;
    ``returns`` are the running extrinsic returns of the episodes."""
    inner = jax.tree.map(lambda *x: jnp.stack(x), *states)
    zeros = jnp.zeros(len(states))
    ret = zeros if returns is None else jnp.array(returns, jnp.float32)
    return LogEnvState(inner, ret, zeros, zeros, zeros, zeros)


def _call(state: Any, nxt: Any, done: list[bool], t: int = 0, env_state: Any = None):
    obs = jnp.zeros((len(done), SPEC.obs_dim))
    return jax.jit(intrinsic_reward)(
        state, env_state, nxt, obs, jnp.array(done), jnp.int32(t)
    )


def _key(s: Any) -> int:
    return int(hash_fields(fields(s, "recipe"), 20))


def test_scripted_counts_and_bonuses(start: Any, picked: Any) -> None:
    a, b = _key(start), _key(picked)
    assert a != b
    st = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    script = [
        ((start, start), [False, False], 2, [1 / math.sqrt(2)] * 2),
        ((picked, start), [False, False], 3, [1.0, 1 / math.sqrt(3)]),
        ((start, start), [False, True], 4, [0.5, 0.0]),
    ]
    for states, done, n_a, bonus in script:
        r, st = _call(st, _batch(*states), done)
        assert int(st.table[a]) == n_a
        np.testing.assert_allclose(r, BETA * np.array(bonus), rtol=1e-6)
    assert int(st.table[b]) == 1
    assert int(st.table.sum()) == 5  # the done step counted nothing


def test_bonus_ignores_the_pre_step_state(start: Any, picked: Any) -> None:
    st = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    nxt = _batch(picked, start)
    r1, s1 = _call(st, nxt, [False, True], env_state=_batch(start, start))
    r2, s2 = _call(st, nxt, [False, True], env_state=jnp.full(7, -1))
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(s1.table, s2.table)


def test_init_sizes_from_the_spec() -> None:
    spec = HookSpec(
        n_envs=4, obs_dim=8268, n_actions=43, n_achievements=67, total_steps=10**6
    )
    st = init(_cfg(), jax.random.key(2), spec, batch=4, game="craftax")
    assert st.table.shape == (2**20,)
    assert st.table.dtype == jnp.int32
    assert not st.table.any()
    assert st.proj.shape == (8268, 64)
    assert st.total_steps == 1_000_000
    assert st.seen.shape == (2**20, 23)
    assert st.window.floor_sum.shape == (9,)
    # per-env leaves (leading dim == envs) are sharded over the mesh: the episodes'
    per_env = [x for x in jax.tree.leaves(st) if np.ndim(x) and np.shape(x)[0] == 4]
    assert len(per_env) == 2
    assert {id(x) for x in per_env} == {id(st.ep_int), id(st.ep_ext)}


def test_linear_decay_halves_beta_at_half_budget(start: Any) -> None:
    cfg = _cfg(decay="linear")
    st = init(cfg, jax.random.key(1), SPEC, **SIZES)
    r, _ = _call(st, _batch(start, start), [False, False], t=SPEC.total_steps // 2)
    np.testing.assert_allclose(r, BETA / 2 / math.sqrt(2), rtol=1e-6)
    r, _ = _call(st, _batch(start, start), [False, False], t=SPEC.total_steps)
    np.testing.assert_array_equal(r, 0.0)


def test_zero_beta_pays_nothing_and_calls_repeat(start: Any, picked: Any) -> None:
    st = init(_cfg(beta=0.0), jax.random.key(1), SPEC, **SIZES)
    nxt = _batch(start, picked)
    r1, s1 = _call(st, nxt, [False, False])
    r2, s2 = _call(st, nxt, [False, False])
    np.testing.assert_array_equal(r1, 0.0)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(s1.table, s2.table)
    assert s1.table.dtype == jnp.int32
    assert int(s1.table.sum()) == 2


def test_simhash_key_separates_one_map_tile(env: Any, start: Any) -> None:
    proj = init(_cfg(key="simhash"), jax.random.key(3), SPEC, **SIZES).proj
    pos = start.player_position + jnp.array([1, 0])
    tile = int(start.map[pos[0], pos[1]])
    new = (
        BlockType.STONE.value
        if tile != BlockType.STONE.value
        else BlockType.WATER.value
    )
    obs = env.get_obs(start)
    moved = env.get_obs(start.replace(map=start.map.at[pos[0], pos[1]].set(new)))
    assert not np.array_equal(obs, moved)
    k = jax.jit(simhash_key, static_argnums=2)
    assert int(k(obs, proj, 20)) == int(k(obs + 0.0, proj, 20))
    assert int(k(obs, proj, 20)) != int(k(moved, proj, 20))
    batch = k(jnp.stack([obs, moved]), proj, 20)
    np.testing.assert_array_equal(batch, [k(obs, proj, 20), k(moved, proj, 20)])


def test_simhash_counts_observations(env: Any, start: Any) -> None:
    cfg = _cfg(key="simhash")
    st = init(cfg, jax.random.key(3), SPEC, **SIZES)
    obs = jnp.stack([env.get_obs(start)] * 2)
    r, st = jax.jit(intrinsic_reward)(
        st, None, _batch(start, start), obs, jnp.array([False, False]), jnp.int32(0)
    )
    np.testing.assert_allclose(r, BETA / math.sqrt(2), rtol=1e-6)
    assert int(st.table[simhash_key(obs[0], st.proj, 20)]) == 2


def test_metrics_match_numpy() -> None:
    st = init(_cfg(), jax.random.key(4), SPEC, **SIZES)
    keys = np.random.default_rng(5).integers(0, 2**20, 3000)
    table = st.table.at[keys].add(1)
    got = jax.jit(metrics)(st.replace(table=table))
    ref = np.zeros(2**20, np.int64)
    np.add.at(ref, keys, 1)
    distinct = np.count_nonzero(ref)
    assert set(got) == CLASSIC_METRICS
    assert int(got["count/distinct_keys"]) == distinct
    np.testing.assert_allclose(got["count/occupancy"], distinct / 2**20, rtol=1e-6)
    np.testing.assert_allclose(got["count/collision_est"], distinct / 2**21, rtol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        {"beta": -0.1},
        {"beta": math.nan},
        {"granularity": "fine"},
        {"decay": "cosine"},
        {"key": "rnd"},
        {"max_occupancy": 0.0},
        {"max_occupancy": 1.5},
        {"max_occupancy": math.nan},
    ],
)
def test_config_refuses_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(bad))):
        _cfg(**bad)


def test_state_round_trips_save_state(tmp_path: Path, start: Any) -> None:
    cfg = _cfg(decay="linear")
    st = init(cfg, jax.random.key(1), SPEC, **SIZES)
    _, st = _call(st, _batch(start, start), [False, False], t=40)
    save_state(tmp_path, st)
    back = load_state(tmp_path, init(cfg, jax.random.key(9), SPEC, **SIZES))
    for got, want in zip(jax.tree.leaves(back), jax.tree.leaves(st), strict=True):
        np.testing.assert_array_equal(got, want)
    assert (back.cfg, back.total_steps) == (cfg, SPEC.total_steps)


def _script(st: Any, steps: list[tuple[Any, list[bool], int]]) -> list[Any]:
    """Run (next state, done, t) calls; the metrics after each call."""
    out = []
    for nxt, done, t in steps:
        _, st = _call(st, nxt, done, t=t)
        out.append(jax.jit(metrics)(st))
    return out


def test_metrics_cover_the_update_window(start: Any, picked: Any) -> None:
    """Two calls make an update (batch 4); the third starts a new window."""
    st = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    first, second, third = _script(
        st,
        [
            (_batch(start, start), [False, False], 0),
            (_batch(picked, start), [False, False], 2),
            (_batch(start, start), [False, True], 4),
        ],
    )
    assert set(first) == CLASSIC_METRICS
    assert all(np.shape(v) == () for v in first.values())
    assert all(
        np.isfinite(float(v)) for m in (first, second, third) for v in m.values()
    )
    window0 = BETA * np.array([2**-0.5, 2**-0.5, 1.0, 3**-0.5])
    assert (int(second["count/distinct_keys"]), int(second["count/new_keys"])) == (2, 2)
    np.testing.assert_allclose(second["count/r_int_mean"], window0.mean(), rtol=1e-6)
    np.testing.assert_allclose(
        second["count/r_int_by_floor/0"], window0.mean(), rtol=1e-6
    )
    # the new window: one key seen again, one done step (paid 0, not on a floor)
    assert (int(third["count/distinct_keys"]), int(third["count/new_keys"])) == (2, 0)
    np.testing.assert_allclose(third["count/r_int_mean"], BETA / 4, rtol=1e-6)
    np.testing.assert_allclose(third["count/r_int_by_floor/0"], BETA / 2, rtol=1e-6)
    assert float(third["count/beta_t"]) == pytest.approx(BETA)
    assert int(third["count/max_key_floor"]) == 0


def test_r_int_share_is_over_finished_episodes(start: Any) -> None:
    """Env 0 earns 1.0 extrinsic and two bonuses, then ends; the share is kept
    through a window where no episode ends."""
    st = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    live = [False, False]
    got = _script(
        st,
        [
            (_batch(start, start, returns=(0.0, 0.0)), live, 0),
            (_batch(start, start, returns=(1.0, 0.0)), live, 2),
            (_batch(start, start), [True, False], 4),
            (_batch(start, start, returns=(0.0, 0.0)), live, 8),
        ],
    )
    assert float(got[1]["count/r_int_share"]) == 0.0  # no episode ended yet
    bonus = BETA * (2**-0.5 + 4**-0.5)  # env 0's two live steps; both envs on a
    share = bonus / (bonus + 1.0)
    np.testing.assert_allclose(got[2]["count/r_int_share"], share, rtol=1e-5)
    np.testing.assert_allclose(got[3]["count/r_int_share"], share, rtol=1e-5)


@pytest.fixture(scope="module")
def craftax_start() -> Any:
    """Craftax (full game) start state."""
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    return env.reset(jax.random.key(0), env.default_params)[1]


def test_bonus_by_floor_on_craftax(craftax_start: Any) -> None:
    assert StaticEnvParams().num_levels == count_bonus.N_FLOORS
    spec = HookSpec(
        n_envs=2, obs_dim=8268, n_actions=43, n_achievements=67, total_steps=1000
    )
    st = init(_cfg(), jax.random.key(1), spec, batch=4, game="craftax")
    deep = craftax_start.replace(player_level=jnp.int32(2))
    obs = jnp.zeros((2, 8268))
    r, st = jax.jit(intrinsic_reward)(
        st, None, _batch(deep, craftax_start), obs, jnp.array([False, False]), 0
    )
    got = jax.jit(metrics)(st)
    floors = {f"count/r_int_by_floor/{f}" for f in range(9)}
    assert floors <= set(got)
    by_floor = [float(got[f"count/r_int_by_floor/{f}"]) for f in range(9)]
    np.testing.assert_allclose(by_floor, [r[1], 0, r[0], *[0] * 6], rtol=1e-6)
    assert int(got["count/max_key_floor"]) == 2


def test_one_update_logs_every_metric_and_keys_never_vanish() -> None:
    trainer = METHODS["count-bonus-ppo-rnn"]
    cfg = {
        "env": CLASSIC,
        "envs": 8,
        "num_steps": 8,
        "minibatches": 2,
        "width": 16,
        "rnn_width": 16,
        "steps": 3 * 64,
        **KNOBS,
    }
    _, m = trainer.train_chunk(trainer.init(cfg, jax.random.key(0)), 3)
    count = {k: np.asarray(v) for k, v in m.items() if k.startswith("count/")}
    assert set(count) == CLASSIC_METRICS
    assert all(v.shape == (3,) and np.isfinite(v).all() for v in count.values())
    distinct = count["count/distinct_keys"]
    assert (np.diff(distinct) >= 0).all()
    np.testing.assert_array_equal(count["count/new_keys"], np.diff(distinct, prepend=0))
    assert (count["count/r_int_mean"] > 0).all()


def test_the_first_visit_writes_the_fields_of_the_slot(start: Any, picked: Any) -> None:
    st = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    _, st = _call(st, _batch(picked, start), [False, False])
    _, st = _call(st, _batch(start, start), [False, False])
    for s in (start, picked):
        np.testing.assert_array_equal(st.seen[_key(s)], fields(s, "recipe"))


def test_top_keys_lists_the_most_and_least_visited(
    start: Any, picked: Any, tmp_path: Path
) -> None:
    st = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    for nxt in (_batch(picked, start), _batch(start, start)):
        _, st = _call(st, nxt, [False, False])
    top = top_keys(st)
    assert top["distinct_keys"] == 2
    assert [(e["slot"], e["count"]) for e in top["most"]] == [
        (_key(start), 3),
        (_key(picked), 1),
    ]
    assert [e["slot"] for e in top["least"]] == [_key(picked), _key(start)]
    assert top["least"][0]["fields"] == dict(
        zip(CLASSIC_FIELDS, np.asarray(fields(picked, "recipe")).tolist(), strict=True)
    )
    many = st.replace(table=st.table.at[jnp.arange(200)].add(jnp.arange(1, 201)))
    top = top_keys(many, n=50)
    assert len(top["most"]) == len(top["least"]) == 50
    assert top["most"][0]["count"] == 200
    assert top["least"][0]["count"] == 1
    path = write_top_keys(tmp_path, 10, st)
    write_top_keys(tmp_path, 20, many)
    assert path == tmp_path / "eval" / "count_top_keys.json"
    saved = json.loads(path.read_text())
    assert set(saved) == {"10", "20"}
    assert saved["10"] == top_keys(st)


# --- abort rules -----------------------------------------------------------------


def test_the_guards_stay_zero_until_the_table_breaks(start: Any) -> None:
    st = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    _, st = _call(st, _batch(start, start), [False, False])
    m = jax.jit(metrics)
    ok = m(st)
    assert float(ok["count/table_full"]) == float(ok["count/negative_counts"]) == 0.0
    k = st.table.shape[0]
    quarter = st.replace(table=jnp.zeros_like(st.table).at[: k // 4].set(1))
    assert float(m(quarter)["count/table_full"]) == 0.0  # occupancy 0.25 is allowed
    full = quarter.replace(table=quarter.table.at[k // 4].set(1))
    assert np.isnan(float(m(full)["count/table_full"]))
    empty = int(np.flatnonzero(np.asarray(st.table) == 0)[0])
    decremented = m(st.replace(table=st.table.at[empty].add(-1)))
    assert np.isnan(float(decremented["count/negative_counts"]))
    assert float(decremented["count/table_full"]) == 0.0


def test_the_method_file_aborts_on_the_bonus_and_the_guards() -> None:
    """Plus both controls' rules: each variant still reports its base's metrics."""
    finite = ["count/r_int_mean", "count/table_full", "count/negative_counts"]
    ppo, pqn = (load_method(m).abort or {} for m in ("ppo-rnn", "pqn-rnn"))
    assert load_method("count-bonus").abort == {
        "finite": ppo["finite"] + pqn["finite"] + finite,
        "permanent": ["count/table_full"],  # a retry fills it again
        "limits": ppo["limits"] + pqn["limits"],
        "flags": ppo["flags"] + pqn["flags"],
    }


def _spec(**over: Any) -> RunSpec:
    """A 2-update Classic PPO-RNN variant run with the method file's abort rules."""
    name = "count-bonus-ppo-rnn"
    cfg = {
        "method": name,
        "env": CLASSIC,
        "envs": 8,
        "num_steps": 8,
        "minibatches": 2,
        "update_epochs": 1,
        "rnn_width": 16,
        **KNOBS,
    } | over
    abort = load_method("count-bonus").abort
    return RunSpec(
        name,
        "count-bonus",
        "0.128k",
        CLASSIC,
        0,
        cfg,
        config_hash(cfg),
        "x",
        "final",
        abort,
    )


def test_a_full_table_fails_the_run(run_store: Any) -> None:
    spec = _spec(log2_k=1, width=19)  # 2 slots: the first key fills half the table
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FAILED"
    tags = run.data.tags
    assert (tags["abort/metric"], tags["abort/update"]) == ("count/table_full", "1")
    assert tags["abort/guard"] == "count/table_full"  # skipped, not retried


def test_a_nan_bonus_fails_the_run(
    run_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(count_bonus, "_beta_t", lambda state, t: jnp.float32(jnp.nan))
    spec = _spec(width=18)  # a config no other test compiles: the patch is traced
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FAILED"
    tags = run.data.tags
    assert tags["abort/update"] == "1"
    # the NaN reaches the losses in the same update; either check stops the run
    assert tags["abort/metric"] in {*LOSS_COLUMNS, "count/r_int_mean"}


def _fill(st: Any, n: int) -> Any:
    """``st`` with ``n`` uniform random insertions into its table."""
    k = st.table.shape[0]
    keys = np.random.default_rng(6).integers(0, k, n)
    return st.replace(table=st.table.at[keys].add(1))


def test_the_simhash_ablation_has_room_for_a_new_code_every_step() -> None:
    """1M steps of distinct codes (a 1M run where every observation is new) stay
    below the ablation's threshold; the progress-key table would abort on them."""
    ablation = init(_cfg(**SIMHASH), jax.random.key(1), SPEC, **SIZES)
    m = jax.jit(metrics)(_fill(ablation, 10**6))
    assert float(m["count/table_full"]) == 0.0
    assert float(m["count/occupancy"]) < 0.06
    progress = init(_cfg(), jax.random.key(1), SPEC, **SIZES)
    assert np.isnan(float(jax.jit(metrics)(_fill(progress, 10**6))["count/table_full"]))


def test_the_threshold_is_the_config_knob() -> None:
    st = init(_cfg(max_occupancy=0.5), jax.random.key(1), SPEC, **SIZES)
    k = st.table.shape[0]
    half = st.replace(table=st.table.at[: k // 2].set(1))
    assert float(jax.jit(metrics)(half)["count/table_full"]) == 0.0
    more = half.replace(table=half.table.at[k // 2].set(1))
    assert np.isnan(float(jax.jit(metrics)(more)["count/table_full"]))


def test_simhash_keeps_no_side_table(env: Any, start: Any) -> None:
    st = init(_cfg(**SIMHASH), jax.random.key(3), SPEC, **SIZES)
    assert st.seen.shape == (0, len(CLASSIC_FIELDS))
    obs = jnp.stack([env.get_obs(start)] * 2)
    r, st = jax.jit(intrinsic_reward)(
        st, None, _batch(start, start), obs, jnp.array([False, False]), jnp.int32(0)
    )
    np.testing.assert_allclose(r, BETA / math.sqrt(2), rtol=1e-6)
    assert st.seen.shape == (0, len(CLASSIC_FIELDS))
    top = top_keys(st)
    assert [(e["count"], e["fields"]) for e in top["most"]] == [(2, None)]
