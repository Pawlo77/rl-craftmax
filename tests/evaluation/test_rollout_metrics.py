"""Tests for the offline replay of stored eval files: checks, records, frames, CLI."""

import math
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pandas as pd
import pytest

from wut_murl_craftax.analysis import obstacles
from wut_murl_craftax.evaluation.coverage import (
    coverage_summary,
    init_seen,
    update_seen,
)
from wut_murl_craftax.evaluation.episode_records import episode_record, trace_of
from wut_murl_craftax.evaluation.rollout_metrics import (
    EPISODE_COLUMNS,
    HOARD_CAPS,
    IDENTITY,
    INVENTORY_COLUMNS,
    MISMATCH_COLUMNS,
    RUN_IDENTITY,
    SKILL_ENDS_COLUMNS,
    Identity,
    Replayer,
    ReplayMismatchError,
    RunRef,
    at_cap,
    episode_columns,
    file_tables,
    main,
    mismatch_path,
    out_paths,
    phase_runs,
    plan_jobs,
    process_file,
    replay_file,
)
from wut_murl_craftax.training.dyna_q import MODEL_GRAPH_FILE, MODEL_GRAPH_TYPES
from wut_murl_craftax.training.progress_key import CAPS
from wut_murl_craftax.training.skills import REASONS, SKILLS, decode_obs
from wut_murl_craftax.training.smdp import top_key, write_update_rows
from wut_murl_craftax.visualisation.eval_rollouts import (
    EvalRun,
    load_eval,
    replay_episode,
)

GAME = "classic"
UPDATE = 2
ID = Identity("run-x", "1m", 3)
REPLAY_SEEDS = (10000, 10001)
"""The seeds of the `skill_run` fixture's eval files (conftest)."""


@pytest.fixture(scope="module")
def rp(skill_run: Path) -> Replayer:
    """One compiled replayer of the tiny skills run, shared by the module."""
    return Replayer(EvalRun.load(skill_run))


@pytest.fixture(scope="module")
def stochastic(skill_run: Path) -> dict[str, np.ndarray]:
    return load_eval(skill_run, UPDATE)


@pytest.fixture(scope="module")
def tables(rp: Replayer, stochastic: dict[str, np.ndarray]) -> Any:
    return file_tables(rp, stochastic, UPDATE, "stochastic", ID)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, list | np.ndarray) or isinstance(b, list | np.ndarray):
        return [_plain(x) for x in a] == [_plain(x) for x in b]
    return bool(a == b)


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {
            k: np.asarray(v).tolist() if v is not None else None for k, v in x.items()
        }
    return np.asarray(x).tolist()


@jax.jit
def _scan_seen(seen: jax.Array, states: Any) -> jax.Array:
    return jax.lax.scan(lambda s, x: (update_seen(s, x, GAME), None), seen, states)[0]


def _reference(run: EvalRun, seed: int, actions: np.ndarray) -> dict[str, Any]:
    """One episode replayed alone, its record and coverage built from all its states."""
    ref = replay_episode(run, seed, actions)
    n = len(ref.reward)
    states = jax.tree.map(np.asarray, ref.states)
    trace = trace_of(states, GAME)
    first = jax.tree.map(lambda x: x[0], ref.states)
    rest = jax.tree.map(lambda x: x[1:], ref.states)
    seen = _scan_seen(init_seen(first, GAME), rest)
    row = episode_record(ref.events, actions[:n], trace, game=GAME).row()
    return row | coverage_summary(seen, trace.position, trace.floor)


def _key(cfg: Any, obs: Any) -> int:
    """`top_key` of one observation at the run's settings."""
    v = decode_obs(jnp.asarray(obs))
    return int(top_key(v, cfg.granularity, cfg.log2_rows, low_flags=cfg.low_flags))


def test_the_batched_replay_equals_one_episode_replayed_alone(
    rp: Replayer, stochastic: dict[str, np.ndarray], tables: Any
) -> None:
    """Every record and coverage column (trace fields too) equals the reference."""
    for i, seed in enumerate(REPLAY_SEEDS):
        want = _reference(rp.run, seed, stochastic["actions"][i])
        got = tables.episodes.iloc[i]
        diff = [k for k, v in want.items() if not _same(got[k], v)]
        assert not diff, diff


def test_the_replayed_trace_rows_equal_the_states_of_a_lone_replay(
    rp: Replayer, stochastic: dict[str, np.ndarray]
) -> None:
    """Every trace field of every state, so stall starts read the right row."""
    fr = replay_file(rp, stochastic)
    for i, seed in enumerate(REPLAY_SEEDS):
        states = replay_episode(rp.run, seed, stochastic["actions"][i]).states
        want = trace_of(jax.tree.map(np.asarray, states), GAME)
        for name, w, g in zip(want._fields, want, fr.traces[i], strict=True):
            np.testing.assert_array_equal(g, w, err_msg=name)


def test_coverage_stops_at_the_stored_length_even_if_actions_go_on(
    rp: Replayer, stochastic: dict[str, np.ndarray]
) -> None:
    """A done seed keeps stepping in the batch; its tiles stop at its length."""
    cut = stochastic["length"].copy()
    cut[0] //= 2
    short = stochastic | {"length": cut}  # its later actions stay in the file
    got = file_tables(rp, short, UPDATE, "stochastic", ID).episodes.iloc[0]
    want = _reference(rp.run, REPLAY_SEEDS[0], stochastic["actions"][0, : cut[0]])
    for k in ("coverage/overworld", "new_tiles_per_100", "revisit_rate"):
        assert got[k] == want[k], k


def test_the_skill_key_is_read_at_the_decision_not_after_it(
    rp: Replayer, classic_start: Any
) -> None:
    from craftax.craftax_classic.renderer import render_craftax_symbolic

    s = classic_start
    rich = s.replace(inventory=s.inventory.replace(wood=jnp.asarray(3, jnp.int32)))
    obs, obs2 = render_craftax_symbolic(s), render_craftax_symbolic(rich)
    cfg = rp.skills
    keys = [_key(cfg, o) for o in (obs, obs2)]
    assert keys[0] != keys[1]
    zero = jnp.int32(0)
    got = rp._skill_end((obs, zero, jnp.float32(0), obs2, zero, zero))
    assert int(got["key"]) == keys[0]


def _family(column: str) -> str:
    """The `EPISODE_COLUMNS` entry a column belongs to (exact name or family)."""
    if column in EPISODE_COLUMNS:
        return column
    return next(f for f in EPISODE_COLUMNS if f.endswith("/") and column.startswith(f))


def test_the_table_columns_are_episode_columns_in_contract_order(
    tables: Any,
) -> None:
    """EPISODE_COLUMNS lists the contract's families in order; files lack config_ref."""
    want = episode_columns(GAME)
    assert want[: len(IDENTITY)] == IDENTITY
    assert list(tables.episodes.columns) == [c for c in want if c != "config_ref"]
    groups = [
        f
        for i, f in enumerate(map(_family, want))
        if i == 0 or f != _family(want[i - 1])
    ]
    present = [f for f in EPISODE_COLUMNS if f in groups]
    assert groups == present  # each family once, in contract order
    assert set(EPISODE_COLUMNS) - set(groups) == {"adv/"}  # Classic has no magic
    assert {"kills/zombie", "kills/skeleton", "kills/passive"} <= set(want)


def test_one_row_per_seed_with_the_identity_and_the_replay_fields(
    tables: Any, stochastic: dict[str, np.ndarray]
) -> None:
    ep = tables.episodes
    assert tuple(ep.columns[: len(RUN_IDENTITY)]) == RUN_IDENTITY
    assert "config_ref" not in ep.columns  # the phase export adds it
    assert ep["eval_seed"].tolist() == list(REPLAY_SEEDS)
    assert (ep["method"] == "smdp-skills").all()
    assert (ep[["run_id", "budget", "seed"]].iloc[0].tolist()) == ["run-x", "1m", 3]
    assert (ep["update"] == UPDATE).all()
    assert (ep["mode"] == "stochastic").all()
    assert ep["length"].tolist() == stochastic["length"].tolist()
    for col in ("gathered/wood", "used/wood", "wasted_crafts", "coverage/overworld"):
        assert ep[col].notna().all(), col
    assert ep["revisit_rate"].between(0, 1).all()


def test_a_stored_event_the_replay_does_not_reproduce_raises(
    rp: Replayer, stochastic: dict[str, np.ndarray]
) -> None:
    bad = dict(stochastic)
    bad["events/health_delta"] = stochastic["events/health_delta"].copy()
    bad["events/health_delta"][1, 3] += 1
    where = "run run-x update 2 mode stochastic"
    with pytest.raises(
        ReplayMismatchError, match=f"{where} seed 10001 differs.*health_delta"
    ):
        file_tables(rp, bad, UPDATE, "stochastic", ID)


def _stale(stochastic: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A copy whose stored kills and damage differ from what the replay computes."""
    bad = dict(stochastic)
    for k in ("events/damage_dealt", "events/kills"):
        bad[k] = stochastic[k].copy()
    bad["events/damage_dealt"][1, 3] += 2
    bad["events/kills"][0, 5, 0] += 1
    return bad


def test_recompute_takes_the_replayed_events_and_records_each_mismatch(
    rp: Replayer, stochastic: dict[str, np.ndarray], tables: Any
) -> None:
    got = file_tables(rp, _stale(stochastic), UPDATE, "stochastic", ID, recompute=True)
    pd.testing.assert_frame_equal(got.episodes, tables.episodes)  # the true events
    m = got.mismatches.sort_values("eval_seed", ignore_index=True)
    assert tuple(m.columns) == MISMATCH_COLUMNS
    assert m[["run_id", "update", "mode"]].drop_duplicates().values.tolist() == [
        ["run-x", UPDATE, "stochastic"]
    ]
    assert m[["eval_seed", "field", "steps", "first_step"]].values.tolist() == [
        [10000, "kills", 1, 5],
        [10001, "damage_dealt", 1, 3],
    ]
    kills, damage = m["old"] - m["new"]
    assert (kills, damage) == (1.0, 2.0)  # the stored value minus the replayed one


def test_recompute_of_an_agreeing_file_has_no_mismatch(
    rp: Replayer, stochastic: dict[str, np.ndarray], tables: Any
) -> None:
    got = file_tables(rp, stochastic, UPDATE, "stochastic", ID, recompute=True)
    assert got.mismatches.empty
    assert tuple(got.mismatches.columns) == MISMATCH_COLUMNS
    pd.testing.assert_frame_equal(got.episodes, tables.episodes)
    assert tables.mismatches.empty


def test_recompute_raises_for_a_field_the_rules_did_not_change(
    rp: Replayer, stochastic: dict[str, np.ndarray]
) -> None:
    bad = _stale(stochastic)  # kills and damage_dealt differ: allowed
    bad["events/health_delta"] = stochastic["events/health_delta"].copy()
    bad["events/health_delta"][1, 3] += 1
    where = "run run-x update 2 mode stochastic"
    with pytest.raises(
        ReplayMismatchError, match=rf"{where} seed 10001 differs.*\['health_delta'\]"
    ):
        file_tables(rp, bad, UPDATE, "stochastic", ID, recompute=True)


def test_recompute_still_raises_on_a_skill_end_mismatch(
    rp: Replayer, stochastic: dict[str, np.ndarray]
) -> None:
    steps = stochastic["extras/steps_in_skill"].copy()
    steps[0, int(np.flatnonzero(steps[0] > 0)[0])] = 0
    bad = stochastic | {"extras/steps_in_skill": steps}
    with pytest.raises(ReplayMismatchError, match="ends skills"):
        file_tables(rp, bad, UPDATE, "stochastic", ID, recompute=True)


def test_process_recompute_writes_the_mismatch_csv_before_the_done_marker(
    rp: Replayer, stochastic: dict[str, np.ndarray], tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "wut_murl_craftax.visualisation.eval_rollouts.load_eval",
        lambda *a, **k: _stale(stochastic),
    )
    ident = Identity("run-x", "1m", 3)
    assert process_file(rp, ident, UPDATE, "stochastic", tmp_path, recompute=True)
    csv = mismatch_path(tmp_path, "run-x", UPDATE, "stochastic")
    assert (
        csv.parent
        == out_paths(tmp_path, "run-x", UPDATE, "stochastic")["episodes"].parent
    )
    m = pd.read_csv(csv)
    assert tuple(m.columns) == MISMATCH_COLUMNS
    assert sorted(m["field"]) == ["damage_dealt", "kills"]
    with pytest.raises(ReplayMismatchError, match="differs"):  # strict is the default
        process_file(rp, ident, UPDATE, "greedy", tmp_path)


def test_a_rerun_without_mismatches_removes_a_stale_csv(
    rp: Replayer, tmp_path: Path
) -> None:
    ident = Identity("run-x", "1m", 3)
    stale = mismatch_path(tmp_path, "run-x", UPDATE, "stochastic")
    stale.parent.mkdir(parents=True)
    stale.write_text("old\n")
    assert process_file(rp, ident, UPDATE, "stochastic", tmp_path, recompute=True)
    assert not stale.exists()


def test_a_stored_skill_start_the_replay_does_not_end_raises(
    rp: Replayer, stochastic: dict[str, np.ndarray]
) -> None:
    steps = stochastic["extras/steps_in_skill"].copy()
    at = int(np.flatnonzero(steps[0] > 0)[0])  # a running skill's step
    steps[0, at] = 0
    with pytest.raises(ReplayMismatchError, match=f"seed 10000 ends skills.*step {at}"):
        file_tables(
            rp, stochastic | {"extras/steps_in_skill": steps}, UPDATE, "stochastic", ID
        )


def test_skill_ends_tile_each_episode_with_the_decision_key_and_reason(
    rp: Replayer, stochastic: dict[str, np.ndarray], tables: Any
) -> None:
    ends = tables.skill_ends
    assert tuple(ends.columns) == SKILL_ENDS_COLUMNS  # not dyna-q: no pair_count
    cfg = rp.skills
    for i, seed in enumerate(REPLAY_SEEDS):
        n = int(stochastic["length"][i])
        e = ends[ends["eval_seed"] == seed]
        steps, skill = (
            stochastic[f"extras/{k}"][i, :n] for k in ("steps_in_skill", "skill")
        )
        assert e["start"].tolist() == np.flatnonzero(steps == 0).tolist()
        assert e["end"].tolist() == [*e["start"].tolist()[1:], n]
        assert e["skill"].tolist() == [SKILLS[k].name for k in skill[e["start"]]]
        assert e["reason"].iloc[-1] == "episode_end"
        assert set(e["reason"].iloc[:-1]) <= set(REASONS) - {"running", "episode_end"}
        obs = replay_episode(rp.run, seed, stochastic["actions"][i]).obs
        assert e["key"].tolist() == [_key(cfg, obs[s]) for s in e["start"]]
    assert (ends[["update", "env_steps"]].iloc[0].tolist()) == [
        UPDATE,
        rp.run.env_steps(UPDATE),
    ]


def test_dyna_rows_get_the_count_of_the_checkpoint_model_graph(
    rp: Replayer, stochastic: dict[str, np.ndarray], tables: Any, skill_run: Path
) -> None:
    first = tables.skill_ends.iloc[0]
    graph = skill_run / "eval" / MODEL_GRAPH_FILE
    pair = {"key": [first["key"]], "skill": [first["skill"]]}
    write_update_rows(graph, 0, pair | {"count": [99]}, MODEL_GRAPH_TYPES)
    write_update_rows(graph, UPDATE, pair | {"count": [7]}, MODEL_GRAPH_TYPES)
    try:
        ends = file_tables(rp, stochastic, UPDATE, "stochastic", ID).skill_ends
    finally:
        graph.unlink()
    assert tuple(ends.columns) == (*SKILL_ENDS_COLUMNS, "pair_count")
    same = (ends["key"] == first["key"]) & (ends["skill"] == first["skill"])
    assert (ends.loc[same, "pair_count"] == 7).all()
    assert (ends.loc[~same, "pair_count"] == 0).all()


def test_only_materials_with_a_cap_above_1_count_as_capped(
    classic_start: Any, craftax_start: Any
) -> None:
    """R98: one sapling, arrow, torch, book, potion or gem is not hoarding."""
    assert set(HOARD_CAPS) == {"wood", "stone", "coal", "iron", "diamond"}

    def inv(s: Any, **kw: int) -> Any:
        return s.replace(
            inventory=s.inventory.replace(**{
                k: jnp.full_like(getattr(s.inventory, k), v) for k, v in kw.items()
            })
        )

    c = classic_start
    assert not bool(at_cap(inv(c, sapling=1, wood=3, stone=2)))
    assert bool(at_cap(inv(c, wood=4)))
    s = craftax_start
    ones = dict.fromkeys(
        ("sapling", "sapphire", "ruby", "arrows", "torches", "books"), 1
    )
    assert not bool(at_cap(inv(s, potions=1, **ones)))
    assert bool(at_cap(inv(s, diamond=3)))


def test_eval_inventory_counts_the_states_with_a_material_at_its_cap(
    rp: Replayer, stochastic: dict[str, np.ndarray], tables: Any
) -> None:
    inv = tables.inventory
    assert tuple(inv.columns) == INVENTORY_COLUMNS
    assert inv["steps"].tolist() == stochastic["length"].tolist()
    for i, seed in enumerate(REPLAY_SEEDS):
        s = replay_episode(rp.run, seed, stochastic["actions"][i]).states.inventory
        n = int(stochastic["length"][i])
        capped = np.zeros(n + 1, bool)
        for m, cap in CAPS.items():
            if hasattr(s, m) and cap > 1:
                capped |= np.asarray(getattr(s, m)) >= cap
        assert inv["capped_steps"].iloc[i] == capped[:n].sum()


def test_the_frame_columns_are_the_analysis_ones() -> None:
    assert SKILL_ENDS_COLUMNS == obstacles.SKILL_ENDS_COLUMNS
    assert INVENTORY_COLUMNS == obstacles.INVENTORY_COLUMNS


def test_skill_extras_without_a_skills_policy_raise(
    rp: Replayer, stochastic: dict[str, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrapper policy the replayer does not know would silently drop skill_ends."""
    monkeypatch.setattr(rp, "skills", None)
    with pytest.raises(ReplayMismatchError, match="extras/skill"):
        file_tables(rp, stochastic, UPDATE, "stochastic", ID)


def _store(tmp_path: Path) -> tuple[str, mlflow.MlflowClient]:
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    return uri, mlflow.MlflowClient(uri)


def _run_in_store(
    client: mlflow.MlflowClient, src: Path, root: Path, name: str, phases: str
) -> str:
    """A finished run with the runner's tags whose artifacts are a copy of ``src``."""
    exp = client.get_experiment_by_name(name) or None
    exp_id = (
        exp.experiment_id
        if exp
        else client.create_experiment(name, artifact_location=(root / name).as_uri())
    )
    run = client.create_run(exp_id, run_name=f"{name}-run")
    tags = {"phases": phases, "method": "smdp-skills", "budget": "1m", "seed": "3"}
    for k, v in tags.items():
        client.set_tag(run.info.run_id, k, v)
    shutil.copytree(src, Path(run.info.artifact_uri.removeprefix("file://")))
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def test_process_writes_once_logs_once_and_skips_after(
    rp: Replayer, tmp_path: Path
) -> None:
    _, client = _store(tmp_path)
    exp = client.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    run_id = client.create_run(exp).info.run_id
    ident, out = Identity(run_id, "1m", 3), tmp_path / "out"
    assert process_file(rp, ident, UPDATE, "stochastic", out, client)
    paths = out_paths(out, run_id, UPDATE, "stochastic")
    assert all(p.is_file() for p in paths.values())
    got = pd.read_parquet(paths["episodes"])
    assert got["eval_seed"].tolist() == list(REPLAY_SEEDS)
    assert got["stalls"].map(len).sum() == got["n_stalls"].sum()
    logged = {a.path for a in client.list_artifacts(run_id, "eval_tables")}
    assert logged == {f"eval_tables/{p.name}" for p in paths.values()}
    key = "eval/stochastic/revisit_rate"
    (m,) = client.get_metric_history(run_id, key)
    assert m.step == rp.run.env_steps(UPDATE)
    assert m.value == pytest.approx(got["revisit_rate"].mean())
    mtime = paths["episodes"].stat().st_mtime_ns
    assert not process_file(rp, ident, UPDATE, "stochastic", out, client)
    assert paths["episodes"].stat().st_mtime_ns == mtime
    assert len(client.get_metric_history(run_id, key)) == 1


def test_a_logging_cut_after_the_metrics_is_finished_without_duplicates(
    rp: Replayer, tmp_path: Path
) -> None:
    """A crash between the summaries and the done marker leaves a ``.logging`` one."""
    _, client = _store(tmp_path)
    exp = client.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    run_id = client.create_run(exp).info.run_id
    ident, out = Identity(run_id, "1m", 3), tmp_path / "out"
    process_file(rp, ident, UPDATE, "stochastic", out, client)
    done = out_paths(out, run_id, UPDATE, "stochastic")["episodes"].with_suffix(
        ".logged"
    )
    done.rename(done.with_suffix(".logging"))  # as if killed before the last write
    assert process_file(rp, ident, UPDATE, "stochastic", out, client)
    assert done.is_file()
    assert not done.with_suffix(".logging").exists()
    key = "eval/stochastic/revisit_rate"
    assert len(client.get_metric_history(run_id, key)) == 1


def test_phase_runs_select_by_tag_and_the_plan_puts_final_checkpoints_first(
    skill_run: Path, tmp_path: Path
) -> None:
    uri, client = _store(tmp_path)
    a = _run_in_store(
        client, skill_run, tmp_path, "tune-coarse", "tune-coarse,final-1m"
    )
    b = _run_in_store(client, skill_run, tmp_path, "final-1m", "final-1m")
    _run_in_store(client, skill_run, tmp_path, "smoke", "smoke")
    runs = phase_runs(uri, ["final-1m"])
    assert sorted(r.run_id for r in runs) == sorted([a, b])
    assert {(r.method, r.budget, r.seed) for r in runs} == {("smdp-skills", "1m", 3)}
    jobs = [
        (r.run_id, u, m) for r, u, m in plan_jobs(runs, "all", ("stochastic", "greedy"))
    ]
    assert len(jobs) == 8
    assert {u for _, u, _ in jobs[:4]} == {UPDATE}
    assert {u for _, u, _ in jobs[4:]} == {0}
    only = plan_jobs(runs, "final", ("greedy",))
    assert [(u, m) for _, u, m in only] == [(UPDATE, "greedy")] * 2


def test_the_cli_processes_a_phase_without_logging_and_resumes(
    skill_run: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    uri, client = _store(tmp_path)
    run_id = _run_in_store(client, skill_run, tmp_path, "final-1m", "final-1m")
    out = tmp_path / "out"
    argv = ["--phase", "final-1m", "--tracking-uri", uri, "--out", str(out)]
    main([*argv, "--updates", "2", "--mode", "greedy", "--no-log"])
    paths = out_paths(out, run_id, UPDATE, "greedy")
    assert all(p.is_file() for p in paths.values())
    assert not out_paths(out, run_id, 0, "greedy")["episodes"].exists()
    assert not client.list_artifacts(run_id, "eval_tables")
    assert pd.read_parquet(paths["episodes"])["method"][0] == "smdp-skills"
    mtimes = {p: p.stat().st_mtime_ns for p in paths.values()}
    caplog.set_level("INFO")
    main([*argv, "--updates", "2", "--mode", "greedy", "--no-log"])  # resumed
    assert "0 files to do" in caplog.text
    assert {p: p.stat().st_mtime_ns for p in paths.values()} == mtimes


def test_the_cli_skips_a_mismatching_file_and_exits_1(
    skill_run: Path, tmp_path: Path
) -> None:
    uri, client = _store(tmp_path)
    run_id = _run_in_store(client, skill_run, tmp_path, "final-1m", "final-1m")
    run_dir = Path(client.get_run(run_id).info.artifact_uri.removeprefix("file://"))
    bad = load_eval(run_dir, UPDATE, greedy=True)
    bad["events/health_delta"][0, 2] += 1
    np.savez_compressed(run_dir / "eval" / "greedy" / f"{UPDATE:05d}.npz", **bad)
    out = tmp_path / "out"
    argv = ["--run", run_id, "--tracking-uri", uri, "--out", str(out), "--no-log"]
    with pytest.raises(SystemExit, match=r"1 files failed.*mode greedy seed 10000"):
        main([*argv, "--updates", "2"])
    assert out_paths(out, run_id, UPDATE, "stochastic")["episodes"].is_file()
    assert not out_paths(out, run_id, UPDATE, "greedy")["episodes"].exists()


def test_the_cli_recomputes_a_mismatching_file_with_the_flag(
    skill_run: Path, tmp_path: Path
) -> None:
    uri, client = _store(tmp_path)
    run_id = _run_in_store(client, skill_run, tmp_path, "final-1m", "final-1m")
    run_dir = Path(client.get_run(run_id).info.artifact_uri.removeprefix("file://"))
    bad = load_eval(run_dir, UPDATE, greedy=True)
    bad["events/damage_dealt"][0, 2] += 1
    np.savez_compressed(run_dir / "eval" / "greedy" / f"{UPDATE:05d}.npz", **bad)
    out = tmp_path / "out"
    argv = ["--run", run_id, "--tracking-uri", uri, "--out", str(out), "--no-log"]
    main([*argv, "--updates", "2", "--mode", "greedy", "--recompute-events"])
    assert out_paths(out, run_id, UPDATE, "greedy")["episodes"].is_file()
    (row,) = pd.read_csv(mismatch_path(out, run_id, UPDATE, "greedy")).to_dict(
        "records"
    )
    assert (row["eval_seed"], row["field"], row["first_step"]) == (
        10000,
        "damage_dealt",
        2,
    )
    assert row["old"] - row["new"] == 1


def test_the_cli_flag_does_not_excuse_another_field(
    skill_run: Path, tmp_path: Path
) -> None:
    uri, client = _store(tmp_path)
    run_id = _run_in_store(client, skill_run, tmp_path, "final-1m", "final-1m")
    run_dir = Path(client.get_run(run_id).info.artifact_uri.removeprefix("file://"))
    bad = load_eval(run_dir, UPDATE, greedy=True)
    bad["events/health_delta"][0, 2] += 1
    np.savez_compressed(run_dir / "eval" / "greedy" / f"{UPDATE:05d}.npz", **bad)
    argv = ["--run", run_id, "--tracking-uri", uri, "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match=r"1 files failed.*health_delta"):
        main([
            *argv,
            "--no-log",
            "--updates",
            "2",
            "--mode",
            "greedy",
            "--recompute-events",
        ])


def test_plan_skips_runs_without_eval_files(tmp_path: Path) -> None:
    ref = RunRef("r", "r-name", "ppo", "1m", 0, tmp_path)
    assert plan_jobs([ref], "all", ("stochastic",)) == []
